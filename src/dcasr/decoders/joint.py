"""Joint CTC+AED one-pass beam search (plan §6.3 / Watanabe et al. [12]) — the best-WER
headline read-out of one hybrid model.

A label-synchronous beam driven by the AED head, where each partial hypothesis is also
scored by CTC via the CTC *prefix score* (Graves' forward algorithm over the blank/label
lattice), so the combined score is
    score(h) = (1 - ctc_weight) · logP_AED(h) + ctc_weight · logP_CTC-prefix(h)
The AED head supplies fluency; the CTC prefix score keeps hypotheses anchored to the audio
(no skips/repeats). ctc_weight=0 reduces this to a pure AED (attention) beam — the AED
read-out. The search itself is acoustic-only; the external LM enters as a SECOND PASS that
re-ranks the completed n-best (joint_beam_search_nbest -> lm_rescore), so it never affects the
per-step EOS/length decisions — the aed_beam_lm / joint_beam_lm cells (see lm_fusion.py).

Per-utterance (clear + correct; the offline paper decode is not latency-bound). The CTC
prefix scorer is validated against a brute-force alignment enumerator in the tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from dcasr.logging_utils import get_logger

logger = get_logger(__name__)

_LOGZERO = -1e10


class CTCPrefixScorer:
    """Incremental CTC prefix log-probability for growing label prefixes (one utterance).

    State per prefix is r [T, 2]: r[t,0]=log prob the prefix is emitted by frame t ending in
    a non-blank (its last label), r[t,1]=... ending in blank. `score` extends a prefix by each
    candidate label, returning the absolute CTC prefix log-prob of each extension + its state.
    Mirrors the ESPnet/Watanabe CTCPrefixScore recursion.
    """

    def __init__(self, logp: torch.Tensor, blank_id: int, eos_id: int):
        self.logp = logp                                   # [T, V+1] CTC log-probs (fp32)
        self.T = logp.shape[0]
        self.blank = blank_id
        self.eos = eos_id

    def initial_state(self) -> torch.Tensor:
        """State of the empty prefix: only the all-blank path exists (ends in blank)."""
        r = self.logp.new_full((self.T, 2), _LOGZERO)
        r[0, 1] = self.logp[0, self.blank]
        for t in range(1, self.T):
            r[t, 1] = r[t - 1, 1] + self.logp[t, self.blank]
        return r

    def score(self, prefix: list[int], cand_ids: torch.Tensor,
              r_prev: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """prefix (labels so far, no sos), cand_ids [C], r_prev [T,2] -> (scores [C], states [C,T,2])."""
        T, C = self.T, cand_ids.shape[0]
        out_len = len(prefix)
        # eos may sit outside the CTC class dim; its column is overwritten below, so clamp the gather
        xs = self.logp[:, cand_ids.clamp(max=self.logp.shape[1] - 1)]   # [T, C]
        r = self.logp.new_full((T, 2, C), _LOGZERO)
        if out_len == 0:
            r[0, 0] = xs[0]                                 # first label may start at frame 0
        # else: r[out_len-1] stays logzero (a new label can't share the last label's frame)

        r_sum = torch.logaddexp(r_prev[:, 0], r_prev[:, 1])   # [T] prob of the prefix at each t
        log_phi = r_sum.unsqueeze(1).expand(T, C)          # transition prob into the new label
        if out_len > 0:                                    # a candidate == last label needs a blank between
            eq_last = cand_ids == prefix[-1]
            if bool(eq_last.any()):
                log_phi = torch.where(eq_last.unsqueeze(0), r_prev[:, 1].unsqueeze(1), log_phi)

        start = max(out_len, 1)
        log_psi = r[start - 1, 0].clone()                  # [C] accumulates the prefix score
        for t in range(start, T):
            r[t, 0] = torch.logaddexp(r[t - 1, 0], log_phi[t - 1]) + xs[t]
            r[t, 1] = torch.logaddexp(r[t - 1, 0], r[t - 1, 1]) + self.logp[t, self.blank]
            log_psi = torch.logaddexp(log_psi, log_phi[t - 1] + xs[t])

        eos_mask = cand_ids == self.eos                    # ending the sequence = prob of exactly the prefix
        if bool(eos_mask.any()):
            log_psi = torch.where(eos_mask, r_sum[-1].expand(C), log_psi)
        return log_psi, r.permute(2, 0, 1).contiguous()    # [C], [C, T, 2]


@dataclass
class _Hyp:
    tokens: list[int]
    aed: float                                             # cumulative AED log-prob (incl. eos)
    ctc: float                                             # absolute CTC prefix log-prob (incl. eos)
    ctc_state: torch.Tensor | None = None
    score: float = field(default=0.0)                      # acoustic: (1-w)·AED + w·CTC + bonus·len


@torch.no_grad()
def joint_beam_search_nbest(ctc_head, aed_head, memory: torch.Tensor,
                            memory_lengths: torch.Tensor, *, beam_size: int = 10,
                            ctc_weight: float = 0.3, bos_id: int = 1, eos_id: int = 2,
                            pad_id: int = 3, blank_id: int | None = None,
                            max_len_ratio: float = 1.0, length_bonus: float = 0.0,
                            pre_beam: int | None = None,
                            nbest: int = 1) -> list[list[_Hyp]]:
    """Acoustic-only joint CTC+AED beam; returns per utterance the top-`nbest` COMPLETE
    hypotheses, sorted best-first, each retaining its component AED/CTC log-probs for
    second-pass LM rescoring (`lm_rescore`).

    score(h) = (1-ctc_weight)·AED + ctc_weight·CTC-prefix + length_bonus·len.
    NO external LM participates in the search — that is the point of rescoring: the LM never
    touches the per-step EOS/length decisions, so it cannot truncate or over-generate.
    ctc_weight=0 -> pure AED beam (ctc_head may be None). Run heads in eval(). Returns a
    per-utterance list of `_Hyp` (up to `nbest`).
    """
    B = memory.shape[0]
    V = aed_head.vocab_size
    if blank_id is None and ctc_head is not None:
        blank_id = ctc_head.blank_id
    pre = min(V, pre_beam if pre_beam is not None else max(2 * beam_size, 15))
    results: list[list[_Hyp]] = []

    for b in range(B):
        Tf = int(memory_lengths[b])
        mem_b = memory[b : b + 1, :Tf]                     # [1, Tf, d]
        mlen_b = memory_lengths[b : b + 1]
        use_ctc = ctc_weight > 0.0 and ctc_head is not None
        scorer = None
        if use_ctc:
            ctc_logp = ctc_head.log_probs(mem_b)[0]        # [Tf, V+1]
            scorer = CTCPrefixScorer(ctc_logp, blank_id, eos_id)
        init_state = scorer.initial_state() if use_ctc else None
        beam = [_Hyp(tokens=[], aed=0.0, ctc=0.0, ctc_state=init_state, score=0.0)]
        ended: list[_Hyp] = []
        max_steps = min(max(1, int(max_len_ratio * Tf)), Tf - 1, aed_head.max_decode_len)

        for _step in range(max_steps):
            if not beam:
                break
            ys_in = torch.tensor([[bos_id] + h.tokens for h in beam], device=memory.device)
            logits = aed_head.forward(mem_b.expand(len(beam), -1, -1),
                                      mlen_b.expand(len(beam)), ys_in)     # [nb, L, V]
            aed_logp = torch.log_softmax(logits[:, -1].float(), dim=-1)    # [nb, V] next-token
            aed_logp[:, bos_id] = _LOGZERO                 # non-emittable; finite so no 0·inf = NaN
            aed_logp[:, pad_id] = _LOGZERO
            eos_t = aed_logp.new_tensor([eos_id], dtype=torch.long)
            ext: list[tuple[float, bool, _Hyp]] = []          # all one-token expansions this step
            for i, h in enumerate(beam):
                ids = torch.unique(torch.cat([torch.topk(aed_logp[i], pre).indices, eos_t]))
                ids = ids[(ids != bos_id) & (ids != pad_id)]   # never expand specials (any weights)
                ctc_scores, ctc_states = (scorer.score(h.tokens, ids, h.ctc_state)
                                          if use_ctc else (None, None))
                for j in range(ids.shape[0]):
                    c = int(ids[j])
                    is_eos = c == eos_id
                    toks = h.tokens if is_eos else h.tokens + [c]   # eos ends (not emitted)
                    aed_c = h.aed + float(aed_logp[i, c])
                    ctc_c = float(ctc_scores[j]) if use_ctc else 0.0
                    total = ((1.0 - ctc_weight) * aed_c + ctc_weight * ctc_c
                             + length_bonus * len(toks))
                    ext.append((total, is_eos,
                                _Hyp(tokens=toks, aed=aed_c, ctc=ctc_c,
                                     ctc_state=(ctc_states[j] if (use_ctc and not is_eos) else None),
                                     score=total)))
            ext.sort(key=lambda e: e[0], reverse=True)         # global top-k over all expansions
            beam = []
            for _total, is_eos, hyp in ext[:beam_size]:
                (ended if is_eos else beam).append(hyp)
            # Sound early stop for the top-`nbest` complete hyps: when length_bonus<=0 a partial's
            # score only FALLS as it grows and completing (eos) never raises it, so once the best
            # partial cannot beat the current nbest-th completion, no future completion can enter
            # the top-nbest. A positive length_bonus makes partials RISE, so the stop is unsound
            # then -> search to the cap and finalize survivors (for/else below). For nbest=1 this
            # reduces exactly to the previous single-best guard, so those cells are unchanged.
            if length_bonus <= 0.0 and beam and len(ended) >= nbest:
                nth = sorted((e.score for e in ended), reverse=True)[nbest - 1]
                if max(x.score for x in beam) <= nth:
                    break
        else:
            # Reached the step cap with survivors (only when length_bonus>0 kept partials rising).
            # `pool` below keeps only completions, so a surviving partial would be silently
            # discarded and the result handed to whatever ended earliest — an EMPTY hypothesis when
            # one finished at step 0. Finalize survivors with their own eos score so they compete.
            # (The guard `break` skips this else; `not beam` also breaks, so `beam` is non-empty.)
            if beam:
                ys_in = torch.tensor([[bos_id] + h.tokens for h in beam], device=memory.device)
                logits = aed_head.forward(mem_b.expand(len(beam), -1, -1),
                                          mlen_b.expand(len(beam)), ys_in)
                aed_logp = torch.log_softmax(logits[:, -1].float(), dim=-1)
                eos_t = aed_logp.new_tensor([eos_id], dtype=torch.long)
                for i, h in enumerate(beam):
                    aed_c = h.aed + float(aed_logp[i, eos_id])
                    ctc_c = (float(scorer.score(h.tokens, eos_t, h.ctc_state)[0][0])
                             if use_ctc else 0.0)
                    ended.append(_Hyp(
                        tokens=h.tokens, aed=aed_c, ctc=ctc_c,
                        score=((1.0 - ctc_weight) * aed_c + ctc_weight * ctc_c
                               + length_bonus * len(h.tokens))))

        pool = ended if ended else beam
        results.append(sorted(pool, key=lambda x: x.score, reverse=True)[:nbest])
    return results


@torch.no_grad()
def joint_beam_search(ctc_head, aed_head, memory: torch.Tensor, memory_lengths: torch.Tensor, *,
                      beam_size: int = 10, ctc_weight: float = 0.3, bos_id: int = 1,
                      eos_id: int = 2, pad_id: int = 3, blank_id: int | None = None,
                      max_len_ratio: float = 1.0, length_bonus: float = 0.0,
                      pre_beam: int | None = None) -> list[list[int]]:
    """Single-best acoustic joint CTC+AED beam (the aed_beam / joint_beam cells).

    Thin wrapper over `joint_beam_search_nbest` (nbest=1): score = (1-ctc_weight)·AED +
    ctc_weight·CTC-prefix + length_bonus·len; ctc_weight=0 -> pure AED beam. Returns one
    bare-id list per utterance. +LM cells go through `joint_beam_search_nbest` + `lm_rescore`.
    """
    nbest = joint_beam_search_nbest(
        ctc_head, aed_head, memory, memory_lengths, beam_size=beam_size,
        ctc_weight=ctc_weight, bos_id=bos_id, eos_id=eos_id, pad_id=pad_id,
        blank_id=blank_id, max_len_ratio=max_len_ratio, length_bonus=length_bonus,
        pre_beam=pre_beam, nbest=1)
    return [hyps[0].tokens if hyps else [] for hyps in nbest]


@torch.no_grad()
def lm_rescore(nbest: list[_Hyp], lm, lm_weight: float, *, ctc_weight: float,
               device, length_bonus: float = 0.0) -> list[int]:
    """Second-pass LM rescoring of a COMPLETE-hypothesis n-best list (the aed_beam_lm /
    joint_beam_lm cells). Re-ranks by

        S(h) = (1-ctc_weight)·AED(h) + ctc_weight·CTC(h) + lm_weight·logP_LM(h) + length_bonus·len

    where logP_LM(h) is the LM's full-sequence log-prob of the complete hypothesis INCLUDING
    the terminal eos (`lm.sequence_logprob`). Because the LM only re-orders acoustically-complete
    hypotheses — it never participates in the search — it cannot truncate or over-generate. The
    acoustic term ((1-ctc_weight)·AED + ctc_weight·CTC) reproduces the beam's own score, so
    lm_weight=0 returns the acoustic best (identical to the no-LM cell). Returns bare ids.
    """
    if not nbest:
        return []
    lm_scores = lm.sequence_logprob([h.tokens for h in nbest], device)   # [n], full seq incl. eos
    best, best_s = nbest[0], float("-inf")
    for h, lm_s in zip(nbest, lm_scores.tolist()):
        s = ((1.0 - ctc_weight) * h.aed + ctc_weight * h.ctc
             + lm_weight * lm_s + length_bonus * len(h.tokens))
        if s > best_s:
            best_s, best = s, h
    return best.tokens
