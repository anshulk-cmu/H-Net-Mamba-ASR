"""Joint CTC+AED one-pass beam search (plan §6.3 / Watanabe et al. [12]) — the best-WER
headline read-out of one hybrid model.

A label-synchronous beam driven by the AED head, where each partial hypothesis is also
scored by CTC via the CTC *prefix score* (Graves' forward algorithm over the blank/label
lattice), so the combined score is
    score(h) = (1 - ctc_weight) · logP_AED(h) + ctc_weight · logP_CTC-prefix(h)
The AED head supplies fluency; the CTC prefix score keeps hypotheses anchored to the audio
(no skips/repeats). ctc_weight=0 reduces this to a pure AED (attention) beam — the AED
read-out. An external LM plugs in later as an additive scorer (lm_fusion.py / decode.py).

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
    aed: float                                             # cumulative AED log-prob
    ctc: float                                             # absolute CTC prefix log-prob
    lm: float = 0.0                                        # cumulative external-LM log-prob
    ilm: float = 0.0                                       # cumulative internal-LM log-prob
    ctc_state: torch.Tensor | None = None
    score: float = field(default=0.0)


@torch.no_grad()
def joint_beam_search(ctc_head, aed_head, memory: torch.Tensor, memory_lengths: torch.Tensor, *,
                      beam_size: int = 10, ctc_weight: float = 0.3, bos_id: int = 1,
                      eos_id: int = 2, pad_id: int = 3, blank_id: int | None = None,
                      max_len_ratio: float = 1.0, length_bonus: float = 0.0,
                      pre_beam: int | None = None, lm=None, lm_weight: float = 0.0,
                      ilm=None, ilm_weight: float = 0.0) -> list[list[int]]:
    """Joint CTC+AED (+ optional external LM) beam over a batch of encoder outputs.

    score = (1-ctc_weight)·AED + ctc_weight·CTC-prefix + lm_weight·LM - ilm_weight·ILM
            + length_bonus·len.
    ctc_weight=0 -> pure AED beam (ctc_head may be None); lm (a CausalLMScorer, lm_weight>0)
    adds shallow fusion. ilm (an ILMScorer, ilm_weight>0) subtracts the AED's own internal LM
    so fusion does not double-count the language prior (density ratio); ilm_weight=0 leaves
    plain shallow fusion. Run heads/LM in eval(). Returns one bare-id list per utterance.
    """
    B = memory.shape[0]
    V = aed_head.vocab_size
    if blank_id is None and ctc_head is not None:
        blank_id = ctc_head.blank_id
    pre = min(V, pre_beam if pre_beam is not None else max(2 * beam_size, 15))
    results: list[list[int]] = []

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
        beam = [_Hyp(tokens=[], aed=0.0, ctc=0.0, lm=0.0, ctc_state=init_state, score=0.0)]
        ended: list[_Hyp] = []
        use_lm = lm is not None and lm_weight != 0.0
        use_ilm = ilm is not None and ilm_weight != 0.0
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
            lm_logp = lm.next_logprobs([h.tokens for h in beam], memory.device) if use_lm else None
            ilm_logp = (ilm.next_logprobs([h.tokens for h in beam], memory.device)
                        if use_ilm else None)
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
                    lm_c = h.lm + float(lm_logp[i, c]) if use_lm else 0.0
                    ilm_c = h.ilm + float(ilm_logp[i, c]) if use_ilm else 0.0
                    total = ((1.0 - ctc_weight) * aed_c + ctc_weight * ctc_c
                             + lm_weight * lm_c - ilm_weight * ilm_c
                             + length_bonus * len(toks))
                    ext.append((total, is_eos,
                                _Hyp(tokens=toks, aed=aed_c, ctc=ctc_c, lm=lm_c, ilm=ilm_c,
                                     ctc_state=(ctc_states[j] if (use_ctc and not is_eos) else None),
                                     score=total)))
            ext.sort(key=lambda e: e[0], reverse=True)         # global top-k over all expansions
            beam = []
            for _total, is_eos, hyp in ext[:beam_size]:
                (ended if is_eos else beam).append(hyp)
            # partial scores only fall as they grow, so stopping is safe — UNLESS a term that
            # RISES with length lets a partial overtake a finished hyp later; then search to
            # the cap. Both a positive length_bonus and ILM subtraction do that: -ilm_weight·ILM
            # adds |ilm_weight·logP_ILM| > 0 per token.
            if (length_bonus <= 0.0 and ilm_weight <= 0.0 and ended and beam
                    and max(e.score for e in ended) >= max(x.score for x in beam)):
                break
        else:
            # Ran to the step cap without the guard proving a completion dominates. The
            # surviving partials are the highest-scoring hypotheses we have, but `pool`
            # below keeps only completions, so they would be silently discarded and the
            # result handed to whatever ended earliest — an EMPTY hypothesis when one
            # finished at step 0. Finalize them with their own eos score so they compete.
            # (Only reachable with a length-rising term: length_bonus > 0 or ilm_weight > 0.
            # `not beam` breaks out above, so `beam` is non-empty here.)
            if beam:
                ys_in = torch.tensor([[bos_id] + h.tokens for h in beam], device=memory.device)
                logits = aed_head.forward(mem_b.expand(len(beam), -1, -1),
                                          mlen_b.expand(len(beam)), ys_in)
                aed_logp = torch.log_softmax(logits[:, -1].float(), dim=-1)
                lm_logp = (lm.next_logprobs([h.tokens for h in beam], memory.device)
                           if use_lm else None)
                ilm_logp = (ilm.next_logprobs([h.tokens for h in beam], memory.device)
                            if use_ilm else None)
                eos_t = aed_logp.new_tensor([eos_id], dtype=torch.long)
                for i, h in enumerate(beam):
                    aed_c = h.aed + float(aed_logp[i, eos_id])
                    ctc_c = (float(scorer.score(h.tokens, eos_t, h.ctc_state)[0][0])
                             if use_ctc else 0.0)
                    lm_c = h.lm + float(lm_logp[i, eos_id]) if use_lm else 0.0
                    ilm_c = h.ilm + float(ilm_logp[i, eos_id]) if use_ilm else 0.0
                    ended.append(_Hyp(
                        tokens=h.tokens, aed=aed_c, ctc=ctc_c, lm=lm_c, ilm=ilm_c,
                        score=((1.0 - ctc_weight) * aed_c + ctc_weight * ctc_c
                               + lm_weight * lm_c - ilm_weight * ilm_c
                               + length_bonus * len(h.tokens))))

        pool = ended if ended else beam
        best = max(pool, key=lambda x: x.score) if pool else _Hyp([], 0.0, 0.0)
        results.append(best.tokens)
    return results
