"""Unit tests for the CTC head (src/dcasr/decoders/ctc.py).

CPU-only (no GPU gate): the head is a pure-PyTorch Linear + CTC loss + greedy decode.
Decode tests use an identity head (weight=I, bias=0, d_model=V+1) so the input features
ARE the logits — letting us craft an exact per-frame class sequence deterministically.
"""
import torch

from dcasr.decoders.ctc import CTCHead, ctc_greedy_collapse


def _identity_head(vocab_size: int) -> CTCHead:
    """Head whose forward is the identity: features [B,L,V+1] pass through as logits."""
    h = CTCHead(vocab_size + 1, vocab_size)
    with torch.no_grad():
        h.proj.weight.copy_(torch.eye(h.num_classes))
        h.proj.bias.zero_()
    return h


def _features_from_ids(ids: torch.Tensor, num_classes: int) -> torch.Tensor:
    """One-hot (scaled) so argmax over the last dim recovers `ids` exactly."""
    return torch.nn.functional.one_hot(ids, num_classes).float() * 10.0


# ── shape / contract ─────────────────────────────────────────────────────────
def test_output_shape_and_blank_default():
    head = CTCHead(16, 20)
    assert head.blank_id == 20 and head.num_classes == 21   # blank appended at id V
    logits = head(torch.randn(3, 7, 16))
    assert logits.shape == (3, 7, 21)


def test_blank_id_override():
    assert CTCHead(16, 20, blank_id=0).blank_id == 0


def test_log_probs_normalized():
    head = CTCHead(16, 20)
    lp = head.log_probs(torch.randn(2, 5, 16))
    assert lp.shape == (2, 5, 21)
    assert torch.allclose(lp.logsumexp(-1), torch.zeros(2, 5), atol=1e-5)


# ── greedy collapse rule ─────────────────────────────────────────────────────
def test_greedy_collapse_rule():
    B = 9  # blank id in these toy sequences
    assert ctc_greedy_collapse([1, 1, B, 1], B) == [1, 1]      # blank keeps repeats apart
    assert ctc_greedy_collapse([1, 1, 1], B) == [1]            # collapse consecutive dups
    assert ctc_greedy_collapse([B, 1, B, 2, 2, B], B) == [1, 2]
    assert ctc_greedy_collapse([B, B, B], B) == []             # all blank -> empty
    assert ctc_greedy_collapse([], B) == []


def test_frame_argmax_and_greedy_identity():
    head = _identity_head(5)                                   # V=5, blank=5, classes=6
    ids = torch.tensor([[1, 1, 5, 2, 5, 5, 3]])               # 5 = blank
    feats = _features_from_ids(ids, head.num_classes)
    assert torch.equal(head.frame_argmax(feats), ids)
    out = head.greedy_decode(feats, torch.tensor([7]))
    assert out == [[1, 2, 3]]                                  # collapse + drop blank
    assert all(i < head.vocab_size for i in out[0])           # no blank leaks through


def test_greedy_respects_feat_lengths():
    head = _identity_head(5)
    # valid frames -> [1,2]; a spurious label 3 sits in the padded tail (frames 3-4)
    ids = torch.tensor([[1, 2, 5, 3, 3]])
    feats = _features_from_ids(ids, head.num_classes)
    assert head.greedy_decode(feats, torch.tensor([3])) == [[1, 2]]   # tail ignored
    assert head.greedy_decode(feats, torch.tensor([5])) == [[1, 2, 3]]  # tail included


# ── loss ─────────────────────────────────────────────────────────────────────
def test_loss_finite_and_gradients_flow():
    torch.manual_seed(0)
    head = CTCHead(16, 20)
    feats = torch.randn(4, 50, 16, requires_grad=True)
    feat_lens = torch.full((4,), 50, dtype=torch.long)
    targets = torch.randint(0, 20, (4, 8))
    tgt_lens = torch.full((4,), 8, dtype=torch.long)
    loss = head.loss(feats, feat_lens, targets, tgt_lens)
    assert torch.isfinite(loss) and loss.item() >= 0.0
    loss.backward()
    assert torch.isfinite(head.proj.weight.grad).all() and head.proj.weight.grad.abs().sum() > 0
    assert torch.isfinite(feats.grad).all() and feats.grad.abs().sum() > 0


def test_loss_decreases_on_overfit():
    torch.manual_seed(0)
    head = CTCHead(8, 5)
    feats = (torch.randn(1, 40, 8) * 0.1).requires_grad_(True)
    feat_lens = torch.tensor([40])
    targets = torch.tensor([[1, 2, 3]])
    tgt_lens = torch.tensor([3])
    opt = torch.optim.Adam(list(head.parameters()) + [feats], lr=0.05)
    init = head.loss(feats, feat_lens, targets, tgt_lens).item()
    for _ in range(300):
        opt.zero_grad()
        loss = head.loss(feats, feat_lens, targets, tgt_lens)
        loss.backward()
        opt.step()
    final = head.loss(feats, feat_lens, targets, tgt_lens).item()
    assert final < init * 0.1, f"CTC loss did not overfit: {init:.3f} -> {final:.3f}"
    assert head.greedy_decode(feats, feat_lens) == [[1, 2, 3]]   # recovers the target


def test_targets_padding_ignored():
    torch.manual_seed(1)
    head = CTCHead(16, 20)
    feats = torch.randn(2, 50, 16)
    feat_lens = torch.tensor([50, 50])
    targets = torch.tensor([[3, 4, 5, 0], [6, 7, 0, 0]])     # 0 = pad tail
    tgt_lens = torch.tensor([3, 2])
    padded = torch.cat([targets, torch.zeros(2, 5, dtype=torch.long)], dim=1)  # extra pad
    l1 = head.loss(feats, feat_lens, targets, tgt_lens)
    l2 = head.loss(feats, feat_lens, padded, tgt_lens)
    assert torch.allclose(l1, l2), "padding beyond target_lengths must not affect the loss"


def test_targets_1d_equals_2d():
    torch.manual_seed(2)
    head = CTCHead(16, 20)
    feats = torch.randn(2, 40, 16)
    feat_lens = torch.tensor([40, 40])
    tgt_lens = torch.tensor([3, 2])
    targets_2d = torch.tensor([[3, 4, 5], [6, 7, 0]])
    targets_1d = torch.tensor([3, 4, 5, 6, 7])               # concatenated real tokens
    l2d = head.loss(feats, feat_lens, targets_2d, tgt_lens)
    l1d = head.loss(feats, feat_lens, targets_1d, tgt_lens)
    assert torch.allclose(l2d, l1d)
