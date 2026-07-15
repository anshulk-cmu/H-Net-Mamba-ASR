"""Decoders + search + LM fusion (three read-outs of one hybrid model, plan §6.3)."""
from dcasr.decoders.ctc import CTCHead, ctc_greedy_collapse

__all__ = ["CTCHead", "ctc_greedy_collapse"]
