"""Interpretability: boundary alignment, chunk probing, emergence curves (plan §6.4)."""
from dcasr.interp.alignments import (
    alignment_record, check_alignment, load_alignments, parse_textgrid,
    prepare_corpus, select_subset, speaker_of, write_alignments,
)
from dcasr.interp.boundary_align import (
    aggregate, collect_boundaries, frame_boundary_times, match_boundaries,
    r_value, random_baseline, score_utterances, stage2_boundary_times, true_edges,
)
from dcasr.interp.driver import (
    NoisePerturbation, Perturbation, PerturbedDataset, RecordingLoader,
    SilencePerturbation, SpeedPerturbation, assert_disjoint, boundary_report,
    durations_from_entries, emergence_report, flatten_metrics,
    list_epoch_checkpoints, matched_deltas, perturbations_from_config,
    probe_report, robustness_report, score_perturbation, utt_seed,
)
from dcasr.interp.probes import (
    chunk_spans, collapse_stress, collect_probe_data, frame_labels, majority_label,
    phone_class, subsample, to_classes, top_k_filter, train_probe,
)

__all__ = ["parse_textgrid", "alignment_record", "check_alignment", "prepare_corpus",
           "select_subset", "speaker_of", "write_alignments", "load_alignments",
           "frame_boundary_times", "stage2_boundary_times", "true_edges",
           "match_boundaries", "aggregate", "r_value", "random_baseline",
           "score_utterances", "collect_boundaries",
           "frame_labels", "chunk_spans", "majority_label", "collapse_stress",
           "phone_class", "to_classes", "collect_probe_data", "subsample",
           "top_k_filter", "train_probe",
           "Perturbation", "NoisePerturbation", "SpeedPerturbation",
           "SilencePerturbation", "PerturbedDataset", "RecordingLoader",
           "perturbations_from_config", "utt_seed", "assert_disjoint",
           "durations_from_entries", "list_epoch_checkpoints", "matched_deltas",
           "boundary_report", "probe_report", "score_perturbation",
           "robustness_report", "emergence_report", "flatten_metrics"]
