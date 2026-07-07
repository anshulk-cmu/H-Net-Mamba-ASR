"""LibriSpeech-960h dataset + manifest building.

Plan refs: Data (§3), features = 80-d log-Mel @ 100 Hz, global CMVN, SpecAugment [14].
Splits: train-960 (clean-100 + clean-360 + other-500); eval on dev/test {clean,other}.

TODO:
  - build_manifest(root) -> json list of {audio_path, text, dur}
  - LibriSpeechDataset(manifest, feats, bpe): returns (feat[T,80], tokens[U])
  - dynamic/bucketed batching by frame count
"""
