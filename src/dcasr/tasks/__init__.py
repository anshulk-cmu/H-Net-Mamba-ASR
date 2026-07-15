"""Task builders: the single seam between a resolved config and wired DC-ASR objects."""
from dcasr.tasks.asr_task import (
    DCASRModel, build_encoder, build_head, build_loss, build_model,
    build_optimizer, build_scheduler,
)

__all__ = ["DCASRModel", "build_encoder", "build_head", "build_loss", "build_model",
           "build_optimizer", "build_scheduler"]
