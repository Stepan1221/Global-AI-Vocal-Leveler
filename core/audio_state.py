from dataclasses import dataclass


@dataclass
class AudioState:
    ref_path: str | None = None
    target_path: str | None = None
    sample_rate: int | None = None
    output_path: str | None = None
