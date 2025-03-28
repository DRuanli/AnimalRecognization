# src/AnimalRecognization/entity/config_entity.py
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareModelConfig:
    root_dir: Path
    model_path: Path
    params_image_size: list
    params_batch_size: int
    params_learning_rate: float
    params_num_classes: int
    params_dropout_rate: float