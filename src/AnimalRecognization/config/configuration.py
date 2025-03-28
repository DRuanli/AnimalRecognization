# src/AnimalRecognization/config/configuration.py
from src.AnimalRecognization.utils.common import read_yaml, create_directories
from src.AnimalRecognization.entity.config_entity import (DataIngestionConfig,PrepareModelConfig)
from src.AnimalRecognization.constants import *


class ConfigurationManager:
    def __init__(
            self,
            config_filepath=CONFIG_FILE_PATH,
            params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )

        return data_ingestion_config

    def get_prepare_model_config(self) -> PrepareModelConfig:
        config = self.config.prepare_model

        create_directories([config.root_dir])

        prepare_model_config = PrepareModelConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            params_image_size=self.params.base.image_size,
            params_batch_size=self.params.base.batch_size,
            params_learning_rate=self.params.model.learning_rate,
            params_num_classes=self.params.model.num_classes,
            params_dropout_rate=self.params.model.dropout_rate
        )

        return prepare_model_config