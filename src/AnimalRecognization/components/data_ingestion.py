# src/AnimalRecognization/components/data_ingestion.py
import os
import zipfile
import kaggle
from src.AnimalRecognization import logger
from src.AnimalRecognization.entity.config_entity import DataIngestionConfig
from src.AnimalRecognization.utils.common import get_size


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> None:
        '''
        Fetch data from Kaggle
        '''
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs(os.path.dirname(zip_download_dir), exist_ok=True)

            logger.info(f"Downloading animal dataset from {dataset_url}")

            # Extract dataset name from URL (modify as needed for your Kaggle dataset)
            dataset_name = dataset_url.split("/")[-1]

            # Use Kaggle API to download dataset
            # Note: Requires Kaggle API credentials to be set up
            kaggle.api.dataset_download_files(
                dataset=dataset_name,
                path=os.path.dirname(zip_download_dir),
                unzip=False
            )

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")
            logger.info(f"File size: {get_size(self.config.local_data_file)}")

        except Exception as e:
            raise e

    def extract_zip_file(self) -> None:
        """
        Extracts the zip file into the data directory
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        logger.info(f"Extracting zip file: {self.config.local_data_file} into dir: {unzip_path}")

        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

        logger.info(f"Extraction completed")