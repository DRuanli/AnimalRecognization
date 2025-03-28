# main.py
from src.AnimalRecognization import logger
from src.AnimalRecognization.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.AnimalRecognization.pipeline.stage_02_prepare_model import PrepareModelTrainingPipeline


STAGE_NAME = "Data Ingestion stage"

try:
   # Previous Stage 1 code
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

   # Stage 2
   STAGE_NAME = "Prepare Model stage"
   logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   prepare_model = PrepareModelTrainingPipeline()
   prepare_model.main()
   logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

except Exception as e:
   logger.exception(e)
   raise e