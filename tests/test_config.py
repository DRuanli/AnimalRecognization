# tests/test_config.py
import os
import pytest
from pathlib import Path
from src.AnimalRecognization.config.configuration import ConfigurationManager

def test_config_loading():
    config_manager = ConfigurationManager()
    assert config_manager.config is not None
    assert config_manager.params is not None

def test_data_ingestion_config():
    config_manager = ConfigurationManager()
    data_ingestion_config = config_manager.get_data_ingestion_config()
    assert data_ingestion_config.root_dir is not None
    assert data_ingestion_config.source_URL is not None

def test_prepare_model_config():
    config_manager = ConfigurationManager()
    prepare_model_config = config_manager.get_prepare_model_config()
    assert prepare_model_config.root_dir is not None
    assert prepare_model_config.model_path is not None