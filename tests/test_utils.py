# tests/test_utils.py
import os
import yaml
import pytest
from box import ConfigBox
from pathlib import Path
from src.AnimalRecognization.utils.common import read_yaml, create_directories, get_size

def test_read_yaml(tmpdir):
    yaml_file = tmpdir.join("test.yaml")
    yaml_file.write("""
    key: value
    nested:
      inner: data
    """)
    result = read_yaml(yaml_file)
    assert isinstance(result, ConfigBox)
    assert result.key == "value"
    assert result.nested.inner == "data"

def test_create_directories(tmpdir):
    test_dir = os.path.join(tmpdir, "test_dir")
    create_directories([test_dir], verbose=False)
    assert os.path.exists(test_dir)