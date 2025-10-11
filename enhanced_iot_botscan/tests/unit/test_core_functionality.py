import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from utils.config_manager import ConfigManager
from scripts.download_datasets import DatasetDownloader
from data.data_loader import DataLoader

@pytest.fixture
def temp_config_file(tmp_path):
    config_content = '''
    system:
      name: 'Test System'
    machine_learning:
      ensemble:
        algorithms: []
    data:
      data_paths:
        n_baiot: "./data/raw/n_baiot"
        iot_23: "./data/raw/iot_23"
        bot_iot: "./data/raw/bot_iot"
    '''
    config_file = tmp_path / "config.yaml"
    config_file.write_text(config_content)
    return str(config_file)

def test_config_manager_validation(temp_config_file):
    """Test that ConfigManager validates a correct config file without errors."""
    config_manager = ConfigManager(config_path=temp_config_file)
    issues = config_manager.validate_config()
    assert len(issues) == 0, f"Validation issues found: {issues}"

@patch('scripts.download_datasets.pd')
@patch('scripts.download_datasets.np')
def test_dataset_downloader_create_sample_data(mock_np, mock_pd, tmp_path):
    """Test the creation of sample data."""
    downloader = DatasetDownloader(data_dir=str(tmp_path))
    downloader.create_sample_data()

    # Check if files were created
    assert (tmp_path / 'n_baiot' / 'Sample_Device' / 'benign_traffic.csv').exists()
    assert (tmp_path / 'n_baiot' / 'Sample_Device' / 'mirai_attacks.csv').exists()
    assert (tmp_path / 'iot_23' / 'iot_23_sample.csv').exists()
    assert (tmp_path / 'bot_iot' / 'bot_iot_sample.csv').exists()

@patch('data.data_loader.pd')
def test_data_loader_load_n_baiot(mock_pd, tmp_path):
    """Test loading the N-BaIoT dataset."""
    # Create dummy data
    n_baiot_path = tmp_path / "n_baiot" / "Danmini_Doorbell"
    n_baiot_path.mkdir(parents=True)
    (n_baiot_path / "benign.csv").touch()
    (n_baiot_path / "mirai.csv").touch()

    # Mock pandas read_csv and concat
    mock_df = MagicMock()
    mock_df.columns = ['f1', 'f2']
    mock_df.values.astype.return_value = [[1, 2], [3, 4]]
    mock_pd.read_csv.return_value = mock_df
    
    # Since concat is called on a list of dataframes, we need to mock that behavior
    mock_concat_df = MagicMock()
    mock_concat_df.replace.return_value.fillna.return_value.values.astype.return_value = [[1, 2], [3, 4], [5, 6], [7, 8]]
    mock_concat_df.columns = ['f1', 'f2']
    mock_pd.concat.return_value = mock_concat_df

    config = {
        'data_paths': {
            'n_baiot': str(tmp_path / "n_baiot")
        }
    }
    loader = DataLoader(config)
    dataset = loader.load_n_baiot_dataset(device_types=['Danmini_Doorbell'])

    assert dataset is not None
    assert dataset['dataset_name'] == 'N-BaIoT'
    assert len(dataset['features']) > 0
