import os
import sys
import unittest

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.data_loader import DataLoader


class TestDataLoaderSampleDevice(unittest.TestCase):
    def setUp(self):
        self.config = {
            'data_paths': {
                'n_baiot': './data/raw/n_baiot',
                'iot_23': './data/raw/iot_23',
                'bot_iot': './data/raw/bot_iot'
            }
        }
        self.loader = DataLoader(self.config['data_paths'])

    def test_load_n_baiot_sample_device(self):
        dataset = self.loader.load_n_baiot_dataset()
        self.assertGreater(dataset['total_samples'], 0)
        self.assertGreater(dataset['n_features'], 0)
        self.assertIn('Sample_Device', dataset.get('device_metadata', {}))


if __name__ == '__main__':
    unittest.main()
