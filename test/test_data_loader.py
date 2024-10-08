# tests/test_data_loading.py

import unittest
from data_loader import TrajectoryDataset, collate_fn
import torch

class TestDataLoading(unittest.TestCase):
    def test_load_train_data(self):
        train_dataset = TrajectoryDataset('data/train.csv')
        self.assertTrue(len(train_dataset) > 0, "Train dataset is empty.")
        sequence, seq_len, label = train_dataset[0]
        self.assertTrue(sequence.shape[0] == seq_len, "Sequence length mismatch.")
        self.assertTrue(sequence.shape[1] == 2, "Sequence feature dimension mismatch.")
        print('Train data loaded successfully.')

    def test_load_test_data(self):
        test_dataset = TrajectoryDataset('data/test.csv')
        self.assertTrue(len(test_dataset) > 0, "Test dataset is empty.")
        sequence, seq_len, label = test_dataset[0]
        self.assertTrue(sequence.shape[0] == seq_len, "Sequence length mismatch.")
        self.assertTrue(sequence.shape[1] == 2, "Sequence feature dimension mismatch.")
        print('Test data loaded successfully.')

if __name__ == '__main__':
    unittest.main()
