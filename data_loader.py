# data_loader.py

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import ast
from sklearn.preprocessing import MinMaxScaler

class TrajectoryDataset(Dataset):
    def __init__(self, file_path):
        self.data = self._load_data(file_path)
        self.scaler = MinMaxScaler()
        self._normalize_data()

    def _load_data(self, file_path):
        df = pd.read_csv(file_path)
        data = []
        for idx, row in df.iterrows():
            try:
                trajectory_list = ast.literal_eval(row['Data'])
                # Extract longitude and latitude (indices 1 and 2)
                trajectory = np.array([[point[1], point[2]] for point in trajectory_list])
                label = row['Label']
                label = self._process_label(label)
                data.append({'trajectory': trajectory, 'label': label})
            except (ValueError, SyntaxError) as e:
                print(f"Error parsing data at index {idx}: {e}")
                continue
        data = pd.DataFrame(data)
        return data

    def _process_label(self, label):
        # Assuming labels are in the format 'c2', 'c3', etc.
        if isinstance(label, str) and label.startswith('c'):
            return int(label[1:])
        else:
            return int(label)

    def _normalize_data(self):
        all_points = np.concatenate(self.data['trajectory'].tolist(), axis=0)
        self.scaler.fit(all_points)
        self.data['trajectory'] = self.data['trajectory'].apply(self.scaler.transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx]['trajectory']
        seq_len = len(sequence)
        label = self.data.iloc[idx]['label']
        # Return sequence as is, without padding
        return torch.FloatTensor(sequence), seq_len, label

    def inverse_transform(self, sequence):
        return self.scaler.inverse_transform(sequence)

def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    Sorts sequences by length in descending order and pads them.
    """
    # Sort the batch by sequence length in descending order
    batch.sort(key=lambda x: x[1], reverse=True)
    sequences, seq_lengths, labels = zip(*batch)
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    seq_lengths = torch.tensor(seq_lengths)
    labels = torch.tensor(labels)
    return sequences_padded, seq_lengths, labels
