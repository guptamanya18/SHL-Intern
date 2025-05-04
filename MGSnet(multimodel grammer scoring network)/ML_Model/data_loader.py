import torch
from torch.utils.data import Dataset
import pandas as pd  # Make sure pandas is imported
# from transformers.tokenization_utils_base import BatchEncoding

# torch.serialization.add_safe_globals([BatchEncoding])

class MGSNetDataset(Dataset):
    def __init__(self, dataset_path, metadata_csv):
        """
        Custom dataset for loading MGSNet data from the prepared samples.

        :param dataset_path: Path to the preprocessed dataset (.pt file)
        :param metadata_csv: Path to the metadata CSV file containing labels and other info.
        """
        self.dataset = torch.load(dataset_path)  # Load the dataset
        self.metadata = pd.read_csv(metadata_csv)  # Load the metadata CSV

    def __len__(self):
        return len(self.dataset)  # Return the number of samples in the dataset

    def __getitem__(self, idx):
        sample = self.dataset[idx]  # Get the sample at the given index

        # Load the necessary features (audio, spectrogram, text embedding, grammar errors, label)
        waveform = sample['audio']
        spectrogram = sample['spectrogram']
        text_embedding = sample['embedding']
        grammar_vector = sample['grammar_errors']
        label = sample['label']

        return waveform, spectrogram, text_embedding, grammar_vector, label

