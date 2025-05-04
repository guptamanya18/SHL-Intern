import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from ML_Model import MGSNetModelImproved  # Import your model class

# Add necessary classes to safe globals
from transformers.tokenization_utils_base import BatchEncoding
from tokenizers import Encoding  # Import the Encoding class
import torch.serialization

torch.serialization.add_safe_globals([BatchEncoding, Encoding])


# Define a dataset class for your test data
class MGSNetTestDataset(Dataset):
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.samples = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_dir = os.path.join(self.dataset_dir, self.samples[idx])

        # Load all the required features
        waveform = torch.load(os.path.join(sample_dir, 'waveform.pt'))
        spectrogram = torch.load(os.path.join(sample_dir, 'spectrogram.pt'))

        # Handle tokens with safe loading
        try:
            tokens = torch.load(os.path.join(sample_dir, 'tokens.pt'))
        except Exception as e:
            # Fallback to weights_only=False if adding to safe_globals didn't work
            tokens = torch.load(os.path.join(sample_dir, 'tokens.pt'), weights_only=False)

        embedding = torch.load(os.path.join(sample_dir, 'embedding.pt'))
        grammar_vector = torch.load(os.path.join(sample_dir, 'grammar_vector.pt'))

        return {
            'filename': self.samples[idx],
            'waveform': waveform,
            'spectrogram': spectrogram,
            'tokens': tokens,
            'embedding': embedding,
            'grammar_vector': grammar_vector
        }


def generate_predictions_csv(model_path, test_dataset_dir, output_csv_path):
    # Load your trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model class first
    model = MGSNetModelImproved()  # Instead of MGSNetModel

    # Load the state dictionary
    model_state = torch.load(model_path, map_location=device)

    # Load the state into the model
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    # Create dataset and dataloader
    test_dataset = MGSNetTestDataset(test_dataset_dir)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Store predictions
    predictions = []

    # Run inference
    print("Generating predictions...")
    with torch.no_grad():
        for sample in tqdm(test_loader):
            # Extract features needed for your model
            filename = f"{sample['filename'][0]}.wav"

            # Prepare inputs for your specific model format
            waveform = sample['waveform'].to(device)
            spectrogram = sample['spectrogram'].to(device)
            embedding = sample['embedding'].to(device)
            grammar_vector = sample['grammar_vector'].to(device)

            # Get prediction
            # Call the model with the correct parameter order based on your model's forward() method
            outputs = model(waveform, spectrogram, embedding, grammar_vector)

            # Process the output
            predicted_score = outputs.item() if isinstance(outputs, torch.Tensor) and outputs.numel() == 1 else outputs

            # Store prediction
            predictions.append({
                'filename': filename,
                'label': predicted_score
            })

    # Create and save CSV
    pd.DataFrame(predictions).to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")


if __name__ == "__main__":
    # Update these paths as needed
    model_path = r"E:\Hackathon\SHL_Intern\assets\model\UPDATED_model.pth"
    test_dataset_dir = "E:/Hackathon/SHL_Intern/assets/Dataset/test_data/samples"
    output_csv_path = "E:/Hackathon/SHL_Intern/assets/Dataset/submission.csv"

    generate_predictions_csv(model_path, test_dataset_dir, output_csv_path)