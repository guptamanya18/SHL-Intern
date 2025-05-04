import torch
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm
from ML_Model import MGSNetModel  # Import your model class
from ML_Model import MGSNetDataset

# Define paths (same as during training)
dataset_path = "E:\\Hackathon\\SHL_Intern\\assets\\Dataset\\prepared_3\\mgsnet_dataset.pt"
metadata_path = "E:\\Hackathon\\SHL_Intern\\assets\\Dataset\\prepared_3\\mgsnet_metadata.csv"
model_path = "E:\\Hackathon\\SHL_Intern\\assets\\model\\updated_mgsnet_model.pth"
# Create dataset and dataloader (same as during training)
dataset = MGSNetDataset(dataset_path, metadata_path)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)  # No shuffle for evaluation

# Initialize the model and load the trained weights
model = MGSNetModel()
model.load_state_dict(torch.load(model_path))  # Load the trained model weights
model.eval()  # Set the model to evaluation mode

# Set device (CUDA or CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Initialize variables for evaluation
all_labels = []
all_predictions = []

# Disable gradient computation for inference
with torch.no_grad():
    for batch in tqdm(train_loader, desc="Evaluating on Training Data"):
        # Unpack the batch
        waveform, spectrogram, text_embedding, grammar_vector, label = batch

        # Move data to device
        waveform = waveform.to(device)
        spectrogram = spectrogram.to(device)
        text_embedding = text_embedding.to(device)
        grammar_vector = grammar_vector.to(device)
        label = label.to(device)

        # Forward pass
        outputs = model(waveform, spectrogram, text_embedding, grammar_vector)

        # Store true labels and predictions
        all_labels.extend(label.cpu().numpy())  # Store the actual labels
        all_predictions.extend(outputs.cpu().numpy())  # Store the predicted outputs

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(all_labels, all_predictions)

# Calculate R² score
r2 = r2_score(all_labels, all_predictions)

print(f"Mean Squared Error on Training Data: {mse:.4f}")
print(f"R² Score on Training Data: {r2:.4f}")
