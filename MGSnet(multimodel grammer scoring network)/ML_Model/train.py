import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_loader import MGSNetDataset  # Import your custom dataset class
from model_2 import MGSNetModelImproved  # Import your model class

# Define paths
dataset_path = "E:\\Hackathon\\SHL_Intern\\assets\\Dataset\\prepared_3\\mgsnet_dataset.pt"
metadata_path = "E:\\Hackathon\\SHL_Intern\\assets\\Dataset\\prepared_3\\mgsnet_metadata.csv"

# Create dataset and dataloader
dataset = MGSNetDataset(dataset_path, metadata_path)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the model
model = MGSNetModelImproved()


# Training function
def train_model(model, train_loader, epochs=10, learning_rate=1e-3, device='cuda'):
    model.to(device)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()  # Since this is a regression task

    # Loop over epochs
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            # Unpack the batch
            waveform, spectrogram, text_embedding, grammar_vector, label = batch

            # Move to device
            waveform = waveform.to(device)
            spectrogram = spectrogram.to(device)
            text_embedding = text_embedding.to(device)
            grammar_vector = grammar_vector.to(device)
            label = label.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(waveform, spectrogram, text_embedding, grammar_vector)

            # Compute loss
            loss = criterion(outputs.squeeze(), label)
            running_loss += loss.item() * waveform.size(0)
            total_samples += waveform.size(0)

            # Backpropagate and update weights
            loss.backward()
            optimizer.step()

        avg_loss = running_loss / total_samples
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    return model


# Set device (CUDA or CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Train the model
model_trained = train_model(model, train_loader, epochs=10, learning_rate=1e-3, device=device)

# Save the trained model
torch.save(model_trained.state_dict(), 'UPDATED_model.pth')
