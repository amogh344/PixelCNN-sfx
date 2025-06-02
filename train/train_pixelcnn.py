import os
import torch
from torch.utils.data import Dataset, DataLoader
from model.pixelcnn import PixelCNN1D

class AudioDataset(Dataset):
    def __init__(self, path):
        self.files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pt')]

    def __getitem__(self, idx):
        audio = torch.load(self.files[idx])
        return audio[:-1], audio[1:]

    def __len__(self):
        return len(self.files)

def train_model(data_path="data/processed", epochs=10, batch_size=4, device="cuda"):
    print(f"🚀 Training on device: {device}")
    print(f"📁 Loading dataset from: {data_path}")
    dataset = AudioDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    if len(loader) == 0:
        raise RuntimeError("❌ No training data found in processed directory.")

    model = PixelCNN1D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    print("🔁 Starting training for 10 epochs...")
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output.reshape(-1, 256), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        print(f"📉 Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    torch.save(model.state_dict(), "pixelcnn.pt")
    print("✅ Model saved to pixelcnn.pt")

