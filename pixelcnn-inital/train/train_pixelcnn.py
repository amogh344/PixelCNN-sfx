import torch
from torch.utils.data import Dataset, DataLoader
import os
from model.pixelcnn import PixelCNN1D

class AudioDataset(Dataset):
    def __init__(self, path):
        self.files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pt')]
        if not self.files:
            raise RuntimeError(f"No .pt files found in '{path}'")

    def __getitem__(self, idx):
        audio = torch.load(self.files[idx]).long()
        return audio[:-1], audio[1:]

    def __len__(self):
        return len(self.files)

def train_model(data_path="data/processed", epochs=10, batch_size=4, device="cpu", save_path="pixelcnn.pt"):
    print(f"📁 Loading dataset from: {data_path}")
    dataset = AudioDataset(data_path)
    print(f"Dataset size: {len(dataset)}")
    
    # drop_last=False allows smaller final batch (important for small datasets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    if len(loader) == 0:
        raise RuntimeError("No batches to train on. Try lowering batch_size or adding more data.")

    print(f"🧠 Initializing model on {device}")
    model = PixelCNN1D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    print(f"🔁 Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output.reshape(-1, 256), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"✅ Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    print(f"💾 Saving model to: {save_path}")
    torch.save(model.state_dict(), save_path)
