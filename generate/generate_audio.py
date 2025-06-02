import torch
import torchaudio
import os
from model.pixelcnn import PixelCNN1D

def mu_law_decode(encoded, quantization_channels=256):
    mu = quantization_channels - 1
    signal = 2 * (encoded.float() / mu) - 1
    magnitude = (1 / mu) * ((1 + mu) ** torch.abs(signal) - 1)
    return torch.sign(signal) * magnitude

def generate_audio(length=320000, device="cuda", model_path="pixelcnn.pt", out_file="generated.wav", sample_rate=16000):
    if not os.path.isfile(model_path):
        print(f"❌ Model file '{model_path}' not found.")
        return

    print(f"📦 Loading model from: {model_path} to device: {device}")
    model = PixelCNN1D().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print(f"🎼 Generating {length} samples...")
    sequence = torch.zeros(1, length, dtype=torch.long).to(device)

    for i in range(1, length):
        with torch.no_grad():
            output = model(sequence[:, :i])
            probs = torch.softmax(output[0, -1], dim=-1)
            sample = torch.multinomial(probs, 1)
            sequence[0, i] = sample

        if i % 1000 == 0 or i == length - 1:
            print(f"Progress: {i}/{length} samples generated", end="\r")

    print("\n🔊 Decoding mu-law and saving waveform...")
    waveform = mu_law_decode(sequence[0].cpu()).unsqueeze(0)
    torchaudio.save(out_file, waveform, sample_rate)
    print(f"✅ Audio saved to: {out_file}")

