import torchaudio
import os
import torch
import numpy as np

def mu_law_encode(audio, quantization_channels=256):
    mu = quantization_channels - 1
    safe_audio = torch.clamp(audio, -1.0, 1.0)
    magnitude = torch.log1p(mu * torch.abs(safe_audio)) / np.log1p(mu)
    signal = torch.sign(safe_audio) * magnitude
    return ((signal + 1) / 2 * mu + 0.5).long()

def preprocess_audio_file(filepath, output_dir, sample_rate=16000):
    waveform, sr = torchaudio.load(filepath)
    if sr != sample_rate:
        transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = transform(waveform)

    audio = mu_law_encode(waveform[0])  # Mono only
    os.makedirs(output_dir, exist_ok=True)
    torch.save(audio, os.path.join(output_dir, os.path.basename(filepath) + ".pt"))

# Example usage
if __name__ == "__main__":
    import sys
    preprocess_audio_file(sys.argv[1], "data/processed")
