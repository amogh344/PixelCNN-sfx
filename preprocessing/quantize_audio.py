import torchaudio
import torchaudio.transforms as T
import torch
import os

def mu_law_encode(waveform, quantization_channels=256):
    mu = quantization_channels - 1
    safe_audio = torch.clamp(waveform, -1.0, 1.0)
    magnitude = torch.log1p(mu * torch.abs(safe_audio)) / torch.log1p(torch.tensor(mu, dtype=torch.float))
    signal = torch.sign(safe_audio) * magnitude
    return ((signal + 1) / 2 * mu + 0.5).long()

def preprocess_esc50_audio(esc50_dir, output_dir="data/processed", categories=None):
    metadata_path = os.path.join(esc50_dir, "ESC-50-master", "meta", "esc50.csv")
    audio_dir = os.path.join(esc50_dir, "ESC-50-master", "audio")
    import pandas as pd

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df = pd.read_csv(metadata_path)
    for _, row in df.iterrows():
        if categories and row['category'] not in categories:
            continue

        file_path = os.path.join(audio_dir, row['filename'])
        waveform, sr = torchaudio.load(file_path)
        if sr != 16000:
            resampler = T.Resample(orig_freq=sr, new_freq=16000)
            waveform = resampler(waveform)
        encoded = mu_law_encode(waveform[0])
        out_path = os.path.join(output_dir, row['filename'] + ".pt")
        torch.save(encoded, out_path)

