import argparse
import torch
from train.train_pixelcnn import train_model
from generate.generate_audio import generate_audio
from preprocessing.quantize_audio import preprocess_audio_file
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="PixelCNN Sound Effects Generator")
    parser.add_argument("command", choices=["preprocess", "train", "generate"], help="Operation to perform")
    parser.add_argument("--file", type=str, help="Path to WAV file (used with 'preprocess')")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None, help="Device to use (default: auto)")

    args = parser.parse_args()

    # Determine device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.command == "preprocess":
        if not args.file or not os.path.isfile(args.file):
            print("❌ Please provide a valid path to a WAV file using --file")
            sys.exit(1)
        preprocess_audio_file(args.file, "data/processed")

    elif args.command == "train":
        print(f"🚀 Training on device: {device}")
        train_model(device=device)

    elif args.command == "generate":
        print(f"🎵 Generating audio on device: {device}")
        generate_audio(device=device)

if __name__ == "__main__":
    main()
