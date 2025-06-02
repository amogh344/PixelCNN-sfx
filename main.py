import argparse
import torch
from preprocessing.esc50_loader import download_and_extract_esc50
from preprocessing.quantize_audio import preprocess_esc50_audio
from train.train_pixelcnn import train_model
from generate.generate_audio import generate_audio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["download", "preprocess", "train", "generate"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--length", type=int, default=320000)  # ~20 sec
    args = parser.parse_args()

    if args.command == "download":
        download_and_extract_esc50()
    elif args.command == "preprocess":
        preprocess_esc50_audio("data/ESC-50")
    elif args.command == "train":
        train_model(device=args.device)
    elif args.command == "generate":
        generate_audio(length=args.length, device=args.device)

if __name__ == "__main__":
    main()

