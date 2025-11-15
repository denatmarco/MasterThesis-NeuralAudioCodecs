"""
Command-line interface for training, evaluation, and TorchScript export.
"""

from __future__ import annotations
import os
import json
import argparse

from .training import train_encoder, encode_dataset
from .encoder_system import EncoderSystem


def main():
    p = argparse.ArgumentParser(description="Third-octave VQ encoder CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    pt = sub.add_parser("train", help="Train encoder on a folder of WAV files")
    pt.add_argument("--train_folder", type=str, required=True)
    pt.add_argument("--out_dir", type=str, default="./checkpoints")
    pt.add_argument("--sr", type=int, default=16000)
    pt.add_argument("--n_fft", type=int, default=512)
    pt.add_argument("--hop", type=int, default=128)
    pt.add_argument("--latent_dim", type=int, default=24)
    pt.add_argument("--hidden", type=int, default=96)
    pt.add_argument("--num_groups", type=int, default=2)
    pt.add_argument("--codebook_size", type=int, default=256)
    pt.add_argument("--beta", type=float, default=0.25)
    pt.add_argument("--epochs", type=int, default=25)
    pt.add_argument("--batch_size", type=int, default=512)
    pt.add_argument("--lr", type=float, default=2e-3)
    pt.add_argument("--device", type=str, default="cuda")
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--val_folder", type=str, default=None)
    pt.add_argument("--export_ts", action="store_true")

    # encode
    pe = sub.add_parser("encode", help="Encode a folder of WAV files")
    pe.add_argument("--ckpt", type=str, required=True)
    pe.add_argument("--input_folder", type=str, required=True)
    pe.add_argument("--output_folder", type=str, default="./encoded_features")

    # eval
    pv = sub.add_parser("eval", help="Evaluate a folder with a trained checkpoint")
    pv.add_argument("--ckpt", type=str, required=True)
    pv.add_argument("--folder", type=str, required=True)
    pv.add_argument("--samples_per_file", type=int, default=4096)
    pv.add_argument("--max_files", type=int, default=None)

    args = p.parse_args()

    if args.cmd == "train":
        ckpt = train_encoder(
            train_folder=args.train_folder,
            out_dir=args.out_dir,
            sr=args.sr,
            n_fft=args.n_fft,
            hop=args.hop,
            latent_dim=args.latent_dim,
            hidden=args.hidden,
            num_groups=args.num_groups,
            codebook_size=args.codebook_size,
            beta=args.beta,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            seed=args.seed,
        )
        if args.val_folder:
            system = EncoderSystem.load_from_checkpoint(ckpt)
            metrics = system.evaluate_folder(args.val_folder)
            print("Validation metrics:", metrics)
            with open(os.path.join(args.out_dir, "val_metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2)

        if args.export_ts:
            system = EncoderSystem.load_from_checkpoint(ckpt)
            ts_path = os.path.join(args.out_dir, "encoder_script.pt")
            out = system.export_torchscript(ts_path)
            print("Exported TorchScript to:", out)

    elif args.cmd == "encode":
        encode_dataset(args.ckpt, args.input_folder, args.output_folder)

    elif args.cmd == "eval":
        system = EncoderSystem.load_from_checkpoint(args.ckpt)
        metrics = system.evaluate_folder(args.folder, samples_per_file=args.samples_per_file, max_files=args.max_files)
        print("Evaluation metrics:", metrics)


if __name__ == "__main__":
    main()
