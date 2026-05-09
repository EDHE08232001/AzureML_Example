import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _banner() -> None:
    print("=" * 60)
    print("  MCUCoder — Progressive Learned Image Compression")
    print("=" * 60)


def _menu() -> str:
    print("\nSelect an action:")
    print("  1) Train    — train the model on ImageNet / validate on Kodak")
    print("  2) Evaluate — run rate-distortion evaluation and plot RD curves")
    print("  3) Prepare  — pre-process raw ImageNet images for training")
    print()
    return input("Enter 1, 2, or 3: ").strip()


def main() -> None:
    _banner()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["train", "evaluate", "prepare"],
        default=None,
        help="Run non-interactively (for Azure ML jobs).",
    )
    args = parser.parse_args()

    choice = args.mode or {
        "1": "train", "2": "evaluate", "3": "prepare"
    }.get(_menu())

    if choice == "train":
        from src.train import train_model
        print("\n── Training ─────────────────────────────────────────────")
        ckpt = train_model()
        print(f"\nTraining complete.  Checkpoint: {ckpt}")

    elif choice == "evaluate":
        from src.evaluate import evaluate_model
        print("\n── Evaluation ───────────────────────────────────────────")
        summary = evaluate_model()
        print(f"\nEvaluation complete.  Summary: {summary}")

    elif choice == "prepare":
        from src.prepare_data import prepare_imagenet
        print("\n── ImageNet Preparation ─────────────────────────────────")
        out_dir = prepare_imagenet()
        print(f"\nPreparation complete.  Output directory: {out_dir}")

    else:
        print("Invalid input — please run again and enter 1, 2, or 3.")
        sys.exit(1)


if __name__ == "__main__":
    main()