from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from model import CLASS_NAMES, IMAGE_SIZE, build_model


PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the face-mask classifier.")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "models" / "face_mask_model.pt",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    for label, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {class_dir}")
        files = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not files:
            raise FileNotFoundError(f"No images found in {class_dir}")
        for image_path in files:
            try:
                with Image.open(image_path) as image:
                    images.append(
                        np.asarray(
                            image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)),
                            dtype=np.float32,
                        )
                    )
                labels.append(label)
            except (OSError, UnidentifiedImageError) as error:
                raise ValueError(f"Could not read image: {image_path}") from error
    return np.asarray(images) / 255.0, np.asarray(labels, dtype=np.int64)


def make_loader(images: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = torch.from_numpy(images).permute(0, 3, 1, 2)
    return DataLoader(TensorDataset(tensors, torch.from_numpy(labels)), batch_size=batch_size, shuffle=shuffle)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.to(device))
            labels = labels.to(device)
            total_loss += criterion(outputs, labels).item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive")
    set_seed(args.seed)
    images, labels = load_dataset(args.data_dir)
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.2, random_state=args.seed, stratify=labels
    )
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.1, random_state=args.seed, stratify=train_labels
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = make_loader(train_images, train_labels, args.batch_size, shuffle=True)
    val_loader = make_loader(val_images, val_labels, args.batch_size, shuffle=False)
    test_loader = make_loader(test_images, test_labels, args.batch_size, shuffle=False)
    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print(f"Loaded {len(images)} images. Training on {device}.")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, batch_labels in train_loader:
            inputs, batch_labels = inputs.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_labels.size(0)
            correct += (outputs.argmax(dim=1) == batch_labels).sum().item()
            total += batch_labels.size(0)
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"loss={running_loss / total:.4f} acc={correct / total:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

    test_loss, test_accuracy = evaluate(model, test_loader, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "image_size": IMAGE_SIZE,
            "test_accuracy": test_accuracy,
            "seed": args.seed,
        },
        args.output,
    )
    print(f"Test accuracy: {test_accuracy:.4f} (loss: {test_loss:.4f})")
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    train(parse_args())
