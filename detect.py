"""Image object classification helpers used by the command line and web app."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
SAVE_DIR = BASE_DIR / "saved_model"
TORCH_HOME = BASE_DIR / "venv" / "torch_models"

# Keep downloaded torchvision weights inside the project when possible.
os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def load_model() -> tuple[Any, Any, Any, torch.nn.Module, Any, torch.device]:
    """Load the trained artifacts once and reuse them for subsequent images."""
    required_files = ("classifier.pkl", "scaler.pkl", "label_encoder.pkl")
    missing = [name for name in required_files if not (SAVE_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model file(s) in {SAVE_DIR}: {', '.join(missing)}"
        )

    clf = joblib.load(SAVE_DIR / "classifier.pkl")
    scaler = joblib.load(SAVE_DIR / "scaler.pkl")
    label_encoder = joblib.load(SAVE_DIR / "label_encoder.pkl")

    device = _device()
    weights = models.ResNet50_Weights.DEFAULT
    resnet = models.resnet50(weights=weights)
    resnet.fc = torch.nn.Identity()
    resnet = resnet.to(device)
    resnet.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )
    return clf, scaler, label_encoder, resnet, transform, device


def _sliding_window_crops(image: Image.Image) -> list[Image.Image]:
    """Generate overlapping crops so the classifier can vote on each object."""
    width, height = image.size
    crops: list[Image.Image] = [image]

    for fraction in (0.8, 0.6, 0.4):
        crop_width = max(1, int(width * fraction))
        crop_height = max(1, int(height * fraction))
        stride_x = max(1, int(crop_width * 0.3))
        stride_y = max(1, int(crop_height * 0.3))

        x_positions = list(range(0, max(width - crop_width + 1, 1), stride_x))
        y_positions = list(range(0, max(height - crop_height + 1, 1), stride_y))
        if x_positions[-1] != width - crop_width:
            x_positions.append(width - crop_width)
        if y_positions[-1] != height - crop_height:
            y_positions.append(height - crop_height)

        for y in y_positions:
            for x in x_positions:
                crops.append(image.crop((x, y, x + crop_width, y + crop_height)))
    return crops


def classify_image(
    image: Image.Image, confidence_threshold: float = 0.8, min_votes: int = 4
) -> list[dict[str, float | int | str]]:
    """Classify an image and return consensus results sorted by confidence."""
    if not 0 <= confidence_threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if min_votes < 1:
        raise ValueError("min_votes must be at least 1")

    clf, scaler, label_encoder, resnet, transform, device = load_model()
    crops = _sliding_window_crops(image.convert("RGB"))

    tensors = torch.stack([transform(crop) for crop in crops]).to(device)
    with torch.inference_mode():
        embeddings = resnet(tensors).cpu().numpy()

    probabilities = clf.predict_proba(scaler.transform(embeddings.astype(np.float32)))
    votes: dict[str, int] = {}
    max_confidence: dict[str, float] = {}

    for distribution in probabilities:
        label_index = int(np.argmax(distribution))
        confidence = float(distribution[label_index])
        if confidence < confidence_threshold:
            continue
        label = str(label_encoder.inverse_transform([label_index])[0])
        votes[label] = votes.get(label, 0) + 1
        max_confidence[label] = max(max_confidence.get(label, 0.0), confidence)

    detections = [
        {"label": label, "confidence": max_confidence[label], "votes": votes[label]}
        for label in votes
        if votes[label] >= min_votes
    ]
    return sorted(detections, key=lambda item: float(item["confidence"]), reverse=True)


def detect_objects(
    img_path: str | os.PathLike[str], confidence_threshold: float = 0.8, min_votes: int = 4
) -> list[str]:
    """Classify an image file and print a concise command-line report."""
    path = Path(img_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    with Image.open(path) as source:
        image = source.convert("RGB")
    detections = classify_image(image, confidence_threshold, min_votes)

    print(f"\nObjects detected in {path.name}:")
    if detections:
        for detection in detections:
            print(
                f"  {detection['label']}: {detection['confidence']:.1%} "
                f"({detection['votes']} votes)"
            )
    else:
        print("  No objects met the confidence and consensus thresholds.")
    return [str(detection["label"]) for detection in detections]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify objects in an image.")
    parser.add_argument("image", nargs="?", default="test.jpg", help="Path to an image")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--min-votes", type=int, default=4)
    args = parser.parse_args()
    detect_objects(args.image, args.threshold, args.min_votes)
