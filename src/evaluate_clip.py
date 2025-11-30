"""
Evaluate CLIPScore for baseline vs LoRA images.

Usage:
    python evaluate_clip.py \
        --baseline_dir ./results/baseline \
        --lora_dir ./results/lora \
        --text "a korean pretty woman with large eyes, studio lighting"

baseline_dir 또는 lora_dir 한 가지만 넣으면 해당 폴더 평균만 출력합니다.
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from PIL import Image
import open_clip


def load_clip_model(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    model.to(device)
    model.eval()
    return model, preprocess, device


def compute_clip_scores_for_folder(
    folder: Path, text: str, model, preprocess, device: str
) -> Dict[str, float]:
    """
    해당 폴더 내의 png/jpg 이미지를 대상으로 CLIPScore 계산.
    """
    image_paths = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )

    if not image_paths:
        raise ValueError(f"No images found in: {folder}")

    text_tokens = open_clip.tokenize([text]).to(device)

    scores = {}
    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            image_tensor = preprocess(img).unsqueeze(0).to(device)

            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_tokens)

            # 정규화 후 cosine similarity 사용
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).squeeze().item()
            scores[img_path.name] = similarity

    return scores


def print_scores(name: str, scores: Dict[str, float]):
    print(f"\n[{name}] CLIPScore per image")
    for k, v in scores.items():
        print(f"  {k:<40} {v:.4f}")
    avg = sum(scores.values()) / len(scores)
    print(f"\n[{name}] Average CLIPScore: {avg:.4f}")
    return avg


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline_dir",
        type=str,
        default=None,
        help="Directory with baseline images",
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        default=None,
        help="Directory with LoRA fine-tuned images",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="a korean pretty woman with large eyes, studio lighting",
        help="Text prompt to evaluate similarity against",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.baseline_dir is None and args.lora_dir is None:
        raise ValueError("At least one of --baseline_dir or --lora_dir must be set.")

    model, preprocess, device = load_clip_model()

    if args.baseline_dir:
        baseline_scores = compute_clip_scores_for_folder(
            Path(args.baseline_dir), args.text, model, preprocess, device
        )
        baseline_avg = print_scores("Baseline", baseline_scores)
    else:
        baseline_avg = None

    if args.lora_dir:
        lora_scores = compute_clip_scores_for_folder(
            Path(args.lora_dir), args.text, model, preprocess, device
        )
        lora_avg = print_scores("LoRA", lora_scores)
    else:
        lora_avg = None

    # 간단 비교 출력
    if baseline_avg is not None and lora_avg is not None:
        diff = (lora_avg - baseline_avg) / max(1e-8, abs(baseline_avg)) * 100
        print(
            f"\nLoRA vs Baseline: {baseline_avg:.4f} -> {lora_avg:.4f} ({diff:+.2f} %)"
        )


if __name__ == "__main__":
    main()
