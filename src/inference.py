"""
Stable Diffusion + LoRA inference script.

Usage 예시:

1) LoRA 적용 결과만 생성
    python inference.py \
        --pretrained_model runwayml/stable-diffusion-v1-5 \
        --lora_weights ./lora \
        --prompt "a korean pretty woman with large eyes, studio lighting" \
        --output_dir ./results/lora \
        --seed 2350373745 \
        --guidance_scales 7.5 10.0

2) Baseline / LoRA 둘 다 생성(동일 seed)
    python inference.py \
        --pretrained_model runwayml/stable-diffusion-v1-5 \
        --lora_weights ./lora \
        --prompt "a korean pretty woman with large eyes, studio lighting" \
        --output_dir ./results \
        --seed 2350373745 \
        --guidance_scales 7.5 \
        --generate_baseline
"""

import argparse
import os
from pathlib import Path
from typing import List

import torch
from diffusers import StableDiffusionPipeline


def create_pipeline(
    model_name_or_path: str,
    device: str,
    torch_dtype=torch.float16,
    lora_weights: str | None = None,
):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_name_or_path, torch_dtype=torch_dtype
    ).to(device)

    if lora_weights is not None:
        pipe.unet.load_attn_procs(lora_weights)

    return pipe


def generate_images(
    pipe: StableDiffusionPipeline,
    prompt: str,
    output_dir: Path,
    seed: int | None,
    guidance_scales: List[float],
    num_inference_steps: int,
    prefix: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # seed 고정 (없으면 랜덤)
    if seed is None:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()

    print(f"[{prefix}] Using seed: {seed}")
    generator = torch.Generator(device=pipe.device).manual_seed(seed)

    for g in guidance_scales:
        image = pipe(
            prompt,
            generator=generator,
            guidance_scale=g,
            num_inference_steps=num_inference_steps,
        ).images[0]

        filename = f"{prefix}_g{g}_seed{seed}.png"
        save_path = output_dir / filename
        image.save(save_path)
        print(f"[{prefix}] Saved: {save_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Base Stable Diffusion model",
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        default=None,
        help="Path to LoRA weights folder (attn_procs). If None, LoRA는 사용하지 않음.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for image generation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (None이면 랜덤)",
    )
    parser.add_argument(
        "--guidance_scales",
        type=float,
        nargs="+",
        default=[7.5],
        help="Guidance scale list",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=30,
        help="Number of diffusion steps",
    )
    parser.add_argument(
        "--generate_baseline",
        action="store_true",
        help="True면 baseline(LoRA 미적용) 이미지도 같이 생성",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_root = Path(args.output_dir)

    # 1) LoRA 적용 파이프라인
    if args.lora_weights is not None:
        lora_pipe = create_pipeline(
            args.pretrained_model, device=device, lora_weights=args.lora_weights
        )
        generate_images(
            lora_pipe,
            prompt=args.prompt,
            output_dir=output_root / "lora",
            seed=args.seed,
            guidance_scales=args.guidance_scales,
            num_inference_steps=args.num_inference_steps,
            prefix="lora",
        )

    # 2) Baseline 파이프라인 (옵션)
    if args.generate_baseline:
        base_pipe = create_pipeline(
            args.pretrained_model, device=device, lora_weights=None
        )
        generate_images(
            base_pipe,
            prompt=args.prompt,
            output_dir=output_root / "baseline",
            seed=args.seed,
            guidance_scales=args.guidance_scales,
            num_inference_steps=args.num_inference_steps,
            prefix="baseline",
        )


if __name__ == "__main__":
    main()
