"""
LoRA fine-tuning script for Stable Diffusion (Cross-Attention selective update)

- 이미지 + 캡션(JSON) 기반 텍스트-이미지 LoRA 학습
- Accelerate 기반 FP16 / gradient accumulation 지원
- UNet의 Cross-Attention 모듈에만 LoRA 적용
- 학습 종료 후 LoRA 파라미터만 별도 저장 (base 모델과 독립적으로 로드 가능)

사용 전제:
- diffusers == 0.32.2
- StableDiffusionPipeline v1.5 계열
"""

import os
import json
import argparse
from typing import List, Dict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from accelerate import Accelerator
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor

# from transformers import CLIPTokenizer


# -----------------------------
# Dataset 정의
# -----------------------------
class CaptionedDataset(Dataset):
    """
    이미지 폴더 + captions.json 구조를 사용하는 데이터셋
    captions.json 형식 예시:
    {
        "idol_face1.jpg": "a glamorous portrait of a female korean pop star ...",
        "idol_face2.jpg": "..."
    }
    """

    def __init__(
        self,
        image_dir: str,
        caption_file: str,
        resolution: int = 512,
    ):
        self.image_dir = image_dir

        with open(caption_file, "r", encoding="utf-8") as f:
            self.captions: Dict[str, str] = json.load(f)

        # 실제 존재하는 파일만
        # self.image_files = list(self.captions.keys())
        self.items: List[str] = [
            fname
            for fname in self.captions.keys()
            if os.path.isfile(os.path.join(self.image_dir, fname))
        ]

        if not self.items:
            raise ValueError(
                "유효한 이미지/캡션 페어가 없습니다. 경로나 파일명을 확인하세요."
            )

        self.transform = transforms.Compose(
            [
                # transforms.Resize((resolution, resolution)),
                transforms.Resize(
                    (resolution, resolution), interpolation=Image.BICUBIC
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __len__(self):
        # return len(self.image_files)
        return len(self.items)

    def __getitem__(self, idx):
        # filename = self.image_files[idx]
        fname = self.items[idx]
        # image = Image.open(os.path.join(self.image_dir, filename)).convert("RGB")
        img_path = os.path.join(self.image_dir, fname)
        caption = self.captions[fname]

        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)

        # return {
        #     "pixel_values": self.transform(image),
        #     "text": self.captions[filename],
        # }
        return {
            "pixel_values": pixel_values,
            "text": caption,
        }


# -----------------------------
# LoRA 적용
# -----------------------------
def apply_lora_to_unet(unet, alpha: int, rank: int = 4):
    # * 두가지 방식
    """
    UNet의 모든 하위 모듈을 순회
    set_processor 메서드가 있는지 확인하여 Attention 모듈 찾음
    - alpha: scaling a (실제 scale = a / r)
    """

    for name, module in unet.named_modules():
        if hasattr(module, "set_processor"):
            try:
                lora = LoRAAttnProcessor(
                    hidden_size=module.to_q.in_features,
                    cross_attention_dim=module.to_v.in_features,
                    rank=rank,
                )
                # scale α/r 적용
                lora.to(unet.device)
                lora.lora_scale = alpha / rank
                module.set_processor(lora)
                print(f"[LoRA 적용] {name}")

            except Exception as e:
                print(f"[SKIP] {name}: {e}")
    return unet


def apply_lora_to_unet_targeting(unet: nn.Module, alpha: int, rank: int = 4):
    """
    UNet의 Cross-Attention 모듈(attn2)에만 LoRAAttnProcessor를 적용한다.
    - rank: LoRA 내부 랭크 r
    - alpha: scaling a (실제 scale = a / r)
    """

    # diffusers의 UNet은 attn_processors라는 dict를 가지므로
    # key 예시: "mid_block.attentions.0.transformer_blocks.0.attn2.processor"
    attn_processors = {}
    # for name, module in unet.attn_processors.items():
    for name, _ in unet.attn_processors.items():
        # self-attention(attn1) / cross-attention(attn2) 구분
        if "attn2" in name:
            # cross_attention_dim = module.to_v.in_features
            cross_attention_dim = unet.config.cross_attention_dim
        else:
            # self-attention에는 LoRA 적용하지 않음
            cross_attention_dim = None

        if cross_attention_dim is None:
            # attn1 등은 기본 processor 유지
            attn_processors[name] = unet.attn_processors[name]
            continue

        # hidden_size 추론
        if "mid_block" in name:
            hidden_size = unet.config.block_out_channels[-1]
        elif "up_blocks" in name:
            idx = int(name.split(".")[1])
            hidden_size = unet.config.block_out_channels[::-1][idx]
        elif "down_blocks" in name:
            idx = int(name.split(".")[1])
            hidden_size = unet.config.block_out_channels[idx]
        else:
            hidden_size = unet.config.block_out_channels[0]

        lora = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=rank,
        )
        # scale α/r 적용
        lora.to(unet.device)
        lora.lora_scale = alpha / rank

        attn_processors[name] = lora

    unet.set_attn_processor(attn_processors)
    return unet


# -----------------------------
# 학습 루프
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for Stable Diffusion (Cross-Attention only)"
    )

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Stable Diffusion base 모델 경로 혹은 허깅페이스 모델 이름",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        required=True,
        help="학습에 사용할 이미지 폴더 경로",
    )
    parser.add_argument(
        "--caption_file",
        type=str,
        required=True,
        help="이미지 파일명 → caption 매핑을 담은 JSON 파일 경로",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lora_weights",
        help="LoRA 가중치를 저장할 폴더 경로",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="이미지 리사이즈 해상도",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="배치 크기",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="gradient accumulation 스텝 수",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="AdamW learning rate",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="에포크 수",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help="LoRA rank (r)",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=8,
        help="LoRA scaling factor (α). 실제 scale은 α / r",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help="Accelerate mixed precision 모드",
    )

    args = parser.parse_args()
    return args


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    # parser.add_argument("--instance_data_dir", type=str, required=True)
    # parser.add_argument("--output_dir", type=str, required=True)
    # parser.add_argument("--caption_file", type=str, required=True)
    # parser.add_argument("--resolution", type=int, default=512)
    # parser.add_argument("--train_batch_size", type=int, default=1)
    # parser.add_argument("--num_train_epochs", type=int, default=10)
    # parser.add_argument("--learning_rate", type=float, default=1e-4)
    # parser.add_argument("--rank", type=int, default=4)
    # parser.add_argument(
    #     "--mixed_precision",
    #     type=str,
    #     choices=["no", "fp16", "bf16"],
    #     default="fp16",
    # )
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    # # parser.add_argument(
    # #     "--lr_scheduler", type=str, default="constant"
    # # )
    # # parser.add_argument(
    # #     "--checkpointing_steps", type=int, default=100
    # # )

    # args = parser.parse_args()

    args = parse_args()

    # accelerator = Accelerator()
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    if accelerator.is_local_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # -----------------------------
    # 모델 / 토크나이저 / 스케줄러 로드
    # -----------------------------
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        # torch_dtype=torch.float16,
        torch_dtype=torch.float16 if args.mixed_precision != "no" else torch.float32,
    ).to(accelerator.device)
    pipe.set_progress_bar_config(disable=not accelerator.is_local_main_process)

    unet = pipe.unet
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    noise_scheduler = pipe.scheduler

    # VAE, Text Encoder는 freeze
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    unet = apply_lora_to_unet(unet, alpha=args.alpha, rank=args.rank)
    # unet = apply_lora_to_unet_targeting(unet, alpha=args.alpha, rank=args.rank)

    dataset = CaptionedDataset(
        image_dir=args.instance_data_dir,
        caption_file=args.caption_file,
        resolution=args.resolution,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
    )

    # ! 이 부분에서 실수. 가중치 빠뜨렸었네
    # 전체 수행을..
    # optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate)

    lora_params = [p for n, p in unet.named_parameters() if "lora" in n.lower()]
    optimizer = torch.optim.AdamW(lora_params, lr=args.learning_rate)

    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

    unet.train()

    # -----------------------------
    # Training Loop
    # -----------------------------
    num_update_steps_per_epoch = len(dataloader) // args.gradient_accumulation_steps
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    if accelerator.is_local_main_process:
        print("***** LoRA Training *****")
        print(f"  Num examples = {len(dataset)}")
        print(f"  Num Epochs = {args.num_train_epochs}")
        print(f"  Instantaneous batch size per device = {args.train_batch_size}")
        print(f"  Total train batch size (w. parallel, accum) = {total_batch_size}")
        print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        print(f"  Total optimization steps per epoch = {num_update_steps_per_epoch}")

    for epoch in range(args.num_train_epochs):
        # for batch in dataloader:
        for step, batch in enumerate(dataloader):
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(
                    accelerator.device,
                    # dtype=torch.float16,
                    dtype=vae.dtype,
                )

                # 텍스트를 토큰화해서
                prompt = batch["text"]
                inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(accelerator.device)
                encoder_hidden_states = text_encoder(**inputs).last_hidden_state

                # VAE latent로 변환
                latents = pipe.vae.encode(pixel_values).latent_dist.sample() * 0.18215

                # 노이즈 샘플링
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=latents.device,
                ).long()

                # noisy latents 생성
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                model_pred = unet(
                    sample=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample

                # epsilon 예측 기준 MSE loss
                loss = torch.nn.functional.mse_loss(model_pred, noise)

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

        # print(f"Epoch {epoch + 1}/{args.num_train_epochs}, Loss: {loss.item():.4f}")
        if accelerator.is_local_main_process and step % 50 == 0:
            print(
                f"Epoch [{epoch + 1}/{args.num_train_epochs}] "
                f"Step [{step}/{len(dataloader)}] "
                f"Loss: {loss.item():.4f}"
            )
    if accelerator.is_local_main_process:
        print(
            f"Epoch {epoch + 1}/{args.num_train_epochs} 완료. "
            f"마지막 스텝 Loss: {loss.item():.4f}"
        )

    # -----------------------------
    # LoRA 가중치만 추출해서 저장
    # -----------------------------
    if accelerator.is_local_main_process:
        # 현재 UNet state_dict에서 LoRA 관련 키만 추출
        unet_state = unet.state_dict()
        lora_weights = {k: v for k, v in unet_state.items() if "lora" in k.lower()}

        os.makedirs(args.output_dir, exist_ok=True)
        # lora_weights = {k: v for k, v in unet.state_dict().items() if "lora" in k.lower()}
        # torch.save(lora_weights, os.path.join(args.output_dir, "pytorch_lora_weights.bin"))
        save_path = os.path.join(args.output_dir, "pytorch_lora_weights.bin")
        torch.save(lora_weights, save_path)
        print("LoRA 가중치 저장 완료")


if __name__ == "__main__":
    main()
