#  BLIP 보단 BLIP2 이 성능상 우위이므로 직접 캡셔닝 수행 ,Salesforce/blip2-flan-t5-xl,  Salesforce/blip2-opt-2.7b
# BLIP2 ,    Salesforce/blip2-flan-t5-xl 모델 사용
import os
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import requests
import glob


device = "cuda" if torch.cuda.is_available() else "cpu"


# 예시 체크포인트 (FlanT5-xl 기반)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16
)
model.to(device)

# ---------------
# BLIP2 blip2-flan-t5-xl 모델 로 캡셔닝 수행
# ---------------

# image_dir = "/content/drive/MyDrive/content/idol_faces/face1" # 5장
# image_dir = "/content/drive/MyDrive/content/idol_faces_clean2/face1" # 5장
# image_dir = "/content/drive/MyDrive/content/idol_faces_clean2/face2" # 5장
# image_dir = "/content/drive/MyDrive/content/idol_faces_clean2/face3" # 5장
image_dir = "/content/drive/MyDrive/content/idol_faces_clean2/face4"  # 5장

# 확장자가 .jpg 라고 가정 (.png 등도 있으면 수정)
image_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
# image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")) +
#     glob.glob(os.path.join(image_dir, "*.PNG")))

print(f"Found {len(image_paths)} images in {image_dir}")

for img_path in image_paths:
    # 1) 이미지 열기
    image = Image.open(img_path).convert("RGB")

    # 2) 전처리
    #    prompt를 "describe this image" 라고 예시
    # inputs = processor(image, "describe this korea's beautiful female celebrity face", return_tensors="pt").to(device, torch.float16)
    # inputs = processor(image, "describe a photo of a korea's beautiful woman. Provide a short, neutral description.", return_tensors="pt").to(device, torch.float16)
    # inputs = processor(image, return_tensors="pt").to(device)
    inputs = processor(image, "beautiful woman face.", return_tensors="pt").to(device)

    # 3) 추론
    # generated_ids = model.generate(**inputs, max_length=50, min_length =20, do_sample=False, early_stopping=True, no_repeat_ngram_size =2) # , num_beams=1,
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_length=50,
            min_length=10,
            do_sample=False,
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
            repetition_penalty=1.2,
            length_penalty=0.8,
        )

    # generated_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    captions = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    generated_text = captions[0].strip()

    # 4) 결과 출력
    print(f"Image File: {os.path.basename(img_path)}")
    print("Caption:", generated_text)
    print("-" * 60)
