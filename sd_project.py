# 1. 필요한 라이브러리 임포트
import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from PIL import Image
import os

# 2. 설정 변수
# 모델 이름 (Hugging Face Hub에서 사용 가능한 Stable Diffusion XL 모델)
# 'stabilityai/stable-diffusion-xl-base-1.0'이 가장 일반적인 SDXL 버전입니다.
MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0" 

# 생성될 이미지 저장 경로
OUTPUT_DIR = "./generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# GPU 사용 가능 여부 확인 및 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 3. Stable Diffusion XL 파이프라인 로드
print(f"Stable Diffusion XL 모델 '{MODEL_ID}' 로딩 중... (장시간 소요될 수 있습니다)")
try:
    # 안전하게 모델을 로드하기 위해 revision='fp16', torch_dtype=torch.float16 권장 (GPU 사용 시)
    # CPU만 사용하는 경우 revision 및 torch_dtype은 생략하거나 torch.float32 사용
    if DEVICE == "cuda":
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID, 
            torch_dtype=torch.float16
        ).to(DEVICE)
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            MODEL_ID
        ).to(DEVICE)
    print("모델 로딩 완료.")
except Exception as e:
    print(f"모델 로딩 중 오류 발생: {e}")
    print("GPU 메모리가 부족하거나, 인터넷 연결에 문제가 있을 수 있습니다.")
    print("CPU 모드로 다시 시도합니다...")
    # 오류 발생 시 CPU 모드로 재시도 (더 느릴 수 있음)
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID).to("cpu")
    print("모델 CPU 모드로 로딩 완료.")

# DPMSolverMultistepScheduler로 변경
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

# 4. 이미지 생성 함수
def generate_image(prompt: str, filename: str, num_inference_steps: int = 70, guidance_scale: float = 9.0, seed: int = None):
    """
    Stable Diffusion XL을 사용하여 이미지를 생성하고 저장합니다.

    Args:
        prompt (str): 이미지를 생성할 텍스트 프롬프트.
        filename (str): 저장될 이미지 파일 이름 (확장자 포함).
        num_inference_steps (int): 추론 스텝 수. 높을수록 품질이 좋아지지만 느려집니다.
        guidance_scale (float): 텍스트 프롬프트에 대한 이미지의 일치도. 높을수록 프롬프트에 충실합니다.
        seed (int, optional): 난수 시드. 동일한 시드를 사용하면 동일한 이미지를 생성합니다.
    """
    
    generator = None
    if seed is not None:
        generator = torch.Generator(DEVICE).manual_seed(seed)
        print(f"시드 {seed} 사용.")
    
    print(f"\n이미지 생성 시작: '{prompt}'")
    with torch.no_grad(): # 추론 시에는 그래디언트 계산 불필요
        image = pipe(
            prompt, 
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]
    
    save_path = os.path.join(OUTPUT_DIR, filename)
    image.save(save_path)
    print(f"이미지가 저장되었습니다: {save_path}")
    return save_path

# 5. 메인 실행 블록
if __name__ == "__main__":
    print(f"생성된 이미지는 '{OUTPUT_DIR}' 폴더에 저장됩니다.")

    # 사용자가 직접 프롬프트 입력받기
    while True:
        user_prompt = input("\n생성하고 싶은 이미지에 대한 프롬프트를 입력하세요 (종료하려면 'q' 입력): ")
        if user_prompt.lower() == 'q':
            break
        
        user_filename = input("저장할 파일 이름을 입력하세요 (예: my_image.png): ")
        user_seed_str = input("사용할 시드를 입력하세요 (생략하려면 Enter): ")
        user_seed = int(user_seed_str) if user_seed_str.isdigit() else None
        
        # 추가적으로 num_inference_steps, guidance_scale 등도 입력받을 수 있습니다.
        generate_image(user_prompt, user_filename, seed=user_seed)

    print("\n프로젝트 종료.")