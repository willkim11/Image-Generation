# Stable Diffusion XL Image Generator

A simple and powerful Python script to generate high-quality images using **Stable Diffusion XL (SDXL) 1.0**. This project allows you to generate images from text prompts directly from your terminal using a single `.py` file.

## 🌟 Features

* **State-of-the-Art Model**: Uses `stabilityai/stable-diffusion-xl-base-1.0` for high-resolution image generation.
* **High-Quality Sampling**: Implements `DPMSolverMultistepScheduler` for superior detail and faster convergence.
* **GPU Acceleration**: Automatically detects and utilizes CUDA (GPU) for fast inference. Falls back to CPU if GPU is unavailable.
* **Interactive CLI**: Simple command-line interface to input prompts, filenames, and seeds continuously.
* **Customizable**: Pre-configured for high-quality results (70 inference steps, 9.0 guidance scale).

## 🛠️ Prerequisites

### Hardware
* **GPU (Recommended)**: NVIDIA GPU with at least 8GB VRAM (12GB+ recommended for SDXL).
* **CPU**: Possible, but image generation will be significantly slower.

### Software
* Python 3.8 or higher

## 📦 Installation

1.  **Clone the repository** (or download the files):
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Install required Python packages**:
    It is recommended to use a virtual environment.
    ```bash
    pip install torch torchvision torchaudio
    pip install diffusers transformers accelerate safetensors
    ```

## 🚀 Usage

1.  **Run the script**:
    ```bash
    python sd_project.py
    ```

2.  **Follow the on-screen prompts**:
    * **First Run**: The script will download the SDXL model weights from Hugging Face (approx. 6GB+). This happens only once.
    * **Input Prompt**: Enter the English text description of the image you want to generate.
    * **Filename**: Enter the name for the output file (e.g., `scifi_city.png`).
    * **Seed**: (Optional) Enter a number for reproducible results, or press Enter for a random seed.

3.  **Check results**:
    Generated images will be saved in the `./generated_images` folder.

### Example Interaction
```text
Stable Diffusion XL 모델 'stabilityai/stable-diffusion-xl-base-1.0' 로딩 중...
모델 로딩 완료.
생성된 이미지는 './generated_images' 폴더에 저장됩니다.

생성하고 싶은 이미지에 대한 프롬프트를 입력하세요 (종료하려면 'q' 입력): a futuristic city with flying cars, cyberpunk style, 8k resolution
저장할 파일 이름을 입력하세요 (예: my_image.png): cyberpunk_city.png
사용할 시드를 입력하세요 (생략하려면 Enter): 42

이미지 생성 시작: 'a futuristic city with flying cars, cyberpunk style, 8k resolution'
시드 42 사용.
100%|█████████████████████████████████████████| 70/70 [00:15<00:00,  4.50it/s]
이미지가 저장되었습니다: ./generated_images/cyberpunk_city.png

## ⚙️ Configuration

You can modify the `generate_image` function arguments in `sd_project.py` to tweak the generation settings:

* `num_inference_steps`: Default is **70**. Higher values generally produce better quality but take longer.
* `guidance_scale`: Default is **9.0**. Controls how closely the image follows the text prompt.

## 🤝 Contributing

Feel free to submit issues or pull requests if you have suggestions for improvements!

## 📜 License

This project uses the Stable Diffusion XL 1.0 model by Stability AI. Please refer to the [Stable Diffusion XL License](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) for usage terms.

### 💡 팁:
위 내용 중 `YOUR_USERNAME`과 `YOUR_REPOSITORY_NAME` 부분만 본인의 깃허브 주소에 맞게 수정해주시면 완벽합니다.