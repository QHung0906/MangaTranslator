[English](README.md) | [简体中文](docs/translations/zh/README.md) | [한국어](docs/translations/ko/README.md) | [日本語](docs/translations/ja/README.md)

## MangaTranslator

> [!NOTE]
> **Fork & Attribution Notice**: This repository is a customized and significantly enhanced fork of the original [meangrinch/MangaTranslator](https://github.com/meangrinch/MangaTranslator) by [meangrinch](https://github.com/meangrinch). All upstream licenses and intellectual property are fully respected and preserved under the [Apache-2.0 License](LICENSE).
>
> **Major Enhancements & Custom Features developed by [@QHung0906](https://github.com/QHung0906)**:
> 
> 1.  **Remote Colab T4 GPU Inpainting Server (High-Performance Single Worker)**:
>    - Offload heavy inpainting computation to a free Google Colab Tesla T4 GPU instance via zero-config Cloudflare Tunnel (`pycloudflared`).
>    - **Optimized Single Worker Architecture**: Dedicated pipeline leveraging full GPU memory bus bandwidth (~18–20s per inpaint pass on Tesla T4) with direct VAE decoding (tiling/slicing disabled) and auto CUDA warmup.
>    - Keeps local computers cool, saves storage, and prevents local VRAM/system RAM exhaustion.
>    - 1-click execution notebook provided in [colab_inpainting_server.ipynb](colab_inpainting_server.ipynb) with built-in real-time connection tester in the Web UI.
> 
> 2.  **Adaptive Slicing & Smart Clustering Inpainting**:
>    - Intelligent proximity-based bounding box clustering and adaptive slicing (`core/outside_text_processor.py`).
>    - Strictly bounds crop regions to $\le 1024$px, eliminating unnecessary downscaling artifacts and preserving original $1:1$ line-art detail.
>    - Drastically improves text cleaning on textured backgrounds (cabinets, screentones, complex patterns).
> 
> 3.  **Precomputed Text Embeddings & Zero Text-Encoder Architecture**:
>    - Replaces heavy text encoders (~2.8 GB disk/RAM) with precomputed prompt tensors.
>    - **Download Embeddings (Google Drive)**:
>      - [flux4b_manga_embeds.pt (4B)](https://drive.google.com/file/d/14LTQz8OtJBZiKP1MWIPmMjj5CAsb3Xxp/view?usp=drive_link) (~7.8 MB)
>      - [flux9b_manga_embeds.pt (9B)](https://drive.google.com/file/d/1XdiNH4daPo0B5ECux32-OpnKWzUBN0v8/view?usp=drive_link) (~12.5 MB)
>    - Drops pipeline initialization time from ~20s down to **~2s** while freeing **~5.64 GB** disk space and saving massive system RAM/VRAM.
>    - Step-by-step setup and export instructions available in [docs/FLUX_EMBEDDINGS.md](docs/FLUX_EMBEDDINGS.md).
> 
> 4.  **Cloudflare Workers AI Serverless Inpainting (Multi-Account)**:
>    - Leverages Cloudflare's free serverless AI tier (10,000 Neurons/day).
>    - Automatic multi-account round-robin rotation (`ACCOUNT_ID:API_TOKEN`) to seamlessly scale past daily quotas.
>    - Intelligent error handling with automatic fallback to OpenCV on flagged content or rate limits.
> 
> 5.  **Computer Vision OSB Tilt Detection & Oriented Rendering**:
>    - Advanced PCA (Principal Component Analysis) and Oriented Bounding Box (OBB) algorithm (`core/image/tilt_detection.py`).
>    - Accurately detects angle, slant, and elongation for outside-bubble text and sound effects (SFX).
>    - Dynamically rotates and renders translated text along the natural slant of the original artwork.
> 
> 6.  **Variable Font Engine (`fvar` Table) & Typography**:
>    - Full OpenType Variable Font support (`fvar` axis table) with dynamic weight and width interpolation (`core/text/font_manager.py`).
>    - High-performance font feature caching and glyph-level fallback for complex diacritics and Asian typography.
> 
> 7.  **Manual Translation Editor & Two-Phase Pipeline**:
>    - Interactive translation mode (`manual_json_translation`): extract OCR metadata to temporary JSON -> review/edit in text editor (`[Original] || [Translated]`) -> render.
>    - Allows fine-tuning translations without re-running entire detection or inpainting passes.
> 
> 8.  **Smart AI Session Manager & Batch Translation Parser**:
>    - Dedicated AI session manager (`core/services/gemini_session_manager.py`) with customizable context prompts, honorifics analysis, and symbol/onomatopoeia filtering.
>    - Two-way parser (`core/services/batch_translation_parser.py`) for synchronizing AI chat responses with the manual batch editor.
> 
> 9.  **Synchronized Gradio Web UI**:
>    - Redesigned and responsive backend controls matching active inpainting modes.
>    - Live Colab tunnel ping tester, retry options, and enhanced batch progress tracking.
> 
> 10. **Secure Cross-Platform API Vault & Zero-Leakage Masking**:
>    - **Native OS Credential Storage**: Replaces plaintext tokens in project files with native operating system secret vaults.
>      - **Windows**: Windows Credential Manager (`advapi32.dll` via `CredWriteW` / `CredReadW`).
>      - **macOS**: Native Apple Keychain (`/usr/bin/security`).
>      - **Linux**: Secret Service API / encrypted user vault (`~/.config/MangaTranslator/.credentials.enc`, `chmod 600`) with machine-bound key derivation.
>    - **Multi-Key Auto-Rotation**: Native support for rotating Google AI Studio keys and Cloudflare Workers AI accounts (`ACCOUNT_ID:API_TOKEN`).
>    - **Minimalist Web UI**: Full bullet dot-masking (`••••••••`) to prevent on-screen leakage, instant toggle buttons, and clean white layout.

Gradio-based web application for automating the translation of manga/comic page images using AI. Targets speech bubbles and text outside of speech bubbles. Supports 60 languages and custom font pack usage.

<div align="left">
  <table>
    <tr>
      <th style="text-align: left">Original</th>
      <th style="text-align: left">Translated (w/ a single click)</th>
    </tr>
    <tr>
      <td><img src="docs/images/example_original.jpg" width="400" /></td>
      <td><img src="docs/images/example_translation.jpg" width="400" /></td>
    </tr>
  </table>
</div>

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Install](#install)
- [Post-Install Setup](#post-install-setup)
- [Run](#run)
- [Remote Inpainting Server (Google Colab)](#remote-inpainting-server-google-colab)
- [Documentation](#documentation)
- [Updating](#updating)
- [License & Credits](#license--credits)

## Features

- **Detection**: Speech bubble detection & segmentation (YOLO, SAM 2.1/3)
- **Cleaning**: Inpaint speech bubbles and OSB text (FLUX.2 Klein, FLUX.1 Kontext, or OpenCV)
- **Translation**: LLM-powered OCR & translation (60 languages)
- **Rendering**: Text rendering with alignment and custom font packs
- **Upscaling**: 2x-AnimeSharpV4 for enhanced output quality
- **Processing**: Single/batch processing with directory preservation and ZIP support
- **Interfaces**: Web UI (Gradio) and CLI
- **Automation**: One-click translation; no intervention required

## Requirements

- Python 3.10+
- PyTorch (CPU, CUDA, ROCm, XPU, MPS)
- Font pack with `.ttf`/`.otf` files; included with portable package
- LLM for Japanese source text; VLM for other languages (API or local)

## Install

### Portable Package (Recommended)

Download the standalone zip from the releases page: [Portable Build](https://github.com/meangrinch/MangaTranslator/releases/tag/portable)

**Requirements:**

- **Windows:** Bundled Python/Git included; no additional requirements
- **Linux/macOS:** Python 3.10+ and Git must be installed on your system

**Setup:**

1. Extract the zip file
2. Run the setup script for your platform:
   - **Windows:** Double-click `setup.bat`
   - **Linux/macOS:** Run `./setup.sh` in terminal
3. PyTorch version is automatically detected and installed based on your system
4. Open the launcher script created in `./MangaTranslator/`:
   - **Windows:** `start-webui.bat`
   - **Linux/macOS:** `start-webui.sh`

Included font packs:

- _Komika_ (normal text)
- _Comicka_ (normal/OSB text)
- _Roboto_ (supports accents)
- _Noto Sans SC_ (Simplified Chinese)
- _Noto Sans KR_ (Korean)
- _Noto Sans JP_ (Japanese)
- _Noto Sans Thai_ (Thai)

> [!TIP]
> In the event that you need to transfer to a fresh portable package:
>
> - You can safely move the `fonts`, `models`, and `output` directories to the new portable package
> - You might be able to move the `runtime` directory over, assuming the same setup configuration is wanted

### Manual install

1. Clone and enter the repo

```bash
git clone https://github.com/meangrinch/MangaTranslator.git
cd MangaTranslator
```

2. Create and activate a virtual environment (recommended)

```bash
python -m venv venv
# Windows PowerShell/CMD
.\venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

3. Install PyTorch (see: [PyTorch Install](https://pytorch.org/get-started/locally/))

```bash
# Example (CUDA 13.0)
pip install torch==2.11.0+cu130 torchvision==0.26.0+cu130 --extra-index-url https://download.pytorch.org/whl/cu130
# Example (ROCm 7.1)
pip install torch==2.11.0+rocm7.1 torchvision==0.26.0+rocm7.1 --extra-index-url https://download.pytorch.org/whl/rocm7.1
# Example (XPU)
pip install torch==2.11.0+xpu torchvision==0.26.0+xpu --extra-index-url https://download.pytorch.org/whl/xpu
# Example (MPS/CPU)
pip install torch==2.11.0 torchvision==0.26.0
```

4. Install Nunchaku (optional, for FLUX.1 Kontext via Nunchaku backend)

- Nunchaku wheels are not on PyPI. Install directly from the v1.3.0dev20260213 GitHub release URL, matching your OS and Python version. CUDA only, and requires a 2000-series card or newer.

```bash
# Example (Windows, Python 3.13, PyTorch 2.11.0, CUDA 13.0)
pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.3.0dev20260213/nunchaku-1.3.0.dev20260213+cu13.0torch2.11-cp313-cp313-win_amd64.whl

# Example (Linux, Python 3.13, PyTorch 2.11.0, CUDA 13.0)
pip install https://github.com/nunchaku-ai/nunchaku/releases/download/v1.3.0dev20260213/nunchaku-1.3.0.dev20260213+cu13.0torch2.11-cp313-cp313-linux_x86_64.whl
```

> [!NOTE]
> Nunchaku is not necessary for the use of Flux models via the sd.cpp/SDNQ backends.

5. Install dependencies

```bash
pip install -r requirements.txt
```

## Post-Install Setup

### Models

- The application will automatically download and use all required models

### Fonts

- Put font packs as subfolders in `fonts/` with `.otf`/`.ttf` files
- Prefer filenames that include `italic`/`bold` or both so variants are detected
- Example structure:

```text
fonts/
├─ CC Wild Words/
│  ├─ CCWildWords-Regular.otf
│  ├─ CCWildWords-Italic.otf
│  ├─ CCWildWords-Bold.otf
│  └─ CCWildWords-BoldItalic.otf
└─ Komika/
   ├─ KOMIKA-HAND.ttf
   └─ KOMIKA-HANDBOLD.ttf
```

### LLM setup

- Providers: Google, OpenAI, Anthropic, SpaceXAI, Meta Model, DeepSeek, Z.ai, Moonshot AI, Xiaomi MiMo, QwenCloud, OpenCode, OpenRouter, OpenAI-Compatible
- Web UI: configure provider/model/key in the Config tab (stored locally)
- CLI: pass keys/URLs as flags or via env vars
- Env vars: `GOOGLE_API_KEY` / `GEMINI_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `SPACEXAI_API_KEY` / `XAI_API_KEY`, `META_MODEL_API_KEY` / `META_API_KEY`, `DEEPSEEK_API_KEY`, `ZAI_API_KEY`, `MOONSHOT_API_KEY`, `MIMO_API_KEY`, `QWENCLOUD_API_KEY` / `QWEN_API_KEY`, `OPENCODE_API_KEY` / `OPENCODE_ZEN_API_KEY` / `OPENCODE_GO_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_COMPATIBLE_API_KEY`
- OpenAI-Compatible provider supports local endpoints (e.g., `http://localhost:8080/v1`), and Azure OpenAI endpoints (e.g., `https://<resource>.openai.azure.com`)

> [!NOTE]
> The following models are automatically detected when used via the OpenAI-Compatible provider and receive optimized prompting. They are text-only and require two-step translation + local OCR. The `special_instructions` field maps to their corresponding glossary/terminology (one entry per line, e.g., `term -> translation`).
>
> - **YanoljaNEXT-Rosetta** (e.g., `yanolja/YanoljaNEXT-Rosetta-4B-2511-GGUF`)
> - **Hy-MT2** (e.g., `tencent/Hy-MT2-7B`). Also pre-fills the model's recommended sampling parameters

### OSB text setup (optional)

If you want to use the OSB text pipeline, you need a Hugging Face token with access to the following repositories:

- `deepghs/AnimeText_yolo`

#### Steps to create a token:

1. Sign in or create a Hugging Face account
2. Visit and accept the terms on:
   - [AnimeText_yolo](https://huggingface.co/deepghs/AnimeText_yolo)
   - [FLUX.1 Kontext (dev)](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) (optional, if using FLUX.1 Kontext via Nunchaku backend)
   - [SAM 3](https://huggingface.co/facebook/sam3) (optional, if using SAM 3)
3. Create a new access token in your Hugging Face settings with read access to gated repos ("Read access to contents of public gated repos")
4. Add the token to the app:
   - Web UI: set `hf_token` in Config
   - Env var (alternative): set `HF_TOKEN`
5. Save config to preserve the token across sessions

## Run

### Web UI (Gradio)

- **Portable package:**
  - Windows: Double-click `start-webui.bat` inside the `MangaTranslator` folder
  - Linux/macOS: Run `./start-webui.sh` inside the `MangaTranslator` folder
- **Manual install:**
  - Windows: Run `python app.py --open-browser`

Options: `--models` (default `./models`), `--fonts` (default `./fonts`), `--port` (default `7676`), `--cpu`.
First launch can take ~1–2 minutes.

Once launched, configure your LLM provider in the Config tab, then upload images and click Translate.

### CLI

Examples:

```bash
# Single image, Japanese → English, Google provider
python main.py --input <image_path> \
  --font-dir "fonts/Komika" --provider Google --google-api-key <AI...>

# Batch folder, custom source/target languages, OpenAI-Compatible provider (llama.cpp)
python main.py --input <folder_path> --batch \
  --font-dir "fonts/Komika" \
  --input-language <src_lang> --output-language <tgt_lang> \
  --provider OpenAI-Compatible --openai-compatible-url http://localhost:8080/v1 \
  --output ./output

# Single Image, Japanese → English (Google), OSB text pipeline, custom OSB text font
python main.py --input <image_path> \
  --font-dir "fonts/Komika" --provider Google --google-api-key <AI...> \
  --osb-enable --osb-font-dir "fonts/Clementine"

# Cleaning-only mode (no translation/text rendering)
python main.py --input <image_path> --cleaning-only

# Upscaling-only mode (no detection/translation, only upscale)
python main.py --input <image_path> --upscaling-only --image-upscale-mode final --image-upscale-factor 2.0

# Test mode (no translation; render placeholder text)
python main.py --input <image_path> --test-mode

# Full options
python main.py --help
```

## Remote Inpainting Server (Google Colab)

Offload speech bubble & outside-text inpainting to a free **Google Colab Tesla T4 GPU** or Kaggle environment without requiring a dedicated local GPU.

### ✨ Highlights
- **High-Performance Single Worker**: Dedicated pipeline running at full GPU memory bus bandwidth (~18–20s per inpaint pass on Tesla T4).
- **Direct VAE Decoding**: VAE tiling and slicing disabled to decode 1024x1024 in a single fused pass, cutting ~3–5s per image.
- **Zero Text-Encoder Architecture**: Bypasses the ~2.8 GB text encoder entirely via precomputed prompt embeddings (`.pt`), slashing load time to ~2 seconds and saving disk space.
- **Auto Warmup**: Primes CUDA kernels and allocators at server startup so the very first inpaint runs at peak speed.
- **Zero-Config Cloudflare Tunnel**: Generates a secure public URL (`https://xxxx.trycloudflare.com`) with no port forwarding required.

### 📥 Precomputed Embeddings Download
Download the precomputed `.pt` embeddings directly into Google Drive (`MyDrive/`) or upload to Colab:
- **FLUX.2 Klein 4B**: [flux4b_manga_embeds.pt](https://drive.google.com/file/d/14LTQz8OtJBZiKP1MWIPmMjj5CAsb3Xxp/view?usp=drive_link) (~7.8 MB)
- **FLUX.2 Klein 9B**: [flux9b_manga_embeds.pt](https://drive.google.com/file/d/1XdiNH4daPo0B5ECux32-OpnKWzUBN0v8/view?usp=drive_link) (~12.5 MB)

*(For details and local extraction scripts, see [FLUX_EMBEDDINGS.md](docs/FLUX_EMBEDDINGS.md)).*

### 🚀 Usage Instructions
1. Open [`colab_inpainting_server.ipynb`](colab_inpainting_server.ipynb) on Google Colab and select **Runtime -> Change runtime type -> T4 GPU**.
2. Run the notebook (or paste the contents of [`colab_server.py`](colab_server.py) into a cell).
3. Copy the generated Cloudflare URL (e.g. `https://xxxx.trycloudflare.com`).
4. In your local MangaTranslator Web UI:
   - Go to **Settings -> Outside Text Inpainting**
   - Inpainting Method: `Flux.2 Klein 4B`
   - Backend: `Remote Colab`
   - Colab Server URL: Paste your Cloudflare URL
   - Click **Test Connection** to verify.

## Documentation

- [FLUX Prompt Embeddings](docs/FLUX_EMBEDDINGS.md)

- [Hardware Requirements](docs/HARDWARE_REQUIREMENTS.md)
- [Recommended Fonts](docs/FONTS.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## Updating

### Portable Package

- Windows: Run `update.bat` from the portable package root
- Linux/macOS: Run `./update.sh` from the portable package root

### Manual Install

From the repo root:

```bash
git pull
pip install -r requirements.txt  # Or activate venv first if present
```

## License & credits

- License: Apache-2.0 (see [LICENSE](LICENSE))
- Original Author: [meangrinch](https://github.com/meangrinch)
- Fork Customizations: [QHung0906](https://github.com/QHung0906)
<details>
<summary><b>ML Models & Libraries</b></summary>

- YOLOv8m Speech Bubble Detector: [kitsumed](https://huggingface.co/kitsumed/yolov8m_seg-speech-bubble)
- Manga109 Speech Bubble Detector: [huyvux3005](https://huggingface.co/huyvux3005/manga109-segmentation-bubble)
- Comic Text and Bubble Detector RT-DETR-v2: [ogkalu](https://huggingface.co/ogkalu/comic-text-and-bubble-detector)
- Manga109 YOLO: [deepghs](https://huggingface.co/deepghs/manga109_yolo)
- AnimeText YOLO: [deepghs](https://huggingface.co/deepghs/AnimeText_yolo)
- SAM 2.1: Segment Anything in Images and Videos: [Meta AI](https://huggingface.co/facebook/sam2.1-hiera-large)
- SAM 3: [Meta AI](https://huggingface.co/facebook/sam3)
- Manga OCR: [kha-white](https://github.com/kha-white/manga-ocr)
- PaddleOCR-VL-1.6: [PaddlePaddle](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)
- FLUX.1 Kontext: [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
- FLUX.2 Klein 4B: [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
- FLUX.2 Klein 9B: [Black Forest Labs](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)
- Nunchaku: [Nunchaku AI](https://github.com/nunchaku-ai/nunchaku)
- SDNQ Quants: [Disty0](https://huggingface.co/Disty0)
- Unsloth Quants: [Unsloth](https://huggingface.co/unsloth)
- stable-diffusion.cpp: [leejet](https://github.com/leejet/stable-diffusion.cpp)
- 2x-AnimeSharpV4: [Kim2091](https://huggingface.co/Kim2091/2x-AnimeSharpV4)

</details>
