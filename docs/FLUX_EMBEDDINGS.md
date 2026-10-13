# FLUX.2 Klein Zero Text-Encoder Architecture & Embeddings Guide

## 1. Why Use Precomputed Prompt Embeddings?
**FLUX.2 Klein (4B / 9B)** utilizes a massive Qwen2.5-based Text Encoder that consumes approximately **2.82 GB** of disk space and requires ~3 GB of system RAM / GPU VRAM during initialization.

However, in manga and comic inpainting tasks, the prompt is **static and constant**:
> *"Remove all text, including hand-drawn Japanese sound effects and onomatopoeia. Preserve character line art, screentones, panel borders, and background details exactly as they appear..."*

Reloading the heavy text encoder on every launch is completely redundant. By precomputing the prompt embeddings into lightweight `.pt` tensors:
- **Saves Disk Space**: Safely frees over **5.6 GB** of disk space (by purging `text_encoder/model.safetensors` and associated Hugging Face blobs).
- **Conserves System RAM / VRAM**: Reduces memory overhead by ~2.8 GB.
- **Instant Pipeline Startup**: Cuts pipeline load time from **~20 seconds down to ~2 seconds**.

---

## 2. Embedding File Placement

Place the `.pt` files in the following directory:
```text
MangaTranslator/
└── models/
    └── flux/
        └── embeddings/
            ├── flux4b_manga_embeds.pt   <-- (Required for local 4B inpainting, ~7.8 MB)
            └── flux9b_manga_embeds.pt   <-- (Backup for 9B model, ~12.5 MB)
```

---

## 3. How to Obtain the Embedding Files

### Option 1: Direct Download (Recommended - Instant)
Because the precomputed `.pt` tensors are extremely compact (~7.8 MB for 4B), the fastest method is downloading them directly into `models/flux/embeddings/`:
- **Google Drive Download Links**:
  - [flux4b_manga_embeds.pt (4B)](https://drive.google.com/file/d/14LTQz8OtJBZiKP1MWIPmMjj5CAsb3Xxp/view?usp=drive_link) (~7.8 MB)
  - [flux9b_manga_embeds.pt (9B)](https://drive.google.com/file/d/1XdiNH4daPo0B5ECux32-OpnKWzUBN0v8/view?usp=drive_link) (~12.5 MB)

Once placed in `models/flux/embeddings/`, MangaTranslator will automatically detect them and bypass the text encoder during startup.

---

### Option 2: Self-Extraction via Export Script (For Developers)
If you wish to export the embeddings locally or customize the prompt, run the bundled utility script:

```bash
# Export embeddings for 4B:
python tools/export_flux_embeddings.py --variant 4b

# Export embeddings for 9B:
python tools/export_flux_embeddings.py --variant 9b
```

The script will fetch the model components, encode the inpainting prompt, and save the tensor directly to `models/flux/embeddings/`.

---

## 4. Deleting the Text Encoder to Free Disk Space

Once `flux4b_manga_embeds.pt` is in place, you can safely remove the heavy text encoder weights:

1. Navigate to the Flux cache directory:
   `models/flux/models--Disty0--FLUX.2-klein-4B-SDNQ-4bit-dynamic/`
2. Delete `model.safetensors` under:
   `snapshots/<snapshot_id>/text_encoder/model.safetensors`
3. Delete the corresponding ~2.82 GB blob file under:
   `blobs/<blob_id>`
