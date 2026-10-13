"""
FLUX.2 Klein Prompt Embeddings Exporter (4B & 9B).
Extracts static inpainting prompt embeddings into compact .pt tensor files.
Enables zero-encoder operation, saving ~5.6 GB disk space and ~2.8 GB RAM/VRAM.
"""

import argparse
import os
from pathlib import Path
import torch

DEFAULT_KLEIN_PROMPT = (
    "Remove all text, including hand-drawn Japanese sound effects and onomatopoeia. "
    "Preserve character line art, screentones, panel borders, and background details "
    "exactly as they appear. Maintain the original contrast and shading, leaving every "
    "area where text or a sound effect was completely blank."
)

REPO_MAP = {
    "4b": "Disty0/FLUX.2-klein-4B-SDNQ-4bit-dynamic",
    "9b": "Disty0/FLUX.2-klein-9B-SDNQ-4bit-dynamic-svd-r32",
}


def export_embeddings(variant: str = "4b", custom_prompt: str = "", output_dir: str = "models/flux/embeddings"):
    variant = variant.lower()
    if variant not in REPO_MAP:
        raise ValueError(f"Invalid variant: {variant}. Choose '4b' or '9b'.")

    repo_id = REPO_MAP[variant]
    prompt = custom_prompt.strip() or DEFAULT_KLEIN_PROMPT

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target_file = out_path / f"flux{variant}_manga_embeds.pt"

    print(f"[*] Starting embedding extraction for FLUX.2 Klein {variant.upper()}...")
    print(f"[*] Repository: {repo_id}")

    os.environ["DIFFUSERS_SDNQ_TRANSFORMERS"] = "1"
    try:
        from sdnq import SDNQConfig  # noqa: F401
    except ImportError:
        pass

    from diffusers import Flux2KleinPipeline

    print("[*] Loading pipeline for prompt encoding...")
    pipe = Flux2KleinPipeline.from_pretrained(
        repo_id,
        torch_dtype=torch.bfloat16,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Encoding prompt on device: {device}...")
    pipe = pipe.to(device)

    with torch.no_grad():
        try:
            encoded = pipe.encode_prompt(prompt=prompt, prompt_2=None, device=device)
        except TypeError:
            encoded = pipe.encode_prompt(prompt=prompt, device=device)

    prompt_embeds = encoded[0] if isinstance(encoded, tuple) else encoded
    prompt_embeds = prompt_embeds.detach().to("cpu")

    torch.save(prompt_embeds, target_file)
    print(f"[✓] Successfully saved to: {target_file.resolve()}")
    print(f"[*] File size: {target_file.stat().st_size / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export FLUX.2 Klein prompt embeddings to .pt format")
    parser.add_argument("--variant", choices=["4b", "9b"], default="4b", help="Model variant (4b or 9b)")
    parser.add_argument("--prompt", type=str, default="", help="Custom prompt string (leave empty for default manga clean prompt)")
    parser.add_argument("--output-dir", type=str, default="models/flux/embeddings", help="Target output directory")
    args = parser.parse_args()

    export_embeddings(variant=args.variant, custom_prompt=args.prompt, output_dir=args.output_dir)
