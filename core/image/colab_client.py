import io
import json
import base64
import requests
from typing import Optional, Tuple, Dict, Any
from PIL import Image

from utils.logging import log_message
from utils.exceptions import ModelError


class ColabClientManager:
    """Manages connection and inpainting requests to a remote Google Colab GPU server."""

    def __init__(self, server_url: str = "", verbose: bool = False):
        self.server_url = (server_url or "").strip().rstrip("/")
        self.verbose = verbose

    def has_server(self) -> bool:
        return bool(self.server_url and self.server_url.startswith("http"))

    def test_connection(self) -> Tuple[bool, str]:
        """Test if the remote Colab server is reachable and active."""
        if not self.has_server():
            return False, "Chưa nhập Colab Server URL (vd: https://xxxx.trycloudflare.com)"

        url = f"{self.server_url}/health"
        try:
            resp = requests.get(url, timeout=15)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    gpu = data.get("gpu", "N/A")
                    model = data.get("model", "N/A")
                    return True, f"Kết nối Colab thành công! GPU: {gpu} | Model: {model}"
                except Exception:
                    return True, "Kết nối Colab thành công! (200 OK)"
            return False, f"Server trả về mã HTTP {resp.status_code}: {resp.text[:150]}"
        except requests.exceptions.Timeout:
            return False, "Hết thời gian chờ (Timeout) khi kết nối tới Colab Server."
        except Exception as e:
            return False, f"Không thể kết nối tới Colab Server: {e}"

    def run_inference(
        self,
        model_name: str,
        image_pil: Image.Image,
        mask_pil: Optional[Image.Image] = None,
        prompt: str = "clean background, remove text",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        seed: int = -1,
        timeout: int = 180,
    ) -> Image.Image:
        """Send image and mask to remote Colab server for GPU inpainting."""
        if not self.has_server():
            raise ModelError(
                "Chưa cấu hình Colab Server URL. Vui lòng dán link Colab Tunnel (vd: https://xxxx.trycloudflare.com) vào cài đặt."
            )

        # Convert image to PNG bytes
        img_buffer = io.BytesIO()
        image_pil.save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()

        mask_bytes = None
        if mask_pil is not None:
            mask_buffer = io.BytesIO()
            mask_pil.save(mask_buffer, format="PNG")
            mask_bytes = mask_buffer.getvalue()

        url = f"{self.server_url}/inpaint"
        files = {
            "image": ("image.png", img_bytes, "image/png"),
        }
        if mask_bytes:
            files["mask"] = ("mask.png", mask_bytes, "image/png")

        data = {
            "prompt": prompt,
            "width": str(width),
            "height": str(height),
            "num_inference_steps": str(max(1, min(50, int(num_inference_steps)))),
            "guidance_scale": str(float(guidance_scale)),
            "seed": str(int(seed)),
            "model_name": model_name,
        }

        try:
            log_message(
                f"[Colab AI] Sending request to {url} (Image: {width}x{height}, Model: {model_name}, Steps: {num_inference_steps})...",
                verbose=self.verbose,
                always_print=self.verbose,
            )
            resp = requests.post(url, files=files, data=data, timeout=timeout)

            if resp.status_code == 200:
                content = resp.content
                if content.startswith(b"\x89PNG") or content.startswith(b"\xff\xd8"):
                    log_message(
                        f"[Colab AI] Inpainting success 200 OK from Colab Server",
                        verbose=self.verbose,
                        always_print=self.verbose,
                    )
                    return Image.open(io.BytesIO(content)).convert("RGB")

                # Try parsing base64 if returned as JSON
                try:
                    res_json = resp.json()
                    if "image" in res_json:
                        return Image.open(io.BytesIO(base64.b64decode(res_json["image"]))).convert("RGB")
                except Exception:
                    pass

                try:
                    return Image.open(io.BytesIO(content)).convert("RGB")
                except Exception as e:
                    raise ModelError(f"Failed to parse image from Colab Server: {e}")

            raise ModelError(
                f"Colab Server returned HTTP {resp.status_code}: {resp.text[:300]}"
            )

        except requests.exceptions.Timeout:
            raise ModelError(
                f"Colab Server timed out after {timeout}s. Kiểm tra lại xem Colab GPU còn đang chạy không."
            )
        except requests.exceptions.RequestException as e:
            raise ModelError(
                f"Lỗi kết nối Colab Server: {e}. Vui lòng kiểm tra lại link Cloudflare Tunnel."
            )
