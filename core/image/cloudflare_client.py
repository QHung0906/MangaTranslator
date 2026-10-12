import io
import json
import base64
import requests
from typing import List, Optional, Tuple, Dict, Any
from PIL import Image

from utils.logging import log_message
from utils.exceptions import ModelError

MODEL_MAP = {
    "flux_klein_4b": "@cf/black-forest-labs/flux-2-klein-4b",
    "flux_klein_9b": "@cf/black-forest-labs/flux-2-klein-9b",
    "flux_dev": "@cf/black-forest-labs/flux-2-dev",
    "sd_inpainting": "@cf/runwayml/stable-diffusion-v1-5-inpainting",
}


class CloudflareAccount:
    """Represents a Cloudflare Account ID and API Token pair."""

    def __init__(self, account_id: str, api_token: str):
        self.account_id = account_id.strip()
        self.api_token = api_token.strip()
        self.is_exhausted = False
        self.last_error = ""

    def __repr__(self) -> str:
        masked_token = (
            self.api_token[:4] + "..." + self.api_token[-4:]
            if len(self.api_token) > 8
            else "***"
        )
        return f"CloudflareAccount(id={self.account_id}, token={masked_token})"


class CloudflareClientManager:
    """Manages Cloudflare Workers AI requests with multi-account rotation and failover."""

    def __init__(self, accounts_text: str = "", verbose: bool = False):
        self.verbose = verbose
        self.accounts: List[CloudflareAccount] = self._parse_accounts(accounts_text)
        self.current_idx = 0

    def _parse_accounts(self, accounts_text: str) -> List[CloudflareAccount]:
        """Parse account entries from multi-line text (ACCOUNT_ID:API_TOKEN)."""
        accounts = []
        if not accounts_text:
            return accounts

        lines = accounts_text.strip().splitlines()
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            # Accept separators: ':' or ','
            separator = ":" if ":" in line else ("," if "," in line else None)
            if separator:
                parts = line.split(separator, 1)
                acc_id, token = parts[0].strip(), parts[1].strip()
                if acc_id and token:
                    accounts.append(CloudflareAccount(acc_id, token))

        return accounts

    def has_accounts(self) -> bool:
        return len(self.accounts) > 0

    def get_current_account(self) -> Optional[CloudflareAccount]:
        if not self.accounts:
            return None
        # Try to find next non-exhausted account
        for _ in range(len(self.accounts)):
            acc = self.accounts[self.current_idx]
            if not acc.is_exhausted:
                return acc
            self.current_idx = (self.current_idx + 1) % len(self.accounts)

        # If all exhausted, reset state to try again
        for acc in self.accounts:
            acc.is_exhausted = False
        return self.accounts[self.current_idx]

    def rotate_to_next(self, reason: str = ""):
        """Mark current account as exhausted and rotate to next."""
        if not self.accounts:
            return
        curr = self.accounts[self.current_idx]
        curr.is_exhausted = True
        curr.last_error = reason
        old_idx = self.current_idx
        self.current_idx = (self.current_idx + 1) % len(self.accounts)
        log_message(
            f"Rotating Cloudflare account [{old_idx + 1}/{len(self.accounts)}] -> [{self.current_idx + 1}/{len(self.accounts)}] ({reason})",
            always_print=True,
        )

    def _parse_image_response(self, response: requests.Response) -> Image.Image:
        """Parse binary or base64 JSON response into PIL Image."""
        content = response.content
        content_type = response.headers.get("content-type", "")

        # Check binary PNG or JPEG magic bytes
        if content.startswith(b"\x89PNG") or content.startswith(b"\xff\xd8"):
            return Image.open(io.BytesIO(content)).convert("RGB")

        # Check JSON response
        try:
            data = response.json()
            if isinstance(data, dict):
                # Standard Cloudflare response wrapping
                res = data.get("result", data)
                if isinstance(res, dict) and "image" in res:
                    img_b64 = res["image"]
                    return Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
                elif isinstance(res, str):
                    return Image.open(io.BytesIO(base64.b64decode(res))).convert("RGB")
        except Exception:
            pass

        # If direct binary without known magic (or octet-stream), try opening anyway
        try:
            return Image.open(io.BytesIO(content)).convert("RGB")
        except Exception as e:
            raise ModelError(
                f"Failed to decode image from Cloudflare Workers AI: {e} (Raw: {response.text[:200]})"
            )

    def run_inference(
        self,
        model_name: str,
        image_pil: Image.Image,
        mask_pil: Optional[Image.Image] = None,
        prompt: str = "Remove all text.",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 4,
        guidance_scale: float = 2.5,
        timeout: int = 120,
    ) -> Image.Image:
        """Run image inference on Cloudflare Workers AI with automatic failover."""
        if not self.has_accounts():
            raise ModelError(
                "No Cloudflare accounts configured. Please enter 'ACCOUNT_ID:API_TOKEN' in settings."
            )

        # Resolve full model path
        resolved_model = MODEL_MAP.get(model_name, model_name)
        if not resolved_model.startswith("@cf/"):
            resolved_model = f"@cf/{resolved_model}"

        # Convert image to PNG bytes
        img_buffer = io.BytesIO()
        image_pil.save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()

        mask_bytes = None
        if mask_pil is not None:
            mask_buffer = io.BytesIO()
            mask_pil.save(mask_buffer, format="PNG")
            mask_bytes = mask_buffer.getvalue()

        total_attempts = len(self.accounts)
        last_exception = None

        for attempt in range(total_attempts):
            account = self.get_current_account()
            if not account:
                break

            url = f"https://api.cloudflare.com/client/v4/accounts/{account.account_id}/ai/run/{resolved_model}"
            headers = {
                "Authorization": f"Bearer {account.api_token}",
                "User-Agent": "MangaTranslator/1.22",
            }

            try:
                log_message(
                    f"[Cloudflare AI] Calling model '{resolved_model}' via Account '{account.account_id[:6]}...' (Image: {width}x{height}, Steps: {num_inference_steps})",
                    verbose=self.verbose,
                    always_print=self.verbose,
                )

                if "stable-diffusion-v1-5-inpainting" in resolved_model:
                    # SD 1.5 inpainting requires JSON payload with image and mask as uint8 array (list of integers)
                    json_payload = {
                        "prompt": prompt,
                        "image": list(img_bytes),
                        "negative_prompt": "text, watermark, low quality, artifacts",
                        "num_steps": max(1, min(50, int(num_inference_steps))),
                        "guidance": float(guidance_scale),
                    }
                    if mask_bytes:
                        json_payload["mask"] = list(mask_bytes)
                    resp = requests.post(url, headers=headers, json=json_payload, timeout=timeout)
                else:
                    # FLUX models (Klein 4B/9B, Dev) via multipart
                    files = {
                        "input_image_0": ("image.png", img_bytes, "image/png")
                    }
                    data = {
                        "prompt": prompt,
                        "width": str(width),
                        "height": str(height),
                        "guidance": str(guidance_scale),
                    }
                    if "dev" in resolved_model:
                        data["steps"] = str(max(20, min(50, int(num_inference_steps))))
                    resp = requests.post(url, headers=headers, data=data, files=files, timeout=timeout)

                if resp.status_code == 200:
                    log_message(
                        f"[Cloudflare AI] Success 200 OK from Account '{account.account_id[:6]}...' for '{resolved_model}'",
                        verbose=self.verbose,
                        always_print=self.verbose,
                    )
                    return self._parse_image_response(resp)

                # Extract friendly error message from Cloudflare JSON response if available
                error_detail = resp.text[:300]
                try:
                    err_json = resp.json()
                    if isinstance(err_json, dict) and "errors" in err_json and err_json["errors"]:
                        error_detail = "; ".join(
                            e.get("message", str(e)) for e in err_json["errors"] if isinstance(e, dict)
                        ) or resp.text[:300]
                except Exception:
                    pass

                # Check for rate limit / auth / quota exceeded
                if resp.status_code == 400 and "flagged" in error_detail.lower():
                    flag_msg = (
                        f"[Cloudflare AI] Ảnh bị bộ lọc kiểm duyệt an toàn của Cloudflare gắn cờ (Flagged / NSFW Filter). "
                        f"Tự động bỏ qua Cloudflare và fallback sang tô nền OpenCV cho vùng này."
                    )
                    log_message(flag_msg, always_print=True)
                    raise ModelError(f"Safety filter flagged: {error_detail}")

                if resp.status_code in (429, 401, 402, 403):
                    if resp.status_code == 401:
                        reason = f"HTTP 401 Unauthorized: Invalid API Token or missing permissions ({error_detail})"
                    elif resp.status_code == 403:
                        reason = f"HTTP 403 Forbidden: Invalid Account ID or Token has no access to this account ({error_detail})"
                    elif resp.status_code == 429:
                        reason = f"HTTP 429 Rate Limit / Daily Neurons Quota Exceeded (10k Neurons/day free limit reached)"
                    else:
                        reason = f"HTTP {resp.status_code}: {error_detail}"

                    log_message(
                        f"[Cloudflare AI] Account '{account.account_id[:6]}...' failed -> {reason}",
                        always_print=True,
                    )
                    self.rotate_to_next(reason)
                    continue
                else:
                    err_msg = f"[Cloudflare AI] HTTP {resp.status_code} Error: {error_detail}"
                    log_message(err_msg, always_print=True)
                    raise ModelError(err_msg)

            except requests.RequestException as e:
                last_exception = e
                log_message(
                    f"[Cloudflare AI] Network error connecting to Cloudflare with Account '{account.account_id[:6]}...': {e}",
                    always_print=True,
                )
                self.rotate_to_next(f"Network error: {e}")
                continue
            except ModelError as e:
                if "flagged" in str(e).lower():
                    raise e
                last_exception = e
                log_message(f"[Cloudflare AI] Model Error: {e}", always_print=True)
                self.rotate_to_next(str(e))
                continue

        raise ModelError(
            f"All configured Cloudflare accounts failed or exhausted daily Neurons limit. Last error: {last_exception}"
        )


def test_cloudflare_account(account_id: str, api_token: str) -> Tuple[bool, str]:
    """
    Test if a Cloudflare Account ID and API Token pair can access Workers AI.
    Uses the free models search endpoint (does not consume any Neurons).
    """
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id.strip()}/ai/models/search"
    headers = {"Authorization": f"Bearer {api_token.strip()}"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            return True, "Kết nối thành công! Account ID và API Token hoạt động tốt."
        if resp.status_code == 401:
            return False, "HTTP 401 Unauthorized: Sai API Token hoặc Token chưa được cấp quyền 'Workers AI Read'."
        if resp.status_code == 403:
            return False, "HTTP 403 Forbidden: Sai Account ID hoặc Token không có quyền truy cập vào Account này."
        return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return False, f"Lỗi kết nối mạng: {e}"


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        raw = sys.argv[1]
        sep = ":" if ":" in raw else ("," if "," in raw else None)
        if sep:
            aid, tok = raw.split(sep, 1)
            ok, msg = test_cloudflare_account(aid, tok)
            print(f"[{'OK' if ok else 'FAIL'}] {msg}")
        else:
            print("Cú pháp: python -m core.image.cloudflare_client ACCOUNT_ID:API_TOKEN")
    else:
        print("Cú pháp: python -m core.image.cloudflare_client ACCOUNT_ID:API_TOKEN")

