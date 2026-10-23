import io
import json
import base64
import time
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple, Dict, Any, List
from PIL import Image

from utils.logging import log_message
from utils.exceptions import ModelError


# Shared concurrency state across all ColabClientManager instances and worker threads
_GLOBAL_SERVER_SEMAPHORES: Dict[str, threading.Semaphore] = {}
_GLOBAL_RR_INDEX: int = 0
_GLOBAL_POOL_LOCK = threading.Lock()
_FAILED_SERVER_COOLDOWN: Dict[str, float] = {}
_SESSION_DEAD_SERVERS: set = set()
_PREFLIGHT_LOCK = threading.Lock()
_PREFLIGHT_FUTURE = None
_LAST_CHECK_TS: float = 0.0
_LAST_CHECKED_INPUT: str = ""


class ColabClientManager:
    """Manages connection and inpainting requests to one or more remote Google Colab GPU servers."""

    def __init__(self, server_url: str = "", verbose: bool = False):
        self.raw_input = server_url or ""
        self.verbose = verbose
        self.server_urls = self._parse_urls(self.raw_input)
        self.server_url = self.server_urls[0] if self.server_urls else ""

    @classmethod
    def reset_dead_servers(cls):
        """Reset the list of dead servers for the current session."""
        _SESSION_DEAD_SERVERS.clear()

    @classmethod
    def trigger_background_preflight(cls, server_url: str, force: bool = False):
        """Trigger a non-blocking background check for all servers.
        
        Runs during early pipeline steps (Detect/OCR/Translate) so by the time
        inpainting starts, dead servers are already identified without slowing down the pipeline.
        """
        global _PREFLIGHT_FUTURE, _LAST_CHECK_TS, _LAST_CHECKED_INPUT
        raw_text = (server_url or "").strip()
        if not raw_text:
            return

        with _PREFLIGHT_LOCK:
            now = time.time()
            if not force and raw_text == _LAST_CHECKED_INPUT and (now - _LAST_CHECK_TS) < 120:
                return

            if _PREFLIGHT_FUTURE is not None and not _PREFLIGHT_FUTURE.done():
                if not force:
                    return

            manager = cls(server_url=raw_text)
            if not manager.has_server():
                return

            _LAST_CHECKED_INPUT = raw_text

            def _background_worker():
                global _LAST_CHECK_TS
                _LAST_CHECK_TS = time.time()
                manager.test_connection()
                alive = [u for u in manager.server_urls if u not in _SESSION_DEAD_SERVERS]
                log_message(
                    f"[Colab AI] Pre-flight kiểm tra server hoàn tất: {len(alive)}/{len(manager.server_urls)} server online.",
                    always_print=True,
                )

            executor = ThreadPoolExecutor(max_workers=1)
            _PREFLIGHT_FUTURE = executor.submit(_background_worker)

    def ensure_checked(self, timeout: float = 3.5):
        """Wait for any running background preflight check, or execute test_connection if never run."""
        global _PREFLIGHT_FUTURE
        with _PREFLIGHT_LOCK:
            fut = _PREFLIGHT_FUTURE

        if fut is not None and not fut.done():
            try:
                fut.result(timeout=timeout)
                return
            except Exception:
                pass

        global _LAST_CHECK_TS
        if _LAST_CHECK_TS == 0.0 and self.has_server():
            self.test_connection()

    def _parse_urls(self, text: str) -> List[str]:
        if not text:
            return []
        urls = []
        # Support newline, comma, semicolon separation
        cleaned = text.replace(",", "\n").replace(";", "\n")
        for line in cleaned.splitlines():
            u = line.strip().rstrip("/")
            if u and u.startswith("http"):
                if u not in urls:
                    urls.append(u)
        return urls

    def has_server(self) -> bool:
        return len(self.server_urls) > 0

    def _check_single_server(self, url: str) -> Tuple[bool, str]:
        health_url = f"{url}/health"
        try:
            resp = requests.get(health_url, timeout=3.5)
            if resp.status_code == 200:
                try:
                    data = resp.json()
                    gpu = data.get("gpu", "N/A")
                    model = data.get("model", "N/A")
                    return True, f"Online (GPU: {gpu} | Model: {model})"
                except Exception:
                    return True, "Online (200 OK)"
            return False, f"HTTP {resp.status_code}: {resp.text[:80]}"
        except requests.exceptions.Timeout:
            return False, "Hết thời gian chờ (Timeout)"
        except Exception as e:
            return False, f"Lỗi kết nối ({type(e).__name__})"

    def test_connection(self) -> Tuple[bool, str]:
        """Test if the remote Colab server(s) are reachable and active."""
        if not self.has_server():
            return False, "Chưa nhập Colab Server URL (vd: http://100.x.y.z:8000 hoặc https://xxxx.trycloudflare.com)"

        results = {}
        with ThreadPoolExecutor(max_workers=max(1, len(self.server_urls))) as executor:
            future_to_url = {
                executor.submit(self._check_single_server, url): url for url in self.server_urls
            }
            for future in as_completed(future_to_url):
                u = future_to_url[future]
                try:
                    results[u] = future.result()
                except Exception as ex:
                    results[u] = (False, str(ex))

        online_count = sum(1 for ok, _ in results.values() if ok)
        total_count = len(self.server_urls)

        # Update session dead servers set
        for url in self.server_urls:
            ok, _ = results.get(url, (False, ""))
            if ok:
                _SESSION_DEAD_SERVERS.discard(url)
            else:
                _SESSION_DEAD_SERVERS.add(url)

        if total_count == 1:
            ok, msg = results[self.server_urls[0]]
            if ok:
                return True, f"Kết nối Colab thành công! {msg}"
            return False, f"Không thể kết nối tới Colab Server: {msg}"

        # Multi-server report
        report_lines = [f"Trạng thái cụm Colab ({online_count}/{total_count} server khả dụng):"]
        for url in self.server_urls:
            ok, msg = results.get(url, (False, "Chưa kiểm tra"))
            icon = "✅" if ok else "❌"
            # Show display name without long protocol prefix
            display_url = url.replace("https://", "").replace("http://", "")
            report_lines.append(f"{icon} {display_url}: {msg}")

        overall_ok = online_count > 0
        return overall_ok, "\n".join(report_lines)

    def _acquire_server(self, exclude_urls: Optional[set] = None) -> Tuple[str, threading.Semaphore]:
        """Find an idle server, or wait on the least-congested one, excluding already-tried servers."""
        global _GLOBAL_RR_INDEX
        with _GLOBAL_POOL_LOCK:
            if not self.server_urls:
                raise ModelError("Không có Colab Server URL nào được cấu hình.")

            # Filter out servers determined dead in this session
            alive_servers = [u for u in self.server_urls if u not in _SESSION_DEAD_SERVERS]
            base_pool = alive_servers if alive_servers else list(self.server_urls)

            available = [u for u in base_pool if not exclude_urls or u not in exclude_urls]
            if not available:
                available = list(base_pool)

            n = len(available)
            start_idx = _GLOBAL_RR_INDEX % n
            _GLOBAL_RR_INDEX = (_GLOBAL_RR_INDEX + 1) % n

            ordered_urls = [available[(start_idx + i) % n] for i in range(n)]

            # 1. Try to acquire an idle server without blocking starting from rotated index
            for u in ordered_urls:
                sem = _GLOBAL_SERVER_SEMAPHORES.setdefault(u, threading.Semaphore(1))
                if sem.acquire(blocking=False):
                    return u, sem

            # 2. All are busy: pick rotated server to wait on
            selected_url = ordered_urls[0]
            sem = _GLOBAL_SERVER_SEMAPHORES.setdefault(selected_url, threading.Semaphore(1))

        # Block waiting for the selected server to be free
        sem.acquire(blocking=True)
        return selected_url, sem

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
        """Send image and mask to remote Colab server for GPU inpainting with automatic failover."""
        if not self.has_server():
            raise ModelError(
                "Chưa cấu hình Colab Server URL. Vui lòng dán link Colab (Tailscale IP hoặc Cloudflare Tunnel) vào cài đặt."
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

        data = {
            "prompt": prompt,
            "width": str(width),
            "height": str(height),
            "num_inference_steps": str(max(1, min(50, int(num_inference_steps)))),
            "guidance_scale": str(float(guidance_scale)),
            "seed": str(int(seed)),
            "model_name": model_name,
        }

        # We will attempt servers in the cluster with automatic failover
        self.ensure_checked()
        last_error = None
        tried_urls = set()

        while len(tried_urls) < len(self.server_urls):
            selected_url, sem = self._acquire_server(exclude_urls=tried_urls)
            tried_urls.add(selected_url)
            try:
                url = f"{selected_url}/inpaint"
                files = {
                    "image": ("image.png", img_bytes, "image/png"),
                }
                if mask_bytes:
                    files["mask"] = ("mask.png", mask_bytes, "image/png")

                log_message(
                    f"[Colab AI] Sending request to {url} (Image: {width}x{height}, Model: {model_name}, Steps: {num_inference_steps})...",
                    verbose=self.verbose,
                    always_print=self.verbose,
                )
                resp = requests.post(url, files=files, data=data, timeout=(4.0, timeout))

                if resp.status_code == 200:
                    _SESSION_DEAD_SERVERS.discard(selected_url)
                    _FAILED_SERVER_COOLDOWN.pop(selected_url, None)
                    content = resp.content
                    if content.startswith(b"\x89PNG") or content.startswith(b"\xff\xd8"):
                        log_message(
                            f"[Colab AI] Inpainting success 200 OK from {selected_url}",
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
                        raise ModelError(f"Failed to parse image from Colab Server {selected_url}: {e}")

                raise ModelError(
                    f"Colab Server {selected_url} returned HTTP {resp.status_code}: {resp.text[:300]}"
                )

            except requests.exceptions.Timeout as e:
                _SESSION_DEAD_SERVERS.add(selected_url)
                last_error = ModelError(
                    f"Colab Server {selected_url} timed out sau {timeout}s. Kiểm tra lại GPU Colab."
                )
                log_message(
                    f"[Colab AI] Timeout on {selected_url}. Đã loại bỏ server này khỏi phiên làm việc.",
                    always_print=True,
                )
            except Exception as e:
                _SESSION_DEAD_SERVERS.add(selected_url)
                last_error = ModelError(
                    f"Lỗi khi gửi tác vụ tới Colab Server {selected_url}: {e}"
                )
                log_message(
                    f"[Colab AI] Lỗi kết nối tới {selected_url} ({e}). Đã loại bỏ server này khỏi phiên làm việc.",
                    always_print=True,
                )
            finally:
                sem.release()

        raise last_error or ModelError("Tất cả các Colab Server trong cụm đều thất bại.")
