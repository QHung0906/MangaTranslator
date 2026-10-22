import html
import time
from typing import Any, Dict, List


class NotificationStack:
    def __init__(self, max_items: int = 3, default_ttl: float = 3.0):
        self.max_items = max_items
        self.default_ttl = default_ttl
        self._items: List[Dict[str, Any]] = []

    def push(self, message: str, level: str = "info", ttl: float | None = None) -> None:
        message = (message or "").strip()
        if not message:
            return

        if ttl is None:
            ttl = self.default_ttl

        now = time.time()
        self._prune(now)

        # 1. Tránh lặp lại chính xác tin nhắn liền kề: chỉ refresh timestamp
        if self._items and self._items[-1]["text"] == message:
            self._items[-1]["created_at"] = now
            self._items[-1]["ttl"] = ttl
            self._items[-1]["level"] = level
            return

        # 2. Nếu là thông báo chuyển model, loại bỏ thông báo chuyển model cũ trước đó
        if "Đã chuyển sang model" in message or "đã chuyển sang model" in message.lower():
            self._items = [i for i in self._items if "chuyển sang model" not in i["text"].lower()]

        item = {
            "id": int(now * 1000),
            "text": message,
            "level": level,
            "created_at": now,
            "ttl": ttl,
        }
        self._items.append(item)

        # Giữ tối đa max_items mới nhất
        if len(self._items) > self.max_items:
            self._items = self._items[-self.max_items :]

    def clear(self) -> None:
        self._items.clear()

    def _prune(self, now: float | None = None) -> None:
        if now is None:
            now = time.time()
        self._items = [i for i in self._items if (now - i["created_at"]) < i["ttl"]]

    def get_active_items(self) -> List[Dict[str, Any]]:
        self._prune()
        return list(self._items)

    def render_html(self, current_model: str = "") -> str:
        active_items = self.get_active_items()

        # CSS Stylesheet cho Toast Stack
        style = """
<style>
.toast-stack-container {
    display: flex;
    flex-direction: column;
    gap: 4px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 13px;
    min-height: 28px;
}
.toast-model-header {
    font-size: 13px;
    color: var(--body-text-color, #abb2bf);
    display: flex;
    align-items: center;
    gap: 6px;
    font-weight: 500;
}
.toast-model-code {
    background: var(--background-fill-secondary, #282c34);
    border: 1px solid var(--border-color-primary, #3e4451);
    padding: 1px 6px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 12px;
    color: var(--primary-500, #61afef);
}
.toast-item {
    padding: 4px 10px;
    border-radius: 5px;
    font-size: 12px;
    line-height: 1.4;
    animation: toastFadeSlide 3s cubic-bezier(0.4, 0, 0.2, 1) forwards;
    display: flex;
    align-items: center;
    gap: 6px;
    word-break: break-word;
    border-left: 3px solid transparent;
    overflow: hidden;
}
.toast-error {
    background: rgba(224, 108, 117, 0.15);
    border-left-color: #e06c75;
    color: #e06c75;
}
.toast-warning {
    background: rgba(229, 192, 123, 0.15);
    border-left-color: #e5c07b;
    color: #e5c07b;
}
.toast-info {
    background: rgba(97, 175, 239, 0.15);
    border-left-color: #61afef;
    color: #61afef;
}
.toast-success {
    background: rgba(152, 195, 121, 0.15);
    border-left-color: #98c379;
    color: #98c379;
}
@keyframes toastFadeSlide {
    0% {
        opacity: 0;
        transform: translateY(-4px);
        max-height: 50px;
        margin-top: 2px;
        padding: 4px 10px;
    }
    10% {
        opacity: 1;
        transform: translateY(0);
        max-height: 50px;
        margin-top: 2px;
        padding: 4px 10px;
    }
    70% {
        opacity: 1;
        transform: translateY(0);
        max-height: 50px;
        margin-top: 2px;
        padding: 4px 10px;
    }
    90% {
        opacity: 0;
        transform: translateY(-3px);
        max-height: 50px;
        margin-top: 2px;
        padding: 4px 10px;
    }
    100% {
        opacity: 0;
        max-height: 0;
        margin-top: 0;
        margin-bottom: 0;
        padding-top: 0;
        padding-bottom: 0;
        border: none;
        overflow: hidden;
        pointer-events: none;
    }
}
</style>
"""
        model_display = ""
        if current_model:
            escaped_model = html.escape(current_model)
            model_display = f'<div class="toast-model-header"><strong>Model:</strong> <span class="toast-model-code">{escaped_model}</span></div>'

        toasts_html = []
        for item in active_items:
            lvl = item.get("level", "info")
            escaped_text = html.escape(item.get("text", ""))
            toasts_html.append(f'<div class="toast-item toast-{lvl}">{escaped_text}</div>')

        return f'{style}<div class="toast-stack-container">{model_display}{"".join(toasts_html)}</div>'
