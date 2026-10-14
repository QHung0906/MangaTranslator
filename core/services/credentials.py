import os
import sys
import json
from typing import List, Tuple

_TARGET_NAME = "MangaTranslator:GoogleAPIKeys"
_CACHE = {}
_LOCK = __import__("threading").Lock()

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    class CREDENTIAL(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD),
            ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR),
            ("Comment", wintypes.LPWSTR),
            ("LastWritten", wintypes.FILETIME),
            ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_byte)),
            ("Persist", wintypes.DWORD),
            ("AttributeCount", wintypes.DWORD),
            ("Attributes", ctypes.c_void_p),
            ("TargetAlias", wintypes.LPWSTR),
            ("UserName", wintypes.LPWSTR),
        ]

    def _cred_write(target_name: str, payload_str: str) -> bool:
        try:
            data_bytes = payload_str.encode("utf-16-le")
            cred = CREDENTIAL()
            cred.Flags = 0
            cred.Type = 1
            cred.TargetName = target_name
            cred.Comment = "MangaTranslator Secure API Keys"
            cred.CredentialBlobSize = len(data_bytes)
            cred.CredentialBlob = ctypes.cast(
                ctypes.c_char_p(data_bytes), ctypes.POINTER(ctypes.c_byte)
            )
            cred.Persist = 2
            cred.UserName = "MangaTranslator"
            return bool(ctypes.windll.advapi32.CredWriteW(ctypes.byref(cred), 0))
        except Exception:
            return False

    def _cred_read(target_name: str) -> str:
        try:
            pcred = ctypes.POINTER(CREDENTIAL)()
            if ctypes.windll.advapi32.CredReadW(target_name, 1, 0, ctypes.byref(pcred)):
                blob = ctypes.string_at(
                    pcred.contents.CredentialBlob, pcred.contents.CredentialBlobSize
                )
                val = blob.decode("utf-16-le")
                ctypes.windll.advapi32.CredFree(pcred)
                return val
        except Exception:
            pass
        return ""

    def _cred_delete(target_name: str) -> bool:
        try:
            return bool(ctypes.windll.advapi32.CredDeleteW(target_name, 1, 0))
        except Exception:
            return False

else:
    def _get_unix_key() -> bytes:
        import hashlib
        import uuid
        parts = []
        for p in ["/etc/machine-id", "/var/lib/dbus/machine-id"]:
            if os.path.exists(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        parts.append(f.read().strip())
                    break
                except Exception:
                    pass
        parts.append(str(getattr(os, "getuid", lambda: 1000)()))
        parts.append(str(uuid.getnode()))
        salt = b"MangaTranslator-Storage-Salt"
        return hashlib.pbkdf2_hmac("sha256", "".join(parts).encode(), salt, 50000)

    def _cipher_bytes(data: bytes, key: bytes) -> bytes:
        import hashlib
        out = bytearray()
        for i in range(0, len(data), 32):
            block = hashlib.sha256(key + i.to_bytes(4, "big")).digest()
            chunk = data[i : i + 32]
            for b, k in zip(chunk, block):
                out.append(b ^ k)
        return bytes(out)

    def _vault_file() -> str:
        from pathlib import Path
        base = os.environ.get("XDG_CONFIG_HOME")
        cfg = Path(base) / "MangaTranslator" if base else Path.home() / ".config" / "MangaTranslator"
        cfg.mkdir(parents=True, exist_ok=True)
        return str(cfg / ".credentials.enc")

    def _file_vault_write(target: str, data: str) -> bool:
        try:
            path = _vault_file()
            key = _get_unix_key()
            store = {}
            if os.path.exists(path):
                try:
                    with open(path, "rb") as f:
                        raw = _cipher_bytes(f.read(), key)
                    store = json.loads(raw.decode("utf-8"))
                except Exception:
                    store = {}
            store[target] = data
            enc = _cipher_bytes(json.dumps(store).encode("utf-8"), key)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(enc)
            return True
        except Exception:
            return False

    def _file_vault_read(target: str) -> str:
        try:
            path = _vault_file()
            if not os.path.exists(path):
                return ""
            key = _get_unix_key()
            with open(path, "rb") as f:
                raw = _cipher_bytes(f.read(), key)
            store = json.loads(raw.decode("utf-8"))
            return store.get(target, "")
        except Exception:
            return ""

    def _file_vault_delete(target: str) -> bool:
        try:
            path = _vault_file()
            if not os.path.exists(path):
                return True
            key = _get_unix_key()
            store = {}
            with open(path, "rb") as f:
                raw = _cipher_bytes(f.read(), key)
            store = json.loads(raw.decode("utf-8"))
            store.pop(target, None)
            enc = _cipher_bytes(json.dumps(store).encode("utf-8"), key)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "wb") as f:
                f.write(enc)
            return True
        except Exception:
            return False

    if sys.platform == "darwin":
        import subprocess

        def _cred_write(target_name: str, payload_str: str) -> bool:
            try:
                cmd = ["security", "add-generic-password", "-s", target_name, "-a", "MangaTranslator", "-w", payload_str, "-U"]
                res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if res.returncode == 0:
                    return True
            except Exception:
                pass
            return _file_vault_write(target_name, payload_str)

        def _cred_read(target_name: str) -> str:
            try:
                cmd = ["security", "find-generic-password", "-s", target_name, "-a", "MangaTranslator", "-w"]
                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode == 0 and res.stdout:
                    return res.stdout.strip()
            except Exception:
                pass
            return _file_vault_read(target_name)

        def _cred_delete(target_name: str) -> bool:
            try:
                cmd = ["security", "delete-generic-password", "-s", target_name, "-a", "MangaTranslator"]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception:
                pass
            return _file_vault_delete(target_name)

    else:
        import shutil
        import subprocess

        def _cred_write(target_name: str, payload_str: str) -> bool:
            if shutil.which("secret-tool"):
                try:
                    cmd = ["secret-tool", "store", "--label=MangaTranslator", "service", "MangaTranslator", "account", target_name]
                    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    p.communicate(input=payload_str.encode("utf-8"))
                    if p.returncode == 0:
                        return True
                except Exception:
                    pass
            return _file_vault_write(target_name, payload_str)

        def _cred_read(target_name: str) -> str:
            if shutil.which("secret-tool"):
                try:
                    cmd = ["secret-tool", "lookup", "service", "MangaTranslator", "account", target_name]
                    res = subprocess.run(cmd, capture_output=True, text=True)
                    if res.returncode == 0 and res.stdout:
                        return res.stdout.strip()
                except Exception:
                    pass
            return _file_vault_read(target_name)

        def _cred_delete(target_name: str) -> bool:
            if shutil.which("secret-tool"):
                try:
                    cmd = ["secret-tool", "clear", "service", "MangaTranslator", "account", target_name]
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass
            return _file_vault_delete(target_name)


def save_secure_api_keys(keys: List[str], provider: str = "Google") -> Tuple[bool, int, int]:
    seen = set()
    clean_keys = []
    dupes = 0
    for k in keys:
        item = k.strip()
        if not item or item.startswith("#"):
            continue
        if item in seen:
            dupes += 1
        else:
            seen.add(item)
            clean_keys.append(item)

    target = f"MangaTranslator:{provider}APIKeys"
    raw_str = "\n".join(clean_keys)
    if _cred_write(target, raw_str):
        with _LOCK:
            _CACHE[provider.lower()] = clean_keys
        return True, len(clean_keys), dupes
    return False, 0, 0


def get_secure_api_keys(provider: str = "Google") -> List[str]:
    pkey = provider.lower()
    with _LOCK:
        if pkey in _CACHE:
            return list(_CACHE[pkey])

    target = f"MangaTranslator:{provider}APIKeys"
    val = _cred_read(target)
    if val:
        keys = [k.strip() for k in val.splitlines() if k.strip() and not k.strip().startswith("#")]
        with _LOCK:
            _CACHE[pkey] = keys
        return list(keys)
    return []


def count_secure_api_keys(provider: str = "Google") -> int:
    return len(get_secure_api_keys(provider))


def clear_secure_api_keys(provider: str = "Google") -> bool:
    target = f"MangaTranslator:{provider}APIKeys"
    _cred_delete(target)
    with _LOCK:
        _CACHE.pop(provider.lower(), None)
    return True


def save_secure_cloudflare_accounts(raw_text: str) -> Tuple[bool, int]:
    lines = [line.strip() for line in raw_text.splitlines() if line.strip() and not line.strip().startswith("#")]
    seen = set()
    cleaned = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            cleaned.append(line)
    target = "MangaTranslator:CloudflareAccounts"
    payload = "\n".join(cleaned)
    if _cred_write(target, payload):
        with _LOCK:
            _CACHE["cloudflare_accounts"] = cleaned
        return True, len(cleaned)
    return False, 0


def get_secure_cloudflare_accounts() -> List[str]:
    with _LOCK:
        if "cloudflare_accounts" in _CACHE:
            return list(_CACHE["cloudflare_accounts"])
    target = "MangaTranslator:CloudflareAccounts"
    val = _cred_read(target)
    if val:
        accounts = [k.strip() for k in val.splitlines() if k.strip() and not k.strip().startswith("#")]
        with _LOCK:
            _CACHE["cloudflare_accounts"] = accounts
        return list(accounts)

    try:
        from ui import settings_manager
        saved = settings_manager.get_saved_settings()
        old_accs = saved.get("outside_text_cloudflare_accounts", "")
        if old_accs and not all(set(l) <= {"•", "*", "."} for l in old_accs.splitlines() if l.strip()):
            lines = [k.strip() for k in old_accs.splitlines() if k.strip() and not k.strip().startswith("#")]
            if lines:
                save_secure_cloudflare_accounts("\n".join(lines))
                saved["outside_text_cloudflare_accounts"] = ""
                settings_manager.save_config(saved)
                return lines
    except Exception:
        pass

    return []


def clear_secure_cloudflare_accounts() -> bool:
    target = "MangaTranslator:CloudflareAccounts"
    _cred_delete(target)
    with _LOCK:
        _CACHE.pop("cloudflare_accounts", None)
    return True

