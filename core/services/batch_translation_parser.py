import re
from typing import Any, Dict, List, Optional, Tuple


def natural_sort_key(s: str) -> List[Any]:
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def parse_chat_response_to_batch_manual(
    chat_text: str,
    existing_batch_manual_txt: str = "",
) -> str:
    """
    Parses translations from an AI chat response and updates or builds
    the batch manual translation text formatted as:
    
    001.png:
    1: [Original] || [Translated]
    2: [Original] || [Translated]

    Args:
        chat_text: Raw output from AI chat session.
        existing_batch_manual_txt: Existing content in the manual editor to merge into.

    Returns:
        Consolidated and updated batch manual translation text.
    """
    # First, parse existing manual translations into structured dict
    file_translations: Dict[str, Dict[int, str]] = {}
    
    def _parse_lines_into_dict(text: str):
        current_file = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            # Header like "001.png:" or "001.jpg :" or "### 001.png"
            m_header = re.match(
                r"^(?:#{1,4}\s*)?([a-zA-Z0-9_\-.]+\.(?:png|jpg|jpeg|webp))\s*:?$",
                line,
                re.IGNORECASE,
            )
            if m_header:
                current_file = m_header.group(1).strip()
                if current_file not in file_translations:
                    file_translations[current_file] = {}
                continue

            # Inline format "001.png: 1: orig || trans"
            m_inline = re.match(
                r"^([a-zA-Z0-9_\-.]+\.(?:png|jpg|jpeg|webp))\s*:\s*(\d+)\s*:\s*(.*?)$",
                line,
                re.IGNORECASE,
            )
            if m_inline:
                f_name = m_inline.group(1).strip()
                idx = int(m_inline.group(2))
                content = m_inline.group(3).strip()
                if f_name not in file_translations:
                    file_translations[f_name] = {}
                file_translations[f_name][idx] = content
                continue

            # Numbered line like "1: orig || trans" under current file
            m_num = re.match(r"^(\d+)\s*:\s*(.*?)$", line)
            if m_num and current_file:
                idx = int(m_num.group(1))
                content = m_num.group(2).strip()
                file_translations[current_file][idx] = content

    if existing_batch_manual_txt:
        _parse_lines_into_dict(existing_batch_manual_txt)

    # Now parse chat text and overwrite/add entries
    _parse_lines_into_dict(chat_text)

    # Reconstruct the consolidated string sorted naturally by filename
    result_blocks = []
    for f_name in sorted(file_translations.keys(), key=natural_sort_key):
        lines = [f"{f_name}:"]
        items = file_translations[f_name]
        for idx in sorted(items.keys()):
            lines.append(f"{idx}: {items[idx]}")
        result_blocks.append("\n".join(lines))

    return "\n\n".join(result_blocks)


# Regex for symbols that should be left blank according to rules
PURE_SYMBOL_REGEX = re.compile(r"^[\s.\-_~!?♥♡★☆…—·\(\)\[\]/\\\"'“”^]+$")
# Regex for untranslated Asian characters (Chinese, Japanese Kana, Korean Hangul)
ASIAN_RAW_CHAR_REGEX = re.compile(
    r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u1100-\u11ff\u3130-\u318f]"
)
# Japanese Kana sound effects / onomatopoeia / moans / sighs (Hiragana + Katakana + symbols only)
# Contains NO Kanji (\u4e00-\u9fff), NO Chinese characters, NO Korean Hangul, NO Latin letters
JAPANESE_KANA_SFX_REGEX = re.compile(
    r"^[\s\u3040-\u309f\u30a0-\u30ff\u31f0-\u31ff\u30fc\u30fb.\-_~!?♥♡★☆…—·\(\)\[\]/\\\"'“”^]+$"
)


def is_ignorable_original_line(orig_text: str) -> bool:
    """
    Returns True if the original line is:
    1. Empty, pure symbols, or OCR failed.
    2. Pure Japanese Kana SFX/moans/interjections (e.g. ビクッ, ん, あっ♥, はぁ, ふぅ).
       Per rules: "TIẾNG NHẬT: KHÔNG DỊCH từ tượng thanh / cảm thán / rên rỉ -> để trống sau ||".
    """
    cleaned = (orig_text or "").strip()
    if not cleaned:
        return True
    if cleaned.upper() in ("[OCR FAILED]", "OCR FAILED", "[OCR_FAILED]"):
        return True
    if PURE_SYMBOL_REGEX.match(cleaned):
        return True
    # If the original line is pure Japanese Kana + symbols (no Kanji, no Korean, no Latin)
    if JAPANESE_KANA_SFX_REGEX.match(cleaned):
        return True
    return False


def validate_batch_translations_stats(
    batch_manual_txt: str,
) -> Tuple[int, int, Dict[str, List[int]], Dict[str, List[int]]]:
    """
    Checks translation completeness and Asian language bleed across all files.

    Returns:
        (total_lines, translated_lines, missing_lines_by_file, asian_bleed_by_file)
    """
    total_lines = 0
    translated_lines = 0
    missing_by_file: Dict[str, List[int]] = {}
    asian_bleed_by_file: Dict[str, List[int]] = {}

    current_file = None
    for raw_line in (batch_manual_txt or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m_header = re.match(
            r"^(?:#{1,4}\s*)?([a-zA-Z0-9_\-.]+\.(?:png|jpg|jpeg|webp))\s*:?$",
            line,
            re.IGNORECASE,
        )
        if m_header:
            current_file = m_header.group(1).strip()
            continue

        m_num = re.match(r"^(\d+)\s*:\s*(.*?)$", line)
        if m_num and current_file:
            idx = int(m_num.group(1))
            content = m_num.group(2).strip()
            total_lines += 1

            if "||" in content:
                parts = content.split("||", 1)
                orig_part = parts[0].strip()
                trans_part = parts[1].strip() if len(parts) > 1 else ""

                if trans_part:
                    # Check if translation has untranslated Asian characters (Hán, Nhật, Hàn)
                    if ASIAN_RAW_CHAR_REGEX.search(trans_part):
                        asian_bleed_by_file.setdefault(current_file, []).append(idx)
                    translated_lines += 1
                else:
                    # If trans_part is empty, check if original was pure symbols / ocr failed
                    if is_ignorable_original_line(orig_part):
                        translated_lines += 1  # Intentionally blank per rules
                    else:
                        missing_by_file.setdefault(current_file, []).append(idx)
            else:
                if is_ignorable_original_line(content):
                    translated_lines += 1
                else:
                    missing_by_file.setdefault(current_file, []).append(idx)

    return total_lines, translated_lines, missing_by_file, asian_bleed_by_file


def find_problematic_lines(batch_manual_txt: str) -> List[Dict[str, Any]]:
    """
    Finds lines that truly need fixing:
    1. Lines with genuine text that are missing translation.
    2. Lines where translation contains untranslated Asian characters (Chinese, Japanese, Korean).
    """
    problematic = []
    current_file = None

    for raw_line in (batch_manual_txt or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m_header = re.match(
            r"^(?:#{1,4}\s*)?([a-zA-Z0-9_\-.]+\.(?:png|jpg|jpeg|webp))\s*:?$",
            line,
            re.IGNORECASE,
        )
        if m_header:
            current_file = m_header.group(1).strip()
            continue

        m_num = re.match(r"^(\d+)\s*:\s*(.*?)$", line)
        if m_num and current_file:
            idx = int(m_num.group(1))
            content = m_num.group(2).strip()

            if "||" in content:
                parts = content.split("||", 1)
                orig_part = parts[0].strip()
                trans_part = parts[1].strip() if len(parts) > 1 else ""

                if trans_part:
                    # Asian character leak check
                    if ASIAN_RAW_CHAR_REGEX.search(trans_part):
                        problematic.append({
                            "file": current_file,
                            "idx": idx,
                            "orig": orig_part,
                            "trans": trans_part,
                            "reason": "asian_bleed",
                        })
                else:
                    if not is_ignorable_original_line(orig_part):
                        problematic.append({
                            "file": current_file,
                            "idx": idx,
                            "orig": orig_part,
                            "trans": "",
                            "reason": "missing",
                        })
            else:
                if not is_ignorable_original_line(content):
                    problematic.append({
                        "file": current_file,
                        "idx": idx,
                        "orig": content,
                        "trans": "",
                        "reason": "missing",
                    })

    return problematic


def parse_and_validate_batch_manual(
    chat_text: str,
    existing_batch_manual_txt: str = "",
) -> Tuple[str, str]:
    """
    Parses chat translations and validates completeness against existing manual txt.
    Returns (updated_batch_manual_txt, status_summary_markdown).
    """
    updated_txt = parse_chat_response_to_batch_manual(chat_text, existing_batch_manual_txt)
    total_lines, trans_lines, missing_by_file, asian_bleed_by_file = validate_batch_translations_stats(updated_txt)

    if total_lines == 0:
        return updated_txt, "No dialogue lines found in Batch Manual Box."

    all_good = not missing_by_file and not asian_bleed_by_file
    if all_good:
        status_md = f"**Sync complete (100%)!** Translated **{trans_lines}/{total_lines}** lines cleanly."
    else:
        percent = round((trans_lines / total_lines) * 100, 1)
        issues = []
        if missing_by_file:
            lines_info = []
            for fname, missing_idxs in missing_by_file.items():
                lines_info.append(f"• `{fname}`: missing lines {', '.join(map(str, missing_idxs))}")
            issues.append(f"**Missing translations:**\n" + "\n".join(lines_info[:6]))

        if asian_bleed_by_file:
            bleed_info = []
            for fname, bleed_idxs in asian_bleed_by_file.items():
                bleed_info.append(f"• `{fname}`: untranslated characters at line {', '.join(map(str, bleed_idxs))}")
            issues.append(f"**Untranslated character bleed:**\n" + "\n".join(bleed_info[:6]))

        details = "\n\n".join(issues)
        status_md = (
            f"**Synced {trans_lines}/{total_lines} lines ({percent}%).**\n{details}"
        )

    return updated_txt, status_md


def extract_file_names_from_batch_manual(batch_manual_txt: str) -> List[str]:
    """Extract list of file names sorted in natural numerical order from batch manual text."""
    files: List[str] = []
    seen = set()
    for raw_line in (batch_manual_txt or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m_header = re.match(
            r"^(?:#{1,4}\s*)?([a-zA-Z0-9_\-.]+\.(?:png|jpg|jpeg|webp))\s*:?$",
            line,
            re.IGNORECASE,
        )
        if m_header:
            f_name = m_header.group(1).strip()
            if f_name not in seen:
                seen.add(f_name)
                files.append(f_name)
    return sorted(files, key=natural_sort_key)


def extract_ocr_blocks_for_files(batch_manual_txt: str, target_files: List[str]) -> str:
    """Extracts raw OCR text lines for specific target files from batch manual text."""
    target_set = set(target_files)
    current_file = None
    collecting = False
    blocks = []
    current_block = []

    for raw_line in (batch_manual_txt or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        m_header = re.match(
            r"^(?:#{1,4}\s*)?([a-zA-Z0-9_\-.]+\.(?:png|jpg|jpeg|webp))\s*:?$",
            line,
            re.IGNORECASE,
        )
        if m_header:
            f_name = m_header.group(1).strip()
            if current_block:
                blocks.append("\n".join(current_block))
                current_block = []
            if f_name in target_set:
                collecting = True
                current_file = f_name
                current_block.append(f"{f_name}:")
            else:
                collecting = False
                current_file = None
            continue

        if collecting and current_file:
            current_block.append(raw_line)

    if current_block:
        blocks.append("\n".join(current_block))

    return "\n\n".join(blocks)

