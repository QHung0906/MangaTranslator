import re
from typing import Any, Dict, List, Optional, Tuple


def natural_sort_key(s: str) -> List[Any]:
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]

def _clean_entry_content(
    content: str,
    tag: Optional[str] = None,
    expected_idx: Optional[int] = None,
) -> Tuple[Optional[str], str]:
    """
    Cleans entry content by stripping duplicate leading numbers ('1: ') and redundant
    tags ([Bubble], [OSB]), returning (tag, clean_content).
    """
    text = content.strip()
    current_tag = tag
    while True:
        m_tag = re.match(r"^\[(Bubble|OSB)\]\s*", text, re.IGNORECASE)
        if m_tag:
            if not current_tag:
                current_tag = "OSB" if m_tag.group(1).upper() == "OSB" else "Bubble"
            text = text[m_tag.end():].strip()
            continue
        m_num = re.match(r"^(\d+)\s*:\s*", text)
        if m_num:
            num_val = int(m_num.group(1))
            rest = text[m_num.end():].strip()
            if (
                (expected_idx is not None and num_val == expected_idx)
                or re.match(r"^\[(Bubble|OSB)\]", rest, re.IGNORECASE)
                or (current_tag is not None and (expected_idx is None or num_val == expected_idx))
            ):
                text = rest
                continue
        break
    if current_tag:
        current_tag = "OSB" if current_tag.upper() == "OSB" else "Bubble"
    return current_tag, text


def parse_line_index_and_content(line: str) -> Tuple[Optional[str], Optional[int], str]:
    """
    Parses a dialogue line into (tag, index, content).
    Handles all formats reliably:
    - "1: [Bubble] 1: content" or "1: [Bubble] content"
    - "[Bubble] 1: content" or "[OSB] 2: 22:32 ||"
    - "1: content"
    Does NOT swallow numbers that belong to content (e.g. '22:32').
    """
    text = line.strip()
    
    # 1. Outer index followed by tag: "1: [Bubble] 1: content" or "1: [Bubble] content"
    m = re.match(
        r"^(?P<outer_idx>\d+)\s*:\s*\[(?P<tag>Bubble|OSB)\]\s*(?:(?P<inner_idx>\d+)\s*:\s*)?(?P<content>.*)$",
        text,
        re.IGNORECASE,
    )
    if m:
        tag = "OSB" if m.group("tag").upper() == "OSB" else "Bubble"
        idx = int(m.group("inner_idx") or m.group("outer_idx"))
        _, clean = _clean_entry_content(m.group("content"), tag, expected_idx=idx)
        return tag, idx, clean

    # 2. Tag followed by index: "[Bubble] 1: content" or "[OSB] 2: 22:32 ||"
    m = re.match(
        r"^\[(?P<tag>Bubble|OSB)\]\s*(?P<idx>\d+)\s*:\s*(?P<content>.*)$",
        text,
        re.IGNORECASE,
    )
    if m:
        tag = "OSB" if m.group("tag").upper() == "OSB" else "Bubble"
        idx = int(m.group("idx"))
        _, clean = _clean_entry_content(m.group("content"), tag, expected_idx=idx)
        return tag, idx, clean

    # 3. Simple index: "1: content"
    m = re.match(r"^(?P<idx>\d+)\s*:\s*(?P<content>.*)$", text)
    if m:
        idx = int(m.group("idx"))
        tag, clean = _clean_entry_content(m.group("content"), None, expected_idx=idx)
        return tag, idx, clean

    return None, None, text


def parse_chat_response_to_batch_manual(
    chat_text: str,
    existing_batch_manual_txt: str = "",
) -> str:
    """
    Parses translations from an AI chat response and updates or builds
    the batch manual translation text formatted as:
    
    001.png:
    [Bubble] 1: [Original] || [Translated]
    [OSB] 2: [Original] || [Translated]

    Args:
        chat_text: Raw output from AI chat session.
        existing_batch_manual_txt: Existing content in the manual editor to merge into.

    Returns:
        Consolidated and updated batch manual translation text.
    """
    # First, parse existing manual translations into structured dict
    file_translations: Dict[str, Dict[int, Tuple[Optional[str], str]]] = {}
    
    def _parse_lines_into_dict(text: str):
        current_file = None
        for raw_line in text.splitlines():
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
                if current_file not in file_translations:
                    file_translations[current_file] = {}
                continue

            tag, idx, content = parse_line_index_and_content(line)
            if idx is not None and current_file:
                file_translations[current_file][idx] = (tag, content)

    # First, record existing tags from existing manual txt
    existing_tags: Dict[str, Dict[int, str]] = {}
    if existing_batch_manual_txt:
        _parse_lines_into_dict(existing_batch_manual_txt)
        for f_name, entries in file_translations.items():
            existing_tags[f_name] = {}
            for idx, (t, _) in entries.items():
                if t:
                    existing_tags[f_name][idx] = t

    # Parse chat text and merge
    def _parse_chat_lines_into_dict(text: str):
        current_file = None
        for raw_line in text.splitlines():
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
                if current_file not in file_translations:
                    file_translations[current_file] = {}
                continue

            tag, idx, content = parse_line_index_and_content(line)
            if idx is not None and current_file:
                if not tag and current_file in existing_tags and idx in existing_tags[current_file]:
                    tag = existing_tags[current_file][idx]
                file_translations[current_file][idx] = (tag, content)

    _parse_chat_lines_into_dict(chat_text)

    # Reconstruct the consolidated string sorted naturally by filename
    result_blocks = []
    for f_name in sorted(file_translations.keys(), key=natural_sort_key):
        lines = [f"{f_name}:"]
        items = file_translations[f_name]
        for idx in sorted(items.keys()):
            tag, content = items[idx]
            if not tag and f_name in existing_tags and idx in existing_tags[f_name]:
                tag = existing_tags[f_name][idx]
            if tag:
                lines.append(f"[{tag}] {idx}: {content}")
            else:
                lines.append(f"{idx}: {content}")
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


def is_ignorable_original_line(orig_text: str, tag: Optional[str] = None) -> bool:
    """
    Returns True if the original line is:
    1. Empty, pure symbols, or OCR failed.
    2. Pure Japanese Kana SFX/moans/interjections (e.g. ビクッ, ん, あっ♥, はぁ, ふぅ) ONLY when outside bubbles ([OSB]).
       - If tag is 'Bubble': Japanese Kana SFX must be translated to expressive Vietnamese -> NOT ignorable.
       - If tag is 'OSB' or None: Keep blank per rules -> ignorable.
    """
    cleaned = (orig_text or "").strip()
    if tag is None:
        m_tag = re.match(r"^\[(Bubble|OSB)\]", cleaned, re.IGNORECASE)
        if m_tag:
            tag = m_tag.group(1)

    # Strip any leading tag like [Bubble] or [OSB]
    cleaned = re.sub(r"^\[(?:Bubble|OSB)\]\s*", "", cleaned, flags=re.IGNORECASE).strip()
    # Strip any duplicate leading index if present, e.g. "1: "
    cleaned = re.sub(r"^\d+\s*:\s*", "", cleaned).strip()
    if not cleaned:
        return True
    if cleaned.upper() in ("[OCR FAILED]", "OCR FAILED", "[OCR_FAILED]"):
        return True
    if PURE_SYMBOL_REGEX.match(cleaned):
        return True

    # Pure digits, timestamps or numeric codes (e.g. '22:32', '12:00', '100%')
    if re.match(r"^[\d\s:.\-_/%]+$", cleaned):
        return True

    # If the original line is pure Japanese Kana + symbols (no Kanji, no Korean, no Latin)
    if JAPANESE_KANA_SFX_REGEX.match(cleaned):
        # [Bubble] SFX must be translated; [OSB] SFX is ignorable
        if tag and tag.lower() == "bubble":
            return False
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

        tag, idx, content = parse_line_index_and_content(line)
        if idx is not None and current_file:
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
                    if is_ignorable_original_line(orig_part, tag=tag):
                        translated_lines += 1  # Intentionally blank per rules
                    else:
                        missing_by_file.setdefault(current_file, []).append(idx)
            else:
                if is_ignorable_original_line(content, tag=tag):
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

        tag, idx, content = parse_line_index_and_content(line)
        if idx is not None and current_file:
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
                            "tag": tag,
                            "orig": orig_part,
                            "trans": trans_part,
                            "reason": "asian_bleed",
                        })
                else:
                    if not is_ignorable_original_line(orig_part, tag=tag):
                        problematic.append({
                            "file": current_file,
                            "idx": idx,
                            "tag": tag,
                            "orig": orig_part,
                            "trans": "",
                            "reason": "missing",
                        })
            else:
                if not is_ignorable_original_line(content, tag=tag):
                    problematic.append({
                        "file": current_file,
                        "idx": idx,
                        "tag": tag,
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


def extract_ocr_blocks_for_files(
    batch_manual_txt: str,
    target_files: List[str],
    strip_tags: bool = False,
) -> str:
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
            if strip_tags:
                tag, idx, clean_content = parse_line_index_and_content(line)
                if idx is not None:
                    if "||" in clean_content:
                        orig_part = clean_content.split("||", 1)[0].strip()
                    else:
                        orig_part = clean_content.strip()
                    current_block.append(f"{idx}: {orig_part}")
                else:
                    current_block.append(raw_line)
            else:
                current_block.append(raw_line)

    if current_block:
        blocks.append("\n".join(current_block))

    return "\n\n".join(blocks)

