import base64
from concurrent.futures import ThreadPoolExecutor
import gc
import math
import os
import random
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from core.batch_coordinator import (
    expanded_mask_bbox,
    partition_non_overlapping_waves,
    paste_image_region,
)
from core.config import MangaTranslatorConfig
from core.image.image_utils import cv2_to_pil, pil_to_cv2, process_bubble_image_cached
from core.image.inpainting import FluxKleinInpainter, FluxKontextInpainter
from core.image.ocr_detection import OutsideTextDetector, extract_text_with_manga_ocr
from core.image.tilt_detection import (
    compute_effective_tilt_angle,
    detect_crop_tilt_angle,
)
from core.ml.model_manager import get_model_manager
from utils.logging import log_message

# OSB Expansion and Classification Constants
OSB_EXPANSION_PIXEL_BUFFER = 5  # for bubbles, nearby OSB regions, panels
OSB_SOLID_RATIO_THRESHOLD = (
    0.95  # Stricter than interior sampling because border ring has no text mask noise
)
OSB_COLOR_TOLERANCE = 15
OSB_EXPANSION_PX = 2
OSB_WHITE_SNAP_THRESH = 245
OSB_BLACK_SNAP_THRESH = 10


@dataclass
class OutsideTextWork:
    """Prepared outside-text state: translation data ready, inpaint still pending."""

    pil_image: Image.Image
    config: MangaTranslatorConfig
    image_path: Union[str, Path]
    image_format: Optional[str]
    verbose: bool
    outside_text_results: list
    raw_outside_text_results: list
    original_text_colors: dict
    total_bubble_mask: Any
    outside_detector: Any
    mask_groups: list
    img_w: int
    img_h: int
    mime_type: str
    cv2_ext: str
    outside_text_data: List[Dict[str, Any]] = field(default_factory=list)


def _build_outside_text_data(
    pil_image: Image.Image,
    outside_text_results,
    raw_outside_text_results,
    original_text_colors,
    extracted_text_colors,
    none_skipped_clip_bboxes,
    mime_type: str,
    cv2_ext: str,
    config: MangaTranslatorConfig,
    verbose: bool,
) -> List[Dict[str, Any]]:
    outside_text_data = []
    original_cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    for expanded_res, raw_res in zip(outside_text_results, raw_outside_text_results):
        bbox_coords, conf = expanded_res
        raw_coords, _ = raw_res

        x1, y1, x2, y2 = [int(c) for c in bbox_coords]
        bbox_tuple = (x1, y1, x2, y2)
        raw_bbox_tuple = tuple(int(c) for c in raw_coords)

        outside_text_image_cv = original_cv_image[y1:y2, x1:x2].copy()
        outside_text_image_pil = cv2_to_pil(outside_text_image_cv)
        original_crop_pil = outside_text_image_pil.copy()

        osb_upscale_method = (
            "none" if config.test_mode else config.translation.upscale_method
        )

        if osb_upscale_method == "model":
            model_manager = get_model_manager()
            upscale_model = model_manager.load_upscale(verbose=verbose)
            final_text_pil = process_bubble_image_cached(
                outside_text_image_pil,
                upscale_model,
                config.device,
                config.translation.osb_min_side_pixels,
                "min",
                "model",
                verbose,
            )
            model_manager.clear_cache()
        elif osb_upscale_method == "model_lite":
            model_manager = get_model_manager()
            upscale_model = model_manager.load_upscale_lite(verbose=verbose)
            final_text_pil = process_bubble_image_cached(
                outside_text_image_pil,
                upscale_model,
                config.device,
                config.translation.osb_min_side_pixels,
                "min",
                "model_lite",
                verbose,
            )
            model_manager.clear_cache()
        elif osb_upscale_method == "lanczos":
            w, h = outside_text_image_pil.size
            min_side = min(w, h)
            if min_side < config.translation.osb_min_side_pixels:
                scale_factor = config.translation.osb_min_side_pixels / min_side
                new_w = int(w * scale_factor)
                new_h = int(h * scale_factor)
                resized_text = outside_text_image_pil.resize(
                    (new_w, new_h), Image.LANCZOS
                )
            else:
                resized_text = outside_text_image_pil
            final_text_pil = resized_text
        else:
            final_text_pil = outside_text_image_pil

        outside_text_image_cv = pil_to_cv2(final_text_pil)

        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        aspect_ratio = float(h) / float(w)

        needs_text_bg = False
        if none_skipped_clip_bboxes:
            bcx = (x1 + x2) / 2
            bcy = (y1 + y2) / 2
            for clip_bbox in none_skipped_clip_bboxes:
                cx1, cy1, cx2, cy2 = clip_bbox
                if cx1 <= bcx <= cx2 and cy1 <= bcy <= cy2:
                    needs_text_bg = True
                    break

        angle_deg, tilt_conf, obb_dims, tilt_axis = (
            0.0,
            0.0,
            (float(w), float(h)),
            "none",
        )
        osb_tilt_estimation_side_pixels = 256
        if getattr(config.outside_text, "osb_follow_tilt", False) and original_cv_image is not None:
            rx1, ry1, rx2, ry2 = raw_bbox_tuple
            raw_crop_cv = original_cv_image[ry1:ry2, rx1:rx2]
            if raw_crop_cv.size > 0:
                angle_deg, tilt_conf, obb_dims, tilt_axis = detect_crop_tilt_angle(
                    crop_cv=raw_crop_cv,
                    is_dark_text=original_text_colors.get(raw_bbox_tuple, True),
                    max_side=osb_tilt_estimation_side_pixels,
                    deadband_deg=getattr(config.outside_text, "osb_min_tilt_deg", 3.0),
                    max_tilt_deg=45.0,
                )

        try:
            is_success, buffer = cv2.imencode(cv2_ext, outside_text_image_cv)
            if is_success:
                image_b64 = base64.b64encode(buffer).decode("utf-8")
                outside_text_data.append(
                    {
                        "bbox": bbox_tuple,
                        "original_bbox": raw_bbox_tuple,
                        "confidence": conf,
                        "is_outside_text": True,
                        "image_b64": image_b64,
                        "mime_type": mime_type,
                        "is_dark_text": original_text_colors.get(raw_bbox_tuple, True),
                        "text_color_rgb": extracted_text_colors.get(raw_bbox_tuple),
                        "aspect_ratio": aspect_ratio,
                        "needs_text_background": needs_text_bg,
                        "original_crop_pil": original_crop_pil,
                        "tilt_deg": angle_deg,
                        "tilt_conf": tilt_conf,
                        "tilt_axis": tilt_axis,
                        "oriented_dimensions": obb_dims,
                    }
                )
        except Exception as e:
            log_message(
                f"Error encoding outside text bbox {(x1, y1, x2, y2)}: {e}",
                verbose=verbose,
            )

    return outside_text_data


def _apply_inpaint_render_metadata(
    outside_text_data: List[Dict[str, Any]],
    extracted_text_colors: dict,
    none_skipped_clip_bboxes: set,
) -> None:
    """Update existing OSB entries in-place with render-only fields from inpainting."""
    for item in outside_text_data:
        raw_bbox_tuple = item.get("original_bbox")
        bbox = item.get("bbox")
        if raw_bbox_tuple is not None and raw_bbox_tuple in extracted_text_colors:
            item["text_color_rgb"] = extracted_text_colors[raw_bbox_tuple]
        elif item.get("text_color_rgb") is None and extracted_text_colors and bbox:
            x1, y1, x2, y2 = bbox
            bcx, bcy = (x1 + x2) / 2, (y1 + y2) / 2
            for key, color in extracted_text_colors.items():
                if key == raw_bbox_tuple or key == bbox:
                    item["text_color_rgb"] = color
                    break
                if (
                    isinstance(key, tuple)
                    and len(key) == 4
                    and key[0] <= bcx <= key[2]
                    and key[1] <= bcy <= key[3]
                ):
                    item["text_color_rgb"] = color
                    break

        needs_text_bg = False
        if none_skipped_clip_bboxes and bbox is not None:
            x1, y1, x2, y2 = bbox
            bcx, bcy = (x1 + x2) / 2, (y1 + y2) / 2
            for clip_bbox in none_skipped_clip_bboxes:
                cx1, cy1, cx2, cy2 = clip_bbox
                if cx1 <= bcx <= cx2 and cy1 <= bcy <= cy2:
                    needs_text_bg = True
                    break
        item["needs_text_background"] = needs_text_bg


def prepare_outside_text_work(
    pil_image: Image.Image,
    config: MangaTranslatorConfig,
    image_path: Union[str, Path],
    image_format: Optional[str],
    verbose: bool = False,
    bubble_data: Optional[List[Dict[str, Any]]] = None,
    text_free_boxes: Optional[List[List[float]]] = None,
    panels: Optional[List[Tuple[int, int, int, int]]] = None,
) -> Optional[OutsideTextWork]:
    """Detect outside text and build translation crops without inpainting."""
    if not config.outside_text.enabled:
        return None

    log_message("Detecting text outside speech bubbles...", verbose=verbose)

    try:
        outside_detector = OutsideTextDetector(
            device=config.device, hf_token=config.outside_text.huggingface_token
        )
        outside_text_results = outside_detector.detect_outside_text(
            str(image_path),
            yolo_model_path=config.yolo_model_path,
            confidence=config.outside_text.osb_confidence,
            conjoined_confidence=config.detection.conjoined_confidence,
            verbose=verbose,
            image_override=pil_image,
            existing_bubbles=bubble_data,
            text_free_boxes=text_free_boxes,
            bubble_detector_model=config.detection.bubble_detector_model,
            min_area_ignore_ratio=config.outside_text.min_area_ignore_ratio,
            text_free_only=config.outside_text.osb_text_free_only,
        )

        if not outside_text_results:
            log_message("No outside text regions found", verbose=verbose)
            return None

        img_w, img_h = pil_image.size

        min_ignore = max(0.0, min(0.05, config.outside_text.min_area_ignore_ratio))
        if min_ignore > 0.0:
            image_area = float(img_w * img_h)
            kept_results = []
            removed_count = 0
            for bbox, conf in outside_text_results:
                x1, y1, x2, y2 = bbox
                area_ratio = ((x2 - x1) * (y2 - y1)) / max(1.0, image_area)
                if area_ratio < min_ignore:
                    removed_count += 1
                    log_message(
                        f"Skipping small OSB region (area {area_ratio:.4%} < {min_ignore:.4%})",
                        verbose=verbose,
                    )
                    continue
                kept_results.append((bbox, conf))
            outside_text_results = kept_results
            if removed_count:
                log_message(
                    f"Filtered {removed_count} OSB region(s) below min area threshold "
                    f"({min_ignore:.2%} of image); {len(outside_text_results)} remaining",
                    verbose=verbose,
                )
            if not outside_text_results:
                log_message(
                    "No outside text regions remaining after min area filter",
                    verbose=verbose,
                )
                return None

        # Filter out probable page numbers
        # Only run OCR on "suspicious" detections (small & in margin)
        if config.outside_text.enable_page_number_filtering and outside_text_results:
            suspicious_crops = []
            suspicious_indices = []
            safe_results = []

            margin_threshold = max(
                0.0, min(0.3, config.outside_text.page_filter_margin_threshold)
            )
            min_area_threshold = max(
                0.0, min(0.2, config.outside_text.page_filter_min_area_ratio)
            )

            for i, res in enumerate(outside_text_results):
                bbox, _ = res
                x1, y1, x2, y2 = [int(c) for c in bbox]
                cy = (y1 + y2) / 2

                is_in_margin = (cy < img_h * margin_threshold) or (
                    cy > img_h * (1 - margin_threshold)
                )

                area = (x2 - x1) * (y2 - y1)
                is_small = area < (img_w * img_h * min_area_threshold)

                if is_in_margin and is_small:
                    suspicious_crops.append(pil_image.crop((x1, y1, x2, y2)))
                    suspicious_indices.append(i)
                else:
                    safe_results.append(res)

            if suspicious_crops:
                log_message(
                    f"Verifying {len(suspicious_crops)} suspicious OSB regions with OCR...",
                    verbose=verbose,
                )
                suspicious_texts = extract_text_with_manga_ocr(
                    suspicious_crops, verbose=verbose
                )

                kept_suspicious_count = 0
                for i, text in enumerate(suspicious_texts):
                    # Regex for page numbers: digits, "Page 20", "p. 20", etc.
                    is_page_number = bool(
                        re.match(
                            r"^\s*(?:page\.?|p\.?)?\s*\d+\s*$", text, re.IGNORECASE
                        )
                    )

                    if not is_page_number:
                        safe_results.append(outside_text_results[suspicious_indices[i]])
                        kept_suspicious_count += 1
                    else:
                        log_message(
                            f"Filtered out page number: '{text}'", verbose=verbose
                        )

                outside_text_results = safe_results
                log_message(
                    f"Remaining OSB regions after filtering: {len(outside_text_results)}",
                    verbose=verbose,
                )

        raw_outside_text_results = outside_text_results.copy()

        # Apply OSB render expansion for shapes that tend to render too small.
        narrow_expansion_mult = getattr(
            config.outside_text, "osb_render_expansion_narrow_multiplier", 1.0
        )
        tiny_expansion_mult = getattr(
            config.outside_text, "osb_render_expansion_tiny_multiplier", 1.0
        )
        aspect_ratio_threshold = getattr(
            config.outside_text, "osb_render_expansion_aspect_ratio_threshold", 0.4
        )
        area_ratio_threshold = getattr(
            config.outside_text, "osb_render_expansion_area_ratio_threshold", 0.005
        )
        res_scale = max(0.5, min(img_w, img_h) / 1500.0)
        buffer = max(4, int(round(OSB_EXPANSION_PIXEL_BUFFER * res_scale)))

        if outside_text_results:
            log_message(
                "Expanding OSB bboxes dynamically (panel & obstacle-aware space allocation)...",
                verbose=verbose,
            )
            expanded_results = []
            for i, res in enumerate(outside_text_results):
                bbox, conf = res
                x1, y1, x2, y2 = bbox
                w = max(1, x2 - x1)
                h = max(1, y2 - y1)

                aspect_ratio = float(w) / float(h)
                is_horizontal = aspect_ratio >= 1.0

                cx = x1 + w / 2.0
                cy = y1 + h / 2.0

                # 1. Determine panel boundaries
                associated_panel = None
                if panels:
                    for p in panels:
                        px1, py1, px2, py2 = p
                        if px1 <= cx <= px2 and py1 <= cy <= py2:
                            associated_panel = p
                            break

                if associated_panel:
                    c_px1 = min(int(associated_panel[0]) + buffer, int(associated_panel[2]))
                    c_py1 = min(int(associated_panel[1]) + buffer, int(associated_panel[3]))
                    c_px2 = max(int(associated_panel[2]) - buffer, int(associated_panel[0]))
                    c_py2 = max(int(associated_panel[3]) - buffer, int(associated_panel[1]))
                else:
                    c_px1 = buffer
                    c_py1 = buffer
                    c_px2 = max(buffer, img_w - buffer)
                    c_py2 = max(buffer, img_h - buffer)

                # 2. Gather obstacles (speech bubbles and other outside text regions)
                obstacles = []
                if bubble_data:
                    for b in bubble_data:
                        bb = b.get("bbox")
                        if bb and len(bb) == 4:
                            bx1, by1, bx2, by2 = [int(c) for c in bb]
                            obstacles.append(
                                (
                                    max(0, bx1 - buffer),
                                    max(0, by1 - buffer),
                                    min(img_w, bx2 + buffer),
                                    min(img_h, by2 + buffer),
                                )
                            )

                for j, other_res in enumerate(outside_text_results):
                    if i == j:
                        continue
                    if j < i:
                        ob, _ = expanded_results[j]
                    else:
                        ob, _ = other_res
                    obx1, oby1, obx2, oby2 = [int(c) for c in ob]
                    obstacles.append(
                        (
                            max(0, obx1 - buffer),
                            max(0, oby1 - buffer),
                            min(img_w, obx2 + buffer),
                            min(img_h, oby2 + buffer),
                        )
                    )

                # 3. Compute available clear space around (cx, cy)
                avail_x1 = c_px1
                avail_x2 = c_px2
                avail_y1 = c_py1
                avail_y2 = c_py2

                for ox1, oy1, ox2, oy2 in obstacles:
                    if not (y2 <= oy1 or y1 >= oy2):
                        if ox2 <= cx:
                            avail_x1 = max(avail_x1, ox2)
                        elif ox1 >= cx:
                            avail_x2 = min(avail_x2, ox1)
                    if not (x2 <= ox1 or x1 >= ox2):
                        if oy2 <= cy:
                            avail_y1 = max(avail_y1, oy2)
                        elif oy1 >= cy:
                            avail_y2 = min(avail_y2, oy1)

                avail_x1 = min(avail_x1, int(x1))
                avail_x2 = max(avail_x2, int(x2))
                avail_y1 = min(avail_y1, int(y1))
                avail_y2 = max(avail_y2, int(y2))

                avail_w = max(w, avail_x2 - avail_x1)
                avail_h = max(h, avail_y2 - avail_y1)

                # 4. Apply expansion
                if is_horizontal:
                    # Horizontal text: expand width to accommodate Latin text
                    target_w = min(avail_w, max(w, max(int(round(w * 1.8)), w + int(round(60.0 * res_scale)))))
                    nx1 = max(avail_x1, min(avail_x2 - target_w, int(cx - target_w / 2.0)))
                    nx2 = min(avail_x2, nx1 + target_w)
                    nx1 = min(nx1, int(x1))
                    nx2 = max(nx2, int(x2))

                    safe_h = max(h, min(avail_h, int(round(h * 1.3))))
                    ny1 = max(avail_y1, min(avail_y2 - safe_h, int(cy - safe_h / 2.0)))
                    ny2 = min(avail_y2, ny1 + safe_h)
                    ny1 = min(ny1, int(y1))
                    ny2 = max(ny2, int(y2))
                else:
                    # Vertical text: Japanese vertical columns require horizontal width for Latin words.
                    # Derive Latin word width dynamically from character height and resolution.
                    char_h_est = max(12.0, min(float(w), float(h) / 3.0))
                    desired_latin_w = max(float(w), min(130.0 * res_scale, max(float(w) * 2.2, char_h_est * 3.5)))
                    target_w = min(avail_w, int(round(desired_latin_w)))

                    nx1 = max(avail_x1, min(avail_x2 - target_w, int(cx - target_w / 2.0)))
                    nx2 = min(avail_x2, nx1 + target_w)
                    nx1 = min(nx1, int(x1))
                    nx2 = max(nx2, int(x2))

                    safe_h = max(h, min(avail_h, int(round(h * 1.15))))
                    ny1 = max(avail_y1, min(avail_y2 - safe_h, int(cy - safe_h / 2.0)))
                    ny2 = min(avail_y2, ny1 + safe_h)
                    ny1 = min(ny1, int(y1))
                    ny2 = max(ny2, int(y2))

                expanded_results.append(([int(nx1), int(ny1), int(nx2), int(ny2)], conf))

            outside_text_results = expanded_results

        # Build a mask of all detected speech bubbles to prevent OSB inpainting overlap
        total_bubble_mask = np.zeros((img_h, img_w), dtype=bool)
        if bubble_data:
            for bubble in bubble_data:
                try:
                    mask = bubble.get("sam_mask") if isinstance(bubble, dict) else None
                    if mask is not None:
                        mask_np = np.asarray(mask)
                        if mask_np.ndim == 3:
                            mask_np = mask_np[..., 0]
                        mask_bool = mask_np > 0
                        if mask_bool.shape[0] == img_h and mask_bool.shape[1] == img_w:
                            total_bubble_mask |= mask_bool
                            continue

                    bbox = bubble.get("bbox") if isinstance(bubble, dict) else None
                    if bbox and len(bbox) == 4:
                        x0, y0, x1, y1 = [int(c) for c in bbox]
                        x0 = max(0, min(img_w, x0))
                        x1 = max(0, min(img_w, x1))
                        y0 = max(0, min(img_h, y0))
                        y1 = max(0, min(img_h, y1))
                        if x1 > x0 and y1 > y0:
                            total_bubble_mask[y0:y1, x0:x1] = True
                except Exception as e:
                    log_message(
                        f"Warning: Failed to apply bubble mask for OSB exclusion: {e}",
                        verbose=verbose,
                    )

            if np.any(total_bubble_mask):
                # Dilate the bubble mask to provide a safe buffer for OSB fill
                kernel = np.ones((11, 11), np.uint8)
                total_bubble_mask = cv2.dilate(
                    total_bubble_mask.astype(np.uint8), kernel, iterations=1
                ).astype(bool)

        mime_type = (
            "image/png"
            if image_format and image_format.upper() == "PNG"
            else "image/jpeg"
        )
        cv2_ext = ".png" if image_format and image_format.upper() == "PNG" else ".jpg"

        # Probe original text color for OSB rendering
        original_text_colors = {}
        for ocr_result in raw_outside_text_results:
            bbox_coords, conf = ocr_result
            x1, y1, x2, y2 = [int(c) for c in bbox_coords]
            bbox_tuple = (x1, y1, x2, y2)

            bbox_area_img = pil_image.crop((x1, y1, x2, y2)).convert("RGB")
            bbox_array = np.array(bbox_area_img)
            pixels = bbox_array.reshape(-1, 3)

            # Use K-Means to find 2 dominant colors
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            kmeans.fit(pixels)

            labels = kmeans.labels_
            centers = kmeans.cluster_centers_

            unique, counts = np.unique(labels, return_counts=True)
            dominant_cluster_idx = unique[np.argmax(counts)]

            # Dominant cluster is usually the background (text pixels are sparse)
            bg_color_rgb = centers[dominant_cluster_idx]
            # Use proper luminance calculation (ITU-R BT.601)
            bg_brightness = (
                0.299 * bg_color_rgb[0]
                + 0.587 * bg_color_rgb[1]
                + 0.114 * bg_color_rgb[2]
            )
            is_dark_text = (
                bg_brightness < 128
            )  # passed downstream; renderer inverts for text color
            original_text_colors[bbox_tuple] = is_dark_text

            log_message(
                f"OSB bbox {bbox_tuple}: "
                f"{'Dark' if is_dark_text else 'Light'} background detected "
                f"(luminance={bg_brightness:.1f})",
                verbose=verbose,
            )

        mask_groups, _ = outside_detector.get_text_masks(
            str(image_path),
            bbox_expansion_percent_width=config.outside_text.bbox_expansion_percent_width,
            bbox_expansion_percent_height=config.outside_text.bbox_expansion_percent_height,
            text_box_proximity_ratio=config.outside_text.text_box_proximity_ratio,
            verbose=verbose,
            image_override=pil_image,
            existing_results=raw_outside_text_results,
        )

        outside_text_data = _build_outside_text_data(
            pil_image=pil_image,
            outside_text_results=outside_text_results,
            raw_outside_text_results=raw_outside_text_results,
            original_text_colors=original_text_colors,
            extracted_text_colors={},
            none_skipped_clip_bboxes=set(),
            mime_type=mime_type,
            cv2_ext=cv2_ext,
            config=config,
            verbose=verbose,
        )

        return OutsideTextWork(
            pil_image=pil_image,
            config=config,
            image_path=image_path,
            image_format=image_format,
            verbose=verbose,
            outside_text_results=outside_text_results,
            raw_outside_text_results=raw_outside_text_results,
            original_text_colors=original_text_colors,
            total_bubble_mask=total_bubble_mask,
            outside_detector=outside_detector,
            mask_groups=mask_groups,
            img_w=img_w,
            img_h=img_h,
            mime_type=mime_type,
            cv2_ext=cv2_ext,
            outside_text_data=outside_text_data,
        )

    except Exception as e:
        log_message(
            f"Error during outside text detection: {e}",
            always_print=True,
        )
        return None


def cluster_flux_candidates(
    candidates: List[Dict[str, Any]],
    max_dim: int = 1536,
    max_dist: int = 500,
    max_area: int = 1_800_000,
) -> List[List[Dict[str, Any]]]:
    """Group spatially proximate OSB candidates into safe-sized clusters for Flux inpainting.

    Prevents whole-page mega-crops that crush font resolution when scaled down,
    while minimizing the total number of Flux inference passes.
    """
    if not candidates:
        return []
    if len(candidates) == 1:
        return [candidates]

    def get_cand_bbox(c: Dict[str, Any]) -> Tuple[int, int, int, int]:
        if "bbox" in c and c["bbox"] is not None:
            return c["bbox"]
        mask = c["mask"]
        ys, xs = np.where(mask)
        if len(ys) == 0 or len(xs) == 0:
            return (0, 0, 0, 0)
        return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)

    c_bboxes = [get_cand_bbox(c) for c in candidates]
    clusters = [[c] for c in candidates]
    cluster_bboxes = list(c_bboxes)

    def bbox_distance(
        b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]
    ) -> int:
        dx = max(0, max(b1[0], b2[0]) - min(b1[2], b2[2]))
        dy = max(0, max(b1[1], b2[1]) - min(b1[3], b2[3]))
        return max(dx, dy)

    def merge_bboxes(
        b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        return (
            min(b1[0], b2[0]),
            min(b1[1], b2[1]),
            max(b1[2], b2[2]),
            max(b1[3], b2[3]),
        )

    while True:
        best_pair = None
        best_dist = float("inf")

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = bbox_distance(cluster_bboxes[i], cluster_bboxes[j])
                if dist <= max_dist:
                    merged = merge_bboxes(cluster_bboxes[i], cluster_bboxes[j])
                    mw = merged[2] - merged[0]
                    mh = merged[3] - merged[1]
                    if mw <= max_dim and mh <= max_dim and mw * mh <= max_area:
                        if dist < best_dist:
                            best_dist = dist
                            best_pair = (i, j, merged)

        if best_pair is None:
            break

        i, j, merged = best_pair
        clusters[i].extend(clusters[j])
        cluster_bboxes[i] = merged
        del clusters[j]
        del cluster_bboxes[j]

    return clusters


def slice_large_flux_candidate(
    candidate: Dict[str, Any],
    max_slice_dim: int = 1152,
) -> List[Dict[str, Any]]:
    """Splits an oversized OSB candidate along any axis exceeding max_slice_dim.

    - If H > max_slice_dim and W <= max_slice_dim: slice vertically (1D).
    - If W > max_slice_dim and H <= max_slice_dim: slice horizontally (1D).
    - If both W, H > max_slice_dim: slice as a 2D grid (Nx x Ny).
    - If both W, H <= max_slice_dim: return [candidate] intact.

    Uses dynamic overlap calculated as ~8% of slice dimension (min 32px, max 160px).
    Only returns non-empty sub-slices that contain actual text mask pixels.
    """
    bbox = candidate.get("bbox")
    mask = candidate.get("mask")
    if mask is None:
        return [candidate]

    img_h, img_w = mask.shape
    if bbox is not None:
        bx0, by0, bx1, by1 = bbox
    else:
        ys, xs = np.where(mask)
        if len(ys) == 0 or len(xs) == 0:
            return [candidate]
        bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1

    bw = bx1 - bx0
    bh = by1 - by0

    if bw <= max_slice_dim and bh <= max_slice_dim:
        return [candidate]

    target_step = 960
    num_x = max(1, math.ceil(bw / target_step)) if bw > max_slice_dim else 1
    num_y = max(1, math.ceil(bh / target_step)) if bh > max_slice_dim else 1

    if num_x == 1 and num_y == 1:
        return [candidate]

    sub_candidates = []
    base_idx = candidate.get("index", 1)
    fallback_color = candidate.get("fallback_color", (255, 255, 255))

    step_x = bw / num_x
    step_y = bh / num_y

    overlap_x = int(np.clip(round(step_x * 0.08), 32, 160)) if num_x > 1 else 0
    overlap_y = int(np.clip(round(step_y * 0.08), 32, 160)) if num_y > 1 else 0

    sub_i = 0
    for iy in range(num_y):
        sy0 = int(by0 + iy * step_y)
        sy1 = int(by0 + (iy + 1) * step_y) if iy < num_y - 1 else by1

        py0 = max(0, sy0 - overlap_y) if iy > 0 else sy0
        py1 = min(img_h, sy1 + overlap_y) if iy < num_y - 1 else sy1

        for ix in range(num_x):
            sx0 = int(bx0 + ix * step_x)
            sx1 = int(bx0 + (ix + 1) * step_x) if ix < num_x - 1 else bx1

            px0 = max(0, sx0 - overlap_x) if ix > 0 else sx0
            px1 = min(img_w, sx1 + overlap_x) if ix < num_x - 1 else sx1

            sub_mask = np.zeros_like(mask)
            sub_mask[py0:py1, px0:px1] = mask[py0:py1, px0:px1]

            if not np.any(sub_mask):
                continue

            sub_ys, sub_xs = np.where(sub_mask)
            sub_bbox = (
                int(sub_xs.min()),
                int(sub_ys.min()),
                int(sub_xs.max()) + 1,
                int(sub_ys.max()) + 1,
            )

            sub_candidates.append({
                "index": f"{base_idx}_{sub_i + 1}",
                "mask": sub_mask,
                "bbox": sub_bbox,
                "fallback_color": fallback_color,
            })
            sub_i += 1

    return sub_candidates if sub_candidates else [candidate]


def finish_outside_text_work(
    work: OutsideTextWork,
    skipped_bboxes: Optional[set] = None,
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """Run OSB inpainting for prepared work and attach render metadata."""
    pil_image = work.pil_image
    config = work.config
    verbose = work.verbose
    outside_text_results = work.outside_text_results
    raw_outside_text_results = work.raw_outside_text_results
    original_text_colors = work.original_text_colors
    total_bubble_mask = work.total_bubble_mask
    mask_groups = work.mask_groups
    img_w = work.img_w
    img_h = work.img_h
    outside_text_data = work.outside_text_data

    skipped_bboxes = skipped_bboxes or set()

    def _is_group_skipped(grp: Dict[str, Any], skipped_set: set) -> bool:
        if not skipped_set:
            return False

        cand_boxes = []
        orig_b = grp.get("original_bbox")
        if orig_b:
            ox = int(orig_b.get("x", 0))
            oy = int(orig_b.get("y", 0))
            ow = int(orig_b.get("width", 0))
            oh = int(orig_b.get("height", 0))
            if ow > 0 and oh > 0:
                cand_boxes.append((ox, oy, ox + ow, oy + oh))

        exp_b = grp.get("bbox")
        if exp_b:
            bx = int(exp_b.get("x", 0))
            by = int(exp_b.get("y", 0))
            bw = int(exp_b.get("width", 0))
            bh = int(exp_b.get("height", 0))
            if bw > 0 and bh > 0:
                cand_boxes.append((bx, by, bx + bw, by + bh))

        mask_indices = grp.get("mask_indices", [])
        if mask_indices and outside_text_results:
            for idx in mask_indices:
                if 0 <= idx < len(outside_text_results):
                    cand_boxes.append(tuple(int(c) for c in outside_text_results[idx][0]))
        if mask_indices and raw_outside_text_results:
            for idx in mask_indices:
                if 0 <= idx < len(raw_outside_text_results):
                    cand_boxes.append(tuple(int(c) for c in raw_outside_text_results[idx][0]))

        # 1. Exact match
        for cb in cand_boxes:
            if cb in skipped_set:
                return True

        # 2. Geometric intersection & overlap check
        for (gx1, gy1, gx2, gy2) in cand_boxes:
            gcx = (gx1 + gx2) / 2.0
            gcy = (gy1 + gy2) / 2.0
            g_area = max(1.0, float((gx2 - gx1) * (gy2 - gy1)))

            for (sx1, sy1, sx2, sy2) in skipped_set:
                scx = (sx1 + sx2) / 2.0
                scy = (sy1 + sy2) / 2.0
                s_area = max(1.0, float((sx2 - sx1) * (sy2 - sy1)))

                ix1 = max(gx1, sx1)
                iy1 = max(gy1, sy1)
                ix2 = min(gx2, sx2)
                iy2 = min(gy2, sy2)
                if ix2 > ix1 and iy2 > iy1:
                    inter_area = float((ix2 - ix1) * (iy2 - iy1))
                    if (inter_area / g_area >= 0.3) or (inter_area / s_area >= 0.3):
                        return True

                if (sx1 <= gcx <= sx2 and sy1 <= gcy <= sy2) or (gx1 <= scx <= gx2 and gy1 <= scy <= gy2):
                    return True

        return False

    extracted_text_colors = {}
    none_skipped_clip_bboxes = set()
    current_image = pil_image

    try:
        log_message("Inpainting outside text regions...", verbose=verbose)

        # Create inpainter based on selected method
        inpainting_method = config.outside_text.inpainting_method
        inpainter = None
        if inpainting_method in ("flux_klein_9b", "flux_klein_4b", "flux_dev", "sd_inpainting"):
            try:
                backend = config.outside_text.flux_backend
                variant = {
                    "flux_klein_9b": "9b",
                    "flux_klein_4b": "4b",
                    "flux_dev": "dev",
                    "sd_inpainting": "sd_inpainting",
                }[inpainting_method]
                inpainter = FluxKleinInpainter(
                    variant=variant,
                    device=config.device,
                    huggingface_token=config.outside_text.huggingface_token,
                    num_inference_steps=config.outside_text.flux_num_inference_steps,
                    low_vram=config.outside_text.flux_low_vram,
                    luminance_correction=config.outside_text.flux_luminance_correction,
                    upscale_small_crops=config.outside_text.flux_upscale_small_crops,
                    backend=backend,
                    sdcpp_cache_mode=config.outside_text.flux_sdcpp_cache_mode,
                    sdcpp_diffusion_quant=config.outside_text.flux_sdcpp_diffusion_quant,
                    sdcpp_text_encoder_quant=config.outside_text.flux_sdcpp_text_encoder_quant,
                    verbose=verbose,
                    cloudflare_accounts=config.outside_text.cloudflare_accounts,
                    colab_server_url=config.outside_text.colab_server_url,
                )
                backend_label = (
                    "Remote Colab"
                    if backend == "remote_colab"
                    else (
                        "Cloudflare"
                        if backend == "cloudflare"
                        else ("sd.cpp" if backend == "sdcpp" else "SDNQ")
                    )
                )
                model_label = {
                    "9b": "Flux.2 Klein 9B",
                    "4b": "Flux.2 Klein 4B",
                    "dev": "Flux.2 Dev",
                    "sd_inpainting": "Stable Diffusion v1.5 Inpainting",
                }[variant]
                log_message(
                    f"Using {model_label} ({backend_label}) for inpainting",
                    verbose=verbose,
                )
            except Exception as e:
                log_message(
                    f"{inpainting_method} unavailable ({e}), falling back to OpenCV",
                    verbose=verbose,
                )

        if inpainting_method == "flux_kontext":
            try:
                backend = config.outside_text.flux_backend
                low_vram = (
                    config.outside_text.flux_low_vram if backend == "sdnq" else False
                )
                inpainter = FluxKontextInpainter(
                    device=config.device,
                    huggingface_token=config.outside_text.huggingface_token,
                    num_inference_steps=config.outside_text.flux_num_inference_steps,
                    residual_diff_threshold=config.outside_text.flux_residual_diff_threshold,
                    backend=backend,
                    low_vram=low_vram,
                    sdcpp_cache_mode=config.outside_text.flux_sdcpp_cache_mode,
                    sdcpp_diffusion_quant=config.outside_text.flux_sdcpp_diffusion_quant,
                    sdcpp_text_encoder_quant=config.outside_text.flux_sdcpp_text_encoder_quant,
                )
                backend_label = {
                    "sdnq": "SDNQ",
                    "sdcpp": "sd.cpp",
                    "nunchaku": "Nunchaku",
                }[backend]
                log_message(
                    f"Using Flux.1 Kontext ({backend_label}) for inpainting",
                    verbose=verbose,
                )
            except Exception as e:
                log_message(
                    f"Flux Kontext unavailable ({e}), falling back to OpenCV",
                    verbose=verbose,
                )

        if inpainting_method == "none":
            inpainter = None
            log_message(
                "Using text background mode (no inpainting for non-solid regions)",
                verbose=verbose,
            )
        elif inpainting_method == "opencv" or inpainter is None:
            inpainter = None
            log_message("Using OpenCV simple fill for inpainting", verbose=verbose)
        current_image = pil_image
        temp_files = []
        none_skipped_clip_bboxes = set()
        try:
            if mask_groups:
                base_seed = (
                    random.randint(1, 999999)
                    if config.outside_text.seed == -1
                    else config.outside_text.seed
                )

                extracted_text_colors = {}
                flux_inpaints = 0
                cv2_inpaints = 0
                none_skips = 0
                group_flux_regions = bool(config.outside_text.flux_group_regions)
                grouped_flux_candidates = []
                request_coordinator = getattr(config, "request_coordinator", None)
                pending_flux_candidates = []

                def apply_candidate_simple_fill(candidate, color_to_use):
                    new_img = current_image.copy()
                    candidate_group = candidate["group"]
                    candidate_mask = candidate["mask"]
                    candidate_original_bbox = candidate["original_bbox_dict"]
                    c_ox0, c_oy0, c_ox1, c_oy1 = candidate["original_bounds"]

                    mask_indices = candidate_group.get("mask_indices", [])
                    if mask_indices and outside_text_results:
                        p_x0 = max(
                            0,
                            int(
                                min(
                                    [
                                        outside_text_results[idx][0][0]
                                        for idx in mask_indices
                                    ]
                                )
                            ),
                        )
                        p_y0 = max(
                            0,
                            int(
                                min(
                                    [
                                        outside_text_results[idx][0][1]
                                        for idx in mask_indices
                                    ]
                                )
                            ),
                        )
                        p_x1 = min(
                            img_w,
                            int(
                                max(
                                    [
                                        outside_text_results[idx][0][2]
                                        for idx in mask_indices
                                    ]
                                )
                            ),
                        )
                        p_y1 = min(
                            img_h,
                            int(
                                max(
                                    [
                                        outside_text_results[idx][0][3]
                                        for idx in mask_indices
                                    ]
                                )
                            ),
                        )
                    elif (
                        candidate_original_bbox
                        and c_ox1 is not None
                        and c_ox0 is not None
                        and c_oy1 is not None
                        and c_oy0 is not None
                    ):
                        p_x0, p_y0, p_x1, p_y1 = c_ox0, c_oy0, c_ox1, c_oy1
                    else:
                        mask_pil = Image.fromarray(
                            (candidate_mask * 255).astype(np.uint8), mode="L"
                        )
                        patch = Image.new("RGB", new_img.size, color_to_use)
                        new_img.paste(patch, (0, 0), mask=mask_pil)
                        return new_img

                    if p_x1 > p_x0 and p_y1 > p_y0:
                        rect_mask = np.zeros((img_h, img_w), dtype=bool)
                        rect_mask[p_y0:p_y1, p_x0:p_x1] = True
                        rect_mask = np.logical_and(
                            rect_mask, np.logical_not(total_bubble_mask)
                        )

                        region_mask = rect_mask[p_y0:p_y1, p_x0:p_x1]
                        if np.any(region_mask):
                            mask_pil = Image.fromarray(
                                (region_mask * 255).astype(np.uint8), mode="L"
                            )
                            patch = Image.new(
                                "RGB", (p_x1 - p_x0, p_y1 - p_y0), color_to_use
                            )
                            new_img.paste(patch, (p_x0, p_y0), mask=mask_pil)

                    return new_img

                def flush_pending_flux_candidates():
                    nonlocal current_image, flux_inpaints, cv2_inpaints
                    if not pending_flux_candidates:
                        return

                    candidates = list(pending_flux_candidates)
                    pending_flux_candidates.clear()
                    waves = partition_non_overlapping_waves(
                        candidates,
                        lambda candidate: candidate["context_bbox"],
                    )
                    log_message(
                        f"Scheduling OSB Flux in {len(waves)} wave(s)",
                        verbose=verbose,
                    )

                    for wave in waves:
                        base_image = current_image

                        def make_job(candidate):
                            def job():
                                image_for_job = base_image.copy()
                                try:
                                    result_image = inpainter.inpaint_mask(
                                        image_for_job,
                                        candidate["mask"],
                                        seed=candidate["seed"],
                                        verbose=verbose,
                                        strict_mask_clipping=True,
                                        composite_clip_bbox=candidate[
                                            "composite_clip_bbox"
                                        ],
                                    )
                                    if result_image is image_for_job:
                                        raise RuntimeError(
                                            "Flux returned original image (no inpaint)"
                                        )
                                    return {
                                        "candidate": candidate,
                                        "image": result_image,
                                        "error": None,
                                    }
                                except Exception as e:
                                    return {
                                        "candidate": candidate,
                                        "image": None,
                                        "error": e,
                                    }

                            return job

                        results = request_coordinator.map_ordered(
                            [make_job(candidate) for candidate in wave]
                        )
                        for result in results:
                            candidate = result["candidate"]
                            if result["error"] is not None:
                                fallback_color_to_use = candidate["fallback_color"]
                                log_message(
                                    f"Flux failed for OSB region {candidate['index']}"
                                    f" (Flux inpainting error: {result['error']}); "
                                    f"falling back to CV2 fill ({fallback_color_to_use})",
                                    always_print=True,
                                )
                                current_image = apply_candidate_simple_fill(
                                    candidate, fallback_color_to_use
                                )
                                cv2_inpaints += 1
                                continue

                            context_bbox = candidate["context_bbox"]
                            if context_bbox is None:
                                current_image = result["image"]
                            else:
                                current_image = paste_image_region(
                                    current_image,
                                    result["image"],
                                    context_bbox,
                                )
                            flux_inpaints += 1

                for i, group in enumerate(mask_groups):
                    log_message(
                        f"Inpainting outside text region {i + 1}/{len(mask_groups)}",
                        verbose=verbose,
                    )
                    combined_mask = group["combined_mask"]
                    combined_mask = np.logical_and(
                        combined_mask, np.logical_not(total_bubble_mask)
                    )
                    if not np.any(combined_mask):
                        log_message(
                            "Skipping outside text region after bubble masking (no remaining area)",
                            verbose=verbose,
                        )
                        continue
                    region_seed = base_seed + i if base_seed > 0 else base_seed

                    if _is_group_skipped(group, skipped_bboxes):
                        log_message(
                            f"Skipping OSB region {i + 1} inpainting (skipped by user)",
                            always_print=True,
                        )
                        continue

                    original_bbox_dict = group.get("bbox") or group.get("original_bbox")
                    composite_clip_bbox = None
                    fill_color = None
                    fallback_fill_color = None
                    ox0 = oy0 = ox1 = oy1 = None
                    if original_bbox_dict:
                        ox = int(original_bbox_dict.get("x", 0))
                        oy = int(original_bbox_dict.get("y", 0))
                        ow = int(original_bbox_dict.get("width", 0))
                        oh = int(original_bbox_dict.get("height", 0))
                        if ow > 0 and oh > 0:
                            composite_clip_bbox = (ox, oy, ox + ow, oy + oh)
                            ox0 = max(0, min(img_w, ox))
                            oy0 = max(0, min(img_h, oy))
                            ox1 = max(0, min(img_w, ox + ow))
                            oy1 = max(0, min(img_h, oy + oh))

                            # Determine detected text color for this region to ensure contrast
                            group_bg_is_dark = None
                            if original_text_colors:
                                votes_dark = 0
                                votes_light = 0
                                gx1, gy1, gx2, gy2 = ox, oy, ox + ow, oy + oh

                                for (
                                    bx1,
                                    by1,
                                    bx2,
                                    by2,
                                ), t_dark in original_text_colors.items():
                                    # Check if center of OCR box is inside group box
                                    bcx = (bx1 + bx2) / 2
                                    bcy = (by1 + by2) / 2
                                    if (
                                        bcx >= gx1
                                        and bcx <= gx2
                                        and bcy >= gy1
                                        and bcy <= gy2
                                    ):
                                        if t_dark:
                                            votes_dark += 1
                                        else:
                                            votes_light += 1
                                if votes_dark > 0 or votes_light > 0:
                                    group_bg_is_dark = votes_dark >= votes_light

                                    # Detected value represents background brightness
                                    fallback_fill_color = (
                                        (0, 0, 0)
                                        if group_bg_is_dark
                                        else (255, 255, 255)
                                    )

                                    t_type = "Dark" if group_bg_is_dark else "Light"
                                    f_col = (
                                        "White"
                                        if fallback_fill_color == (255, 255, 255)
                                        else "Black"
                                    )
                                    log_message(
                                        f"OSB Region {i + 1}: Detected {t_type} background. "
                                        f"Fallback fill: {f_col}.",
                                        verbose=verbose,
                                    )

                            # Expanded sampling around the original bbox to find background color
                            mask_indices = group.get("mask_indices", [])
                            if mask_indices and raw_outside_text_results:
                                rx0 = int(
                                    min(
                                        [
                                            raw_outside_text_results[idx][0][0]
                                            for idx in mask_indices
                                        ]
                                    )
                                )
                                ry0 = int(
                                    min(
                                        [
                                            raw_outside_text_results[idx][0][1]
                                            for idx in mask_indices
                                        ]
                                    )
                                )
                                rx1 = int(
                                    max(
                                        [
                                            raw_outside_text_results[idx][0][2]
                                            for idx in mask_indices
                                        ]
                                    )
                                )
                                ry1 = int(
                                    max(
                                        [
                                            raw_outside_text_results[idx][0][3]
                                            for idx in mask_indices
                                        ]
                                    )
                                )
                            else:
                                rx0, ry0, rx1, ry1 = ox, oy, ox + ow, oy + oh

                            sx1 = max(0, rx0 - OSB_EXPANSION_PX)
                            sy1 = max(0, ry0 - OSB_EXPANSION_PX)
                            sx2 = min(img_w, rx1 + OSB_EXPANSION_PX)
                            sy2 = min(img_h, ry1 + OSB_EXPANSION_PX)

                            if sx2 > sx1 and sy2 > sy1:
                                mask_h, mask_w = sy2 - sy1, sx2 - sx1
                                local_mask = np.ones((mask_h, mask_w), dtype=bool)

                                lx0 = max(0, rx0 - sx1)
                                ly0 = max(0, ry0 - sy1)
                                lx1 = min(mask_w, rx1 - sx1)
                                ly1 = min(mask_h, ry1 - sy1)

                                if lx1 > lx0 and ly1 > ly0:
                                    local_mask[ly0:ly1, lx0:lx1] = False

                                border_pixels = None
                                min_border_pixels = 20
                                if np.count_nonzero(local_mask) >= min_border_pixels:
                                    sampling_crop = current_image.crop(
                                        (sx1, sy1, sx2, sy2)
                                    )
                                    crop_np = np.array(sampling_crop.convert("RGB"))
                                    border_pixels = crop_np[local_mask]

                                if border_pixels is not None and border_pixels.size > 0:
                                    # Calculate text color using LAB contrast thresholding
                                    bg_rgb = np.median(border_pixels, axis=0).astype(
                                        np.uint8
                                    )
                                    bg_lab = cv2.cvtColor(
                                        np.uint8([[bg_rgb]]), cv2.COLOR_RGB2LAB
                                    )[0][0]

                                    crop_rgb = np.array(
                                        pil_image.crop((rx0, ry0, rx1, ry1)).convert(
                                            "RGB"
                                        )
                                    )
                                    crop_lab = cv2.cvtColor(
                                        crop_rgb, cv2.COLOR_RGB2LAB
                                    ).astype(np.float32)
                                    dist_map = np.linalg.norm(
                                        crop_lab - bg_lab.astype(np.float32), axis=2
                                    )

                                    robust_max_dist = np.percentile(dist_map, 95)
                                    CONTRAST_THRESHOLD = max(30, robust_max_dist * 0.6)
                                    contrast_mask = (
                                        dist_map > CONTRAST_THRESHOLD
                                    ).astype(np.uint8) * 255

                                    kernel_3 = np.ones((3, 3), np.uint8)
                                    contrast_mask = cv2.morphologyEx(
                                        contrast_mask, cv2.MORPH_CLOSE, kernel_3
                                    )
                                    contrast_mask = cv2.erode(
                                        contrast_mask,
                                        np.ones((2, 2), np.uint8),
                                        iterations=1,
                                    )

                                    contours, _ = cv2.findContours(
                                        contrast_mask,
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE,
                                    )
                                    clean_mask = np.zeros_like(contrast_mask)
                                    MIN_COMPONENT_AREA = 4
                                    for cnt in contours:
                                        if cv2.contourArea(cnt) >= MIN_COMPONENT_AREA:
                                            cv2.drawContours(
                                                clean_mask, [cnt], -1, 255, cv2.FILLED
                                            )

                                    text_pixels_rgb = crop_rgb[clean_mask == 255]
                                    if len(text_pixels_rgb) >= 10:
                                        text_color_rgb = tuple(
                                            np.median(text_pixels_rgb, axis=0).astype(
                                                int
                                            )
                                        )
                                        hsv = cv2.cvtColor(
                                            np.uint8([[text_color_rgb]]),
                                            cv2.COLOR_RGB2HSV,
                                        )[0][0]
                                        if hsv[1] < 25:
                                            text_color_rgb = (
                                                (0, 0, 0)
                                                if hsv[2] < 128
                                                else (255, 255, 255)
                                            )
                                        extracted_text_colors[composite_clip_bbox] = (
                                            text_color_rgb
                                        )

                                    bg_rgb_med = np.median(
                                        border_pixels, axis=0
                                    ).astype(int)
                                    diffs = np.max(
                                        np.abs(border_pixels.astype(int) - bg_rgb_med),
                                        axis=1,
                                    )
                                    solid_ratio = float(
                                        np.mean(diffs <= OSB_COLOR_TOLERANCE)
                                    )
                                    is_border_solid = (
                                        solid_ratio >= OSB_SOLID_RATIO_THRESHOLD
                                    )

                                    if (
                                        bg_rgb_med[0] >= OSB_WHITE_SNAP_THRESH
                                        and bg_rgb_med[1] >= OSB_WHITE_SNAP_THRESH
                                        and bg_rgb_med[2] >= OSB_WHITE_SNAP_THRESH
                                    ):
                                        detected_color = (255, 255, 255)
                                    elif (
                                        bg_rgb_med[0] <= OSB_BLACK_SNAP_THRESH
                                        and bg_rgb_med[1] <= OSB_BLACK_SNAP_THRESH
                                        and bg_rgb_med[2] <= OSB_BLACK_SNAP_THRESH
                                    ):
                                        detected_color = (0, 0, 0)
                                    else:
                                        detected_color = (
                                            int(bg_rgb_med[0]),
                                            int(bg_rgb_med[1]),
                                            int(bg_rgb_med[2]),
                                        )

                                    if is_border_solid or fallback_fill_color is None:
                                        fallback_fill_color = detected_color

                                    force_fill = inpainting_method == "opencv"

                                    # Get raw text bounds for this group to check solid color and for cv2 fill
                                    p_x0, p_y0, p_x1, p_y1 = ox, oy, ox + ow, oy + oh
                                    mask_indices = group.get("mask_indices", [])
                                    if mask_indices and raw_outside_text_results:
                                        p_x0 = max(
                                            0,
                                            int(
                                                min(
                                                    [
                                                        raw_outside_text_results[idx][0][0]
                                                        for idx in mask_indices
                                                    ]
                                                )
                                            ),
                                        )
                                        p_y0 = max(
                                            0,
                                            int(
                                                min(
                                                    [
                                                        raw_outside_text_results[idx][0][1]
                                                        for idx in mask_indices
                                                    ]
                                                )
                                            ),
                                        )
                                        p_x1 = min(
                                            img_w,
                                            int(
                                                max(
                                                    [
                                                        raw_outside_text_results[idx][0][2]
                                                        for idx in mask_indices
                                                    ]
                                                )
                                            ),
                                        )
                                        p_y1 = min(
                                            img_h,
                                            int(
                                                max(
                                                    [
                                                        raw_outside_text_results[idx][0][3]
                                                        for idx in mask_indices
                                                    ]
                                                )
                                            ),
                                        )

                                    # Simply check if the boundary is solid color
                                    expanded_is_solid = is_border_solid
                                    if not force_fill:
                                        box_w = max(1, p_x1 - p_x0)
                                        box_h = max(1, p_y1 - p_y0)
                                        adaptive_pad = max(2, min(10, int(round(min(box_w, box_h) * 0.06))))
                                        ex_sx1 = max(0, p_x0 - adaptive_pad)
                                        ex_sy1 = max(0, p_y0 - adaptive_pad)
                                        ex_sx2 = min(img_w, p_x1 + adaptive_pad)
                                        ex_sy2 = min(img_h, p_y1 + adaptive_pad)

                                        if ex_sx2 > ex_sx1 and ex_sy2 > ex_sy1:
                                            # Grab boundary pixels using Numpy directly for speed
                                            crop_img = current_image.crop(
                                                (ex_sx1, ex_sy1, ex_sx2, ex_sy2)
                                            )
                                            ecrop_np = np.array(crop_img.convert("RGB"))
                                            elocal_mask = np.ones(
                                                ecrop_np.shape[:2], dtype=bool
                                            )

                                            ix1 = max(0, p_x0 - ex_sx1)
                                            iy1 = max(0, p_y0 - ex_sy1)
                                            ix2 = min(ecrop_np.shape[1], p_x1 - ex_sx1)
                                            iy2 = min(ecrop_np.shape[0], p_y1 - ex_sy1)

                                            if ix2 > ix1 and iy2 > iy1:
                                                elocal_mask[iy1:iy2, ix1:ix2] = False

                                            eborder_pixels = ecrop_np[elocal_mask]
                                            if eborder_pixels.size > 0:
                                                ebg_rgb_med = np.median(
                                                    eborder_pixels, axis=0
                                                ).astype(int)
                                                ediffs = np.max(
                                                    np.abs(
                                                        eborder_pixels.astype(int)
                                                        - ebg_rgb_med
                                                    ),
                                                    axis=1,
                                                )
                                                esolid_ratio = float(
                                                    np.mean(
                                                        ediffs <= OSB_COLOR_TOLERANCE
                                                    )
                                                )
                                                if (
                                                    ebg_rgb_med[0]
                                                    >= OSB_WHITE_SNAP_THRESH
                                                    and ebg_rgb_med[1]
                                                    >= OSB_WHITE_SNAP_THRESH
                                                    and ebg_rgb_med[2]
                                                    >= OSB_WHITE_SNAP_THRESH
                                                ):
                                                    edetected_color = (255, 255, 255)
                                                elif (
                                                    ebg_rgb_med[0]
                                                    <= OSB_BLACK_SNAP_THRESH
                                                    and ebg_rgb_med[1]
                                                    <= OSB_BLACK_SNAP_THRESH
                                                    and ebg_rgb_med[2]
                                                    <= OSB_BLACK_SNAP_THRESH
                                                ):
                                                    edetected_color = (0, 0, 0)
                                                else:
                                                    edetected_color = (
                                                        int(ebg_rgb_med[0]),
                                                        int(ebg_rgb_med[1]),
                                                        int(ebg_rgb_med[2]),
                                                    )

                                                if (
                                                    esolid_ratio
                                                    >= OSB_SOLID_RATIO_THRESHOLD
                                                ):
                                                    expanded_is_solid = True
                                                    fallback_fill_color = (
                                                        edetected_color
                                                    )

                                    should_simple_fill = expanded_is_solid or force_fill

                                    if should_simple_fill:
                                        fill_color = (
                                            fallback_fill_color
                                            if fallback_fill_color is not None
                                            else detected_color
                                        )

                                        if force_fill:
                                            log_message(
                                                f"Using OpenCV simple fill for OSB region: {fill_color} background",
                                                verbose=verbose,
                                            )
                                        else:
                                            log_message(
                                                f"Skipping Flux for OSB region: detected solid {fill_color} background",
                                                verbose=verbose,
                                            )

                    def apply_simple_fill(color_to_use):
                        new_img = current_image.copy()

                        mask_indices = group.get("mask_indices", [])
                        if mask_indices and raw_outside_text_results:
                            p_x0 = max(
                                0,
                                int(
                                    min(
                                        [
                                            raw_outside_text_results[idx][0][0]
                                            for idx in mask_indices
                                        ]
                                    )
                                ),
                            )
                            p_y0 = max(
                                0,
                                int(
                                    min(
                                        [
                                            raw_outside_text_results[idx][0][1]
                                            for idx in mask_indices
                                        ]
                                    )
                                ),
                            )
                            p_x1 = min(
                                img_w,
                                int(
                                    max(
                                        [
                                            raw_outside_text_results[idx][0][2]
                                            for idx in mask_indices
                                        ]
                                    )
                                ),
                            )
                            p_y1 = min(
                                img_h,
                                int(
                                    max(
                                        [
                                            raw_outside_text_results[idx][0][3]
                                            for idx in mask_indices
                                        ]
                                    )
                                ),
                            )
                        elif (
                            original_bbox_dict
                            and ox1 is not None
                            and ox0 is not None
                            and oy1 is not None
                            and oy0 is not None
                        ):
                            p_x0, p_y0, p_x1, p_y1 = ox0, oy0, ox1, oy1
                        elif original_bbox_dict:
                            p_x0, p_y0, p_x1, p_y1 = ox, oy, ox + ow, oy + oh
                        else:
                            # Full mask fill fallback
                            mask_pil = Image.fromarray(
                                (combined_mask * 255).astype(np.uint8), mode="L"
                            )
                            patch = Image.new("RGB", new_img.size, color_to_use)
                            new_img.paste(patch, (0, 0), mask=mask_pil)
                            return new_img

                        if p_x1 > p_x0 and p_y1 > p_y0:
                            box_w = p_x1 - p_x0
                            box_h = p_y1 - p_y0
                            char_dim = min(box_w, box_h)
                            # Dynamic padding: scale with character height/width, not fixed
                            target_pad = max(2, min(12, int(round(char_dim * 0.06))))

                            # Obstacle-aware probe: avoid expanding into dark panel borders or line art
                            img_arr = np.array(current_image.convert("L"))
                            is_light_bg = (sum(color_to_use) / 3.0) > 128.0

                            def is_obstacle(val):
                                return (val < 70) if is_light_bg else (val > 185)

                            pad_top = 0
                            for step in range(1, target_pad + 1):
                                test_y = p_y0 - step
                                if test_y < 0:
                                    break
                                row_slice = img_arr[test_y, p_x0:p_x1]
                                if np.any(is_obstacle(row_slice)):
                                    break
                                pad_top = step

                            pad_bottom = 0
                            for step in range(1, target_pad + 1):
                                test_y = p_y1 + step - 1
                                if test_y >= img_h:
                                    break
                                row_slice = img_arr[test_y, p_x0:p_x1]
                                if np.any(is_obstacle(row_slice)):
                                    break
                                pad_bottom = step

                            pad_left = 0
                            for step in range(1, target_pad + 1):
                                test_x = p_x0 - step
                                if test_x < 0:
                                    break
                                col_slice = img_arr[p_y0 - pad_top : p_y1 + pad_bottom, test_x]
                                if np.any(is_obstacle(col_slice)):
                                    break
                                pad_left = step

                            pad_right = 0
                            for step in range(1, target_pad + 1):
                                test_x = p_x1 + step - 1
                                if test_x >= img_w:
                                    break
                                col_slice = img_arr[p_y0 - pad_top : p_y1 + pad_bottom, test_x]
                                if np.any(is_obstacle(col_slice)):
                                    break
                                pad_right = step

                            fill_x0 = max(0, p_x0 - pad_left)
                            fill_y0 = max(0, p_y0 - pad_top)
                            fill_x1 = min(img_w, p_x1 + pad_right)
                            fill_y1 = min(img_h, p_y1 + pad_bottom)

                            rect_mask = np.zeros((img_h, img_w), dtype=bool)
                            rect_mask[fill_y0:fill_y1, fill_x0:fill_x1] = True
                            rect_mask = np.logical_and(
                                rect_mask, np.logical_not(total_bubble_mask)
                            )

                            region_mask = rect_mask[fill_y0:fill_y1, fill_x0:fill_x1]
                            if np.any(region_mask):
                                mask_pil = Image.fromarray(
                                    (region_mask * 255).astype(np.uint8), mode="L"
                                )
                                patch = Image.new(
                                    "RGB", (fill_x1 - fill_x0, fill_y1 - fill_y0), color_to_use
                                )
                                new_img.paste(patch, (fill_x0, fill_y0), mask=mask_pil)

                        return new_img

                    if fill_color is not None:
                        flush_pending_flux_candidates()
                        current_image = apply_simple_fill(fill_color)
                        cv2_inpaints += 1
                        continue

                    if inpainting_method == "none":
                        if composite_clip_bbox:
                            none_skipped_clip_bboxes.add(composite_clip_bbox)
                        none_skips += 1
                        log_message(
                            f"Skipping inpaint for non-solid OSB region {i + 1} (none mode)",
                            verbose=verbose,
                        )
                        continue

                    if group_flux_regions and inpainter is not None:
                        flush_pending_flux_candidates()
                        grouped_mask = combined_mask.copy()
                        if composite_clip_bbox is not None:
                            clip_x1, clip_y1, clip_x2, clip_y2 = composite_clip_bbox
                            clip_x1 = max(0, min(img_w, clip_x1))
                            clip_x2 = max(0, min(img_w, clip_x2))
                            clip_y1 = max(0, min(img_h, clip_y1))
                            clip_y2 = max(0, min(img_h, clip_y2))

                            clipped_grouped_mask = np.zeros_like(grouped_mask)
                            if clip_x2 > clip_x1 and clip_y2 > clip_y1:
                                clipped_grouped_mask[
                                    clip_y1:clip_y2, clip_x1:clip_x2
                                ] = grouped_mask[clip_y1:clip_y2, clip_x1:clip_x2]
                            grouped_mask = clipped_grouped_mask

                        if not np.any(grouped_mask):
                            log_message(
                                f"Skipping OSB region {i + 1} after grouped clip (no remaining area)",
                                verbose=verbose,
                            )
                            continue

                        cand_bbox = None
                        if composite_clip_bbox is not None:
                            cand_bbox = composite_clip_bbox
                        else:
                            ys, xs = np.where(grouped_mask)
                            if len(ys) > 0 and len(xs) > 0:
                                cand_bbox = (
                                    int(xs.min()),
                                    int(ys.min()),
                                    int(xs.max()) + 1,
                                    int(ys.max()) + 1,
                                )

                        grouped_flux_candidates.append(
                            {
                                "index": i + 1,
                                "mask": grouped_mask,
                                "bbox": cand_bbox,
                                "fallback_color": (
                                    fallback_fill_color
                                    if fallback_fill_color
                                    else (255, 255, 255)
                                ),
                            }
                        )
                        log_message(
                            f"Queued OSB region {i + 1} for grouped Flux inpainting",
                            verbose=verbose,
                        )
                        continue

                    if request_coordinator is not None and inpainter is not None:
                        fallback_color_to_use = (
                            fallback_fill_color
                            if fallback_fill_color
                            else (255, 255, 255)
                        )
                        pending_flux_candidates.append(
                            {
                                "index": i + 1,
                                "mask": combined_mask.copy(),
                                "seed": region_seed,
                                "composite_clip_bbox": composite_clip_bbox,
                                "fallback_color": fallback_color_to_use,
                                "group": group,
                                "original_bbox_dict": original_bbox_dict,
                                "original_bounds": (ox0, oy0, ox1, oy1),
                                "context_bbox": expanded_mask_bbox(
                                    combined_mask, current_image.size
                                ),
                            }
                        )
                        log_message(
                            f"Queued OSB region {i + 1} for intra-page Flux scheduling",
                            verbose=verbose,
                        )
                        continue

                    flux_failed = False
                    flux_fail_reason = None
                    inpainted_image = None

                    if inpainter is None:
                        flux_failed = True
                        flux_fail_reason = "Flux inpainter unavailable"
                    else:
                        try:
                            inpainted_image = inpainter.inpaint_mask(
                                current_image,
                                combined_mask,
                                seed=region_seed,
                                verbose=verbose,
                                strict_mask_clipping=True,
                                composite_clip_bbox=composite_clip_bbox,
                            )
                            if inpainted_image is current_image:
                                flux_failed = True
                                flux_fail_reason = (
                                    "Flux returned original image (no inpaint)"
                                )
                        except Exception as e:
                            flux_failed = True
                            flux_fail_reason = f"Flux inpainting error: {e}"

                    if flux_failed:
                        fallback_color_to_use = (
                            fallback_fill_color
                            if fallback_fill_color
                            else (255, 255, 255)
                        )
                        log_message(
                            f"Flux failed for OSB region {i + 1}"
                            + (f" ({flux_fail_reason})" if flux_fail_reason else "")
                            + f"; falling back to CV2 fill ({fallback_color_to_use})",
                            always_print=True,
                        )
                        current_image = apply_simple_fill(fallback_color_to_use)
                        cv2_inpaints += 1
                        continue

                    flux_inpaints += 1
                    # Save to disk if more regions remain to reduce memory usage
                    if i < len(mask_groups) - 1:
                        temp_file = None
                        try:
                            temp_fd, temp_file = tempfile.mkstemp(suffix=".png")
                            os.close(temp_fd)
                            inpainted_image.save(temp_file, format="PNG")
                            temp_files.append(temp_file)

                            with Image.open(temp_file) as img_tmp:
                                img_tmp.load()
                                current_image = img_tmp.copy()

                            del inpainted_image
                            gc.collect()
                            log_message(
                                "Saved intermediate inpainting result to disk",
                                verbose=verbose,
                            )
                        except Exception as e:
                            log_message(
                                "Warning: Failed to save intermediate image to disk: "
                                f"{e}. Continuing with in-memory processing.",
                                verbose=verbose,
                            )
                            # Fallback to in-memory if disk save fails
                            current_image = inpainted_image
                            if temp_file and temp_file in temp_files:
                                temp_files.remove(temp_file)
                    else:
                        current_image = inpainted_image

                flush_pending_flux_candidates()

                if grouped_flux_candidates:
                    # 1. Adaptively slice oversized candidates along exceeding axes (1D/2D) with dynamic overlap
                    sliced_candidates = []
                    for cand in grouped_flux_candidates:
                        sliced_candidates.extend(
                            slice_large_flux_candidate(cand, max_slice_dim=1024)
                        )

                    # 2. Smart cluster into safe-sized groups (max_dim=1024, max_area=800k)
                    clusters = cluster_flux_candidates(
                        sliced_candidates,
                        max_dim=1024,
                        max_dist=300,
                        max_area=800_000,
                    )
                    log_message(
                        f"Partitioned {len(sliced_candidates)} OSB regions into {len(clusters)} smart Flux cluster(s)",
                        verbose=verbose,
                    )

                    cluster_concurrency = 1
                    if inpainter is not None:
                        if getattr(inpainter, "backend", None) == "remote_colab" and getattr(inpainter, "colab_client", None):
                            inpainter.colab_client.ensure_checked()
                            from core.image.colab_client import _SESSION_DEAD_SERVERS
                            alive_urls = [u for u in inpainter.colab_client.server_urls if u not in _SESSION_DEAD_SERVERS]
                            cluster_concurrency = max(1, len(alive_urls))
                        elif getattr(inpainter, "backend", None) == "cloudflare" and getattr(inpainter, "cf_client", None):
                            cluster_concurrency = max(1, len(getattr(inpainter.cf_client, "accounts", [])))

                    for batch_start in range(0, len(clusters), cluster_concurrency):
                        batch = list(enumerate(clusters))[batch_start : batch_start + cluster_concurrency]
                        base_image = current_image.copy()

                        def _process_single_cluster(item):
                            c_idx, cluster = item
                            c_mask = np.zeros((img_h, img_w), dtype=bool)
                            for candidate in cluster:
                                c_mask = np.logical_or(c_mask, candidate["mask"])

                            if not np.any(c_mask):
                                return c_idx, cluster, c_mask, None, None

                            log_message(
                                f"Running Flux inpainting for OSB cluster {c_idx + 1}/{len(clusters)} "
                                f"({len(cluster)} region(s))",
                                verbose=verbose,
                            )
                            cluster_seed = (
                                base_seed + c_idx if base_seed > 0 else base_seed
                            )
                            inpaint_kwargs = {
                                "seed": cluster_seed,
                                "verbose": verbose,
                                "strict_mask_clipping": True,
                                "ocr_params": {
                                    "type": "outside_text_cluster",
                                    "cluster_idx": c_idx + 1,
                                    "regions": len(cluster),
                                },
                            }
                            try:
                                res_img = inpainter.inpaint_mask(
                                    base_image.copy(),
                                    c_mask,
                                    **inpaint_kwargs,
                                )
                                if res_img is base_image:
                                    raise RuntimeError(
                                        "Flux returned original image (no inpaint)"
                                    )
                                return c_idx, cluster, c_mask, res_img, None
                            except Exception as e:
                                return c_idx, cluster, c_mask, None, e

                        if len(batch) == 1:
                            batch_results = [_process_single_cluster(batch[0])]
                        else:
                            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                                batch_results = list(executor.map(_process_single_cluster, batch))

                        for c_idx, cluster, c_mask, res_img, err in batch_results:
                            if err is None and res_img is not None:
                                mask_pil = Image.fromarray(
                                    (c_mask * 255).astype(np.uint8), mode="L"
                                )
                                current_image.paste(res_img, (0, 0), mask=mask_pil)
                                flux_inpaints += len(cluster)
                            else:
                                log_message(
                                    f"Flux cluster {c_idx + 1} inpainting failed ({err}); falling back to CV2 fill",
                                    always_print=True,
                                )
                                for candidate in cluster:
                                    mask_pil = Image.fromarray(
                                        (candidate["mask"] * 255).astype(np.uint8),
                                        mode="L",
                                    )
                                    patch = Image.new(
                                        "RGB",
                                        current_image.size,
                                        candidate["fallback_color"],
                                    )
                                    current_image.paste(patch, (0, 0), mask=mask_pil)
                                cv2_inpaints += len(cluster)

                log_message("Outside text inpainting completed", verbose=verbose)
                parts = [
                    f"Flux: {flux_inpaints}",
                    f"CV2: {cv2_inpaints}",
                ]
                if none_skips:
                    parts.append(f"Skipped (none): {none_skips}")
                log_message(
                    f"Inpainted {len(mask_groups)} outside text regions ({', '.join(parts)})",
                    always_print=True,
                )
        finally:
            for temp_file in temp_files:
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass

        _apply_inpaint_render_metadata(
            outside_text_data,
            extracted_text_colors,
            none_skipped_clip_bboxes,
        )
        return current_image, outside_text_data

    except Exception as e:
        log_message(
            f"Error during outside text inpainting: {e}",
            always_print=True,
        )
        return pil_image, outside_text_data


def process_outside_text(
    pil_image: Image.Image,
    config: MangaTranslatorConfig,
    image_path: Union[str, Path],
    image_format: Optional[str],
    verbose: bool = False,
    bubble_data: Optional[List[Dict[str, Any]]] = None,
    text_free_boxes: Optional[List[List[float]]] = None,
    panels: Optional[List[Tuple[int, int, int, int]]] = None,
) -> Tuple[Image.Image, List[Dict[str, Any]]]:
    """
    Process outside text detection, inpainting, and prepare data for translation.

    This function handles the complete outside text processing pipeline:
    1. Detects text outside speech bubbles using OCR
    2. Inpaints the detected text regions using FluxKontext
    3. Prepares the outside text data for translation API calls

    Args:
        pil_image: The PIL image to process
        config: MangaTranslatorConfig containing all settings
        image_path: Path to the original image file
        image_format: Original image format (PNG, JPEG, etc.)
        processing_scale: The scale factor for image processing
        verbose: Whether to print detailed logging

    Returns:
        Tuple containing:
        - processed_pil_image: The image after outside text inpainting
        - outside_text_data: List of dicts with outside text information for translation
    """
    work = prepare_outside_text_work(
        pil_image,
        config,
        image_path,
        image_format,
        verbose=verbose,
        bubble_data=bubble_data,
        text_free_boxes=text_free_boxes,
        panels=panels,
    )
    if work is None:
        return pil_image, []
    return finish_outside_text_work(work)
