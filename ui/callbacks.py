import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import torch
from PIL import Image

from core.config import MangaTranslatorConfig
from core.text.text_processing import text_layout_control_interactivity
from core.validation import (
    validate_mutually_exclusive_modes,
    validate_zip_file,
)
from utils.exceptions import CancellationError, ValidationError, TranslationError
from utils.model_metadata import (
    FLUX_SDCPP_DIFFUSION_QUANTS,
    flux_sdcpp_text_encoder_quants,
    flux_sdcpp_valid_text_encoder_quant,
    is_anthropic_model_family,
    is_openai_model_family,
    is_opencode_multimodal_model,
)
from utils.path_list import read_image_paths_from_txt

from . import logic, settings_manager, utils
from .cancellation import CancellationManager
from .ui_models import (
    UICleaningSettings,
    UIConfigState,
    UIDetectionSettings,
    UIGeneralSettings,
    UIOutputSettings,
    UIOutsideTextSettings,
    UIRenderingSettings,
    UITranslationLLMSettings,
    UITranslationProviderSettings,
    map_ui_to_backend_config,
)

ERROR_PREFIX = "❌ Error: "
SUCCESS_PREFIX = "✅ "

CANCELLATION_MANAGER: Optional[CancellationManager] = None


def _radio_choices(values):
    return [(value, value) for value in values]


def _flux_backend_choices(method: str):
    if method in ("flux_klein_9b", "flux_klein_4b"):
        return [
            ("sd.cpp", "sdcpp"),
            ("SDNQ", "sdnq"),
            ("Cloudflare Workers AI", "cloudflare"),
            ("Remote Colab (T4 GPU)", "remote_colab"),
        ]
    if method in ("flux_dev", "sd_inpainting"):
        return [
            ("Cloudflare Workers AI", "cloudflare"),
            ("Remote Colab (T4 GPU)", "remote_colab"),
        ]
    if method == "flux_kontext":
        return [
            ("sd.cpp", "sdcpp"),
            ("SDNQ", "sdnq"),
            ("Nunchaku (CUDA)", "nunchaku"),
        ]
    return [
        ("SDNQ", "sdnq"),
        ("Cloudflare Workers AI", "cloudflare"),
        ("Remote Colab (T4 GPU)", "remote_colab"),
    ]


_INPAINTING_METHOD_CHOICES_LOCAL = [
    ("Flux.2 Klein 9B", "flux_klein_9b"),
    ("Flux.2 Klein 4B", "flux_klein_4b"),
    ("Flux.1 Kontext (12B)", "flux_kontext"),
    ("OpenCV", "opencv"),
    ("None (text background)", "none"),
]

_INPAINTING_METHOD_CHOICES_CLOUDFLARE = [
    ("Flux.2 Klein 4B", "flux_klein_4b"),
    ("Flux.2 Klein 9B", "flux_klein_9b"),
    ("Flux.2 Dev", "flux_dev"),
    ("SD 1.5 Inpainting", "sd_inpainting"),
    ("OpenCV", "opencv"),
    ("None (text background)", "none"),
]


def _inpainting_method_choices(backend: str):
    if backend in ("cloudflare", "remote_colab"):
        return _INPAINTING_METHOD_CHOICES_CLOUDFLARE
    return _INPAINTING_METHOD_CHOICES_LOCAL


def _clean_error_message(message: Any) -> str:
    """Normalize error text for display and avoid duplicate prefixes/quotes."""
    try:
        text = str(message).strip()
    except Exception:
        text = ""

    if (len(text) >= 2) and (
        (text[0] == text[-1] == "'") or (text[0] == text[-1] == '"')
    ):
        text = text[1:-1].strip()

    if text.startswith(ERROR_PREFIX):
        return text
    return f"{ERROR_PREFIX}{text}"


def _status_update(message: str) -> gr.update:
    """Create a gr.update() for the status textbox with calculated line count.

    Accounts for both explicit newlines and text wrapping (~120 chars per line).
    """
    line_count = 0
    for line in message.split("\n"):
        # Each segment takes at least 1 line, plus additional lines for wrapping
        line_count += max(1, (len(line) + 119) // 120)
    return gr.update(value=message, lines=line_count)


def _build_ui_state_from_args(args: tuple, is_batch: bool) -> UIConfigState:
    """Build UIConfigState from UI component arguments, handling single vs batch mode differences."""
    (
        confidence,
        conjoined_confidence,
        panel_confidence,
        seg_model_val,
        bubble_detector_model_val,
        conjoined_detection_checkbox_val,
        osb_text_verification_checkbox_val,
        use_panel_sorting_checkbox_val,
        thresholding_value,
        use_otsu_threshold,
        inpaint_colored_bubbles,
        roi_shrink_px,
        provider_selector,
        google_api_key,
        openai_api_key,
        anthropic_api_key,
        xai_api_key,
        meta_api_key,
        deepseek_api_key,
        zai_api_key,
        moonshot_api_key,
        mimo_api_key,
        qwencloud_api_key,
        opencode_api_key,
        opencode_tier,
        openrouter_api_key,
        openai_compatible_url_input,
        openai_compatible_api_key_input,
        config_model_name,
        temperature,
        top_p,
        top_k,
        use_custom_sampling_val,
        max_tokens,
        config_reading_direction,
        config_translation_mode,
        ocr_method_val,
        input_language,
        output_language,
        font_dropdown,
        max_font_size,
        min_font_size,
        line_spacing_mult,
        use_subpixel_rendering,
        font_hinting,
        use_ligatures,
        output_format,
        jpeg_quality,
        png_compression,
        verbose,
        cleaning_only_toggle,
        upscaling_only_val,
        test_mode_toggle,
        enable_web_search_val,
        enable_code_execution_val,
        image_detail_val,
        media_resolution_val,
        media_resolution_bubbles_val,
        media_resolution_context_val,
        reasoning_effort_val,
        effort_val,
        verbosity_val,
        send_full_page_context_val,
        whiteout_conjoined_bubbles_val,
        upscale_method_val,
        bubble_min_side_pixels_val,
        context_image_max_side_pixels_val,
        osb_min_side_pixels_val,
        hyphenate_before_scaling_val,
        detach_trailing_punctuation_val,
        auto_vertical_text_val,
        vertical_line_spacing_mult_val,
        vertical_font_size_mult_val,
        auto_rotate_text_val,
        auto_rotate_confidence_threshold_val,
        hyphen_penalty_val,
        hyphenation_min_word_length_val,
        badness_exponent_val,
        padding_pixels_val,
        supersampling_factor_val,
        outside_text_enabled_val,
        outside_text_seed_val,
        outside_text_inpainting_method_val,
        outside_text_flux_backend_val,
        outside_text_cloudflare_accounts_val,
        outside_text_colab_server_url_val,
        outside_text_flux_low_vram_val,
        outside_text_flux_sdcpp_cache_mode_val,
        outside_text_flux_sdcpp_diffusion_quant_val,
        outside_text_flux_sdcpp_text_encoder_quant_val,
        outside_text_flux_num_inference_steps_val,
        outside_text_flux_luminance_correction_val,
        outside_text_flux_upscale_small_crops_val,
        outside_text_flux_group_regions_val,
        outside_text_flux_residual_diff_threshold_val,
        outside_text_osb_confidence_val,
        outside_text_osb_text_free_only_val,
        outside_text_min_area_ignore_ratio_percent_val,
        outside_text_enable_page_number_filtering_val,
        outside_text_page_filter_margin_threshold_val,
        outside_text_page_filter_min_area_ratio_val,
        outside_text_huggingface_token_val,
        outside_text_osb_font_pack_val,
        outside_text_osb_max_font_size_val,
        outside_text_osb_min_font_size_val,
        outside_text_osb_use_ligatures_val,
        outside_text_osb_outline_width_val,
        outside_text_osb_line_spacing_val,
        outside_text_osb_padding_pixels_val,
        outside_text_osb_auto_vertical_text_val,
        outside_text_osb_vertical_line_spacing_mult_val,
        outside_text_osb_vertical_font_size_mult_val,
        outside_text_osb_use_subpixel_rendering_val,
        outside_text_osb_font_hinting_val,
        outside_text_bbox_expansion_percent_width_val,
        outside_text_bbox_expansion_percent_height_val,
        outside_text_osb_render_expansion_narrow_multiplier_val,
        outside_text_osb_render_expansion_aspect_ratio_threshold_val,
        outside_text_osb_render_expansion_tiny_multiplier_val,
        outside_text_osb_render_expansion_area_threshold_percent_val,
        outside_text_text_box_proximity_ratio_val,
        image_upscale_mode_val,
        image_upscale_factor_val,
        image_upscale_model_val,
        auto_scale_val,
        batch_input_language,
        batch_output_language,
        batch_font_dropdown,
        special_instructions_val,
        batch_special_instructions_val,
        batch_parallel_requests_val,
        batch_parallel_within_pages_val,
        overlap_llm_with_inpaint_val,
        batch_overlap_llm_with_inpaint_val,
        batch_retry_failed_once_val,
        batch_previous_context_image_count_val,
        batch_previous_context_text_count_val,
        manual_json_translation_val,
    ) = args

    final_input_language = batch_input_language if is_batch else input_language
    final_output_language = batch_output_language if is_batch else output_language
    final_font_pack = batch_font_dropdown if is_batch else font_dropdown
    final_special_instructions = (
        batch_special_instructions_val if is_batch else special_instructions_val
    )
    final_overlap_llm_with_inpaint = (
        batch_overlap_llm_with_inpaint_val if is_batch else overlap_llm_with_inpaint_val
    )

    return UIConfigState(
        detection=UIDetectionSettings(
            confidence=confidence,
            conjoined_confidence=conjoined_confidence,
            panel_confidence=panel_confidence,
            seg_model=seg_model_val,
            bubble_detector_model=bubble_detector_model_val,
            conjoined_detection=conjoined_detection_checkbox_val,
            use_osb_text_verification=osb_text_verification_checkbox_val,
            use_panel_sorting=use_panel_sorting_checkbox_val,
        ),
        cleaning=UICleaningSettings(
            thresholding_value=thresholding_value,
            use_otsu_threshold=use_otsu_threshold,
            roi_shrink_px=int(roi_shrink_px),
            inpaint_colored_bubbles=inpaint_colored_bubbles,
        ),
        outside_text=UIOutsideTextSettings(
            enabled=outside_text_enabled_val,
            seed=int(outside_text_seed_val),
            huggingface_token=outside_text_huggingface_token_val,
            inpainting_method=outside_text_inpainting_method_val,
            flux_backend=outside_text_flux_backend_val,
            cloudflare_accounts=(
                "\n".join(__import__("core.services.credentials", fromlist=["get_secure_cloudflare_accounts"]).get_secure_cloudflare_accounts())
                if (not outside_text_cloudflare_accounts_val or "•" in str(outside_text_cloudflare_accounts_val))
                else outside_text_cloudflare_accounts_val
            ),
            colab_server_url=outside_text_colab_server_url_val,
            flux_low_vram=outside_text_flux_low_vram_val,
            flux_sdcpp_cache_mode=outside_text_flux_sdcpp_cache_mode_val,
            flux_sdcpp_diffusion_quant=outside_text_flux_sdcpp_diffusion_quant_val,
            flux_sdcpp_text_encoder_quant=outside_text_flux_sdcpp_text_encoder_quant_val,
            flux_num_inference_steps=int(outside_text_flux_num_inference_steps_val),
            flux_luminance_correction=outside_text_flux_luminance_correction_val,
            flux_upscale_small_crops=outside_text_flux_upscale_small_crops_val,
            flux_group_regions=outside_text_flux_group_regions_val,
            flux_residual_diff_threshold=float(
                outside_text_flux_residual_diff_threshold_val
            ),
            osb_confidence=float(outside_text_osb_confidence_val),
            osb_text_free_only=bool(outside_text_osb_text_free_only_val),
            min_area_ignore_ratio=float(outside_text_min_area_ignore_ratio_percent_val)
            / 100.0,
            enable_page_number_filtering=outside_text_enable_page_number_filtering_val,
            page_filter_margin_threshold=float(
                outside_text_page_filter_margin_threshold_val
            ),
            page_filter_min_area_ratio=float(
                outside_text_page_filter_min_area_ratio_val
            ),
            osb_font_dir=outside_text_osb_font_pack_val,
            osb_max_font_size=int(outside_text_osb_max_font_size_val),
            osb_min_font_size=int(outside_text_osb_min_font_size_val),
            osb_use_ligatures=outside_text_osb_use_ligatures_val,
            osb_outline_width=float(outside_text_osb_outline_width_val),
            osb_line_spacing=float(outside_text_osb_line_spacing_val),
            osb_padding_pixels=float(outside_text_osb_padding_pixels_val),
            osb_auto_vertical_text=outside_text_osb_auto_vertical_text_val,
            osb_vertical_line_spacing_mult=float(
                outside_text_osb_vertical_line_spacing_mult_val
            ),
            osb_vertical_font_size_mult=float(
                outside_text_osb_vertical_font_size_mult_val
            ),
            osb_use_subpixel_rendering=outside_text_osb_use_subpixel_rendering_val,
            osb_font_hinting=outside_text_osb_font_hinting_val,
            bbox_expansion_percent_width=float(
                outside_text_bbox_expansion_percent_width_val
            ),
            bbox_expansion_percent_height=float(
                outside_text_bbox_expansion_percent_height_val
            ),
            osb_render_expansion_narrow_multiplier=float(
                outside_text_osb_render_expansion_narrow_multiplier_val
            ),
            osb_render_expansion_tiny_multiplier=float(
                outside_text_osb_render_expansion_tiny_multiplier_val
            ),
            osb_render_expansion_aspect_ratio_threshold=float(
                outside_text_osb_render_expansion_aspect_ratio_threshold_val
            ),
            osb_render_expansion_area_ratio_threshold=float(
                outside_text_osb_render_expansion_area_threshold_percent_val
            )
            / 100.0,
            text_box_proximity_ratio=float(outside_text_text_box_proximity_ratio_val),
        ),
        provider_settings=UITranslationProviderSettings(
            provider=provider_selector,
            google_api_key=(
                "\n".join(__import__("core.services.credentials", fromlist=["get_secure_api_keys"]).get_secure_api_keys("Google"))
                if (not google_api_key or "•" in str(google_api_key))
                else google_api_key
            ),
            openai_api_key=openai_api_key,
            anthropic_api_key=anthropic_api_key,
            xai_api_key=xai_api_key,
            meta_api_key=meta_api_key,
            deepseek_api_key=deepseek_api_key,
            zai_api_key=zai_api_key,
            moonshot_api_key=moonshot_api_key,
            mimo_api_key=mimo_api_key,
            qwencloud_api_key=qwencloud_api_key,
            opencode_api_key=opencode_api_key,
            opencode_tier=("go" if "go" in str(opencode_tier).lower() else "zen"),
            openrouter_api_key=openrouter_api_key,
            openai_compatible_url=openai_compatible_url_input,
            openai_compatible_api_key=openai_compatible_api_key_input,
            omniroute_url=openai_compatible_url_input,
            omniroute_api_key=openai_compatible_api_key_input,
            ninerouter_url=openai_compatible_url_input,
            ninerouter_api_key=openai_compatible_api_key_input,
            manual_json_translation=manual_json_translation_val,
        ),
        llm_settings=UITranslationLLMSettings(
            model_name=config_model_name,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            reading_direction=config_reading_direction,
            translation_mode=config_translation_mode,
            ocr_method=ocr_method_val,
            send_full_page_context=send_full_page_context_val,
            whiteout_conjoined_bubbles=whiteout_conjoined_bubbles_val,
            upscale_method=upscale_method_val,
            bubble_min_side_pixels=bubble_min_side_pixels_val,
            context_image_max_side_pixels=context_image_max_side_pixels_val,
            osb_min_side_pixels=osb_min_side_pixels_val,
            special_instructions=final_special_instructions,
        ),
        rendering=UIRenderingSettings(
            max_font_size=max_font_size,
            min_font_size=min_font_size,
            line_spacing_mult=line_spacing_mult,
            use_subpixel_rendering=use_subpixel_rendering,
            font_hinting=font_hinting,
            use_ligatures=use_ligatures,
            hyphenate_before_scaling=hyphenate_before_scaling_val,
            detach_trailing_punctuation=detach_trailing_punctuation_val,
            auto_vertical_text=auto_vertical_text_val,
            vertical_line_spacing_mult=float(vertical_line_spacing_mult_val),
            vertical_font_size_mult=float(vertical_font_size_mult_val),
            auto_rotate_text=bool(auto_rotate_text_val),
            auto_rotate_confidence_threshold=float(auto_rotate_confidence_threshold_val),
            hyphen_penalty=hyphen_penalty_val,
            hyphenation_min_word_length=hyphenation_min_word_length_val,
            badness_exponent=badness_exponent_val,
            padding_pixels=padding_pixels_val,
            supersampling_factor=int(supersampling_factor_val),
        ),
        output=UIOutputSettings(
            output_format=output_format,
            jpeg_quality=jpeg_quality,
            png_compression=png_compression,
            image_upscale_mode=image_upscale_mode_val,
            image_upscale_factor=image_upscale_factor_val,
            image_upscale_model=image_upscale_model_val,
        ),
        general=UIGeneralSettings(
            verbose=verbose,
            cleaning_only=cleaning_only_toggle,
            upscaling_only=upscaling_only_val,
            test_mode=test_mode_toggle,
            enable_web_search=enable_web_search_val,
            enable_code_execution=enable_code_execution_val,
            use_custom_sampling=use_custom_sampling_val,
            image_detail=image_detail_val,
            media_resolution=media_resolution_val,
            media_resolution_bubbles=media_resolution_bubbles_val,
            media_resolution_context=media_resolution_context_val,
            reasoning_effort=reasoning_effort_val,
            effort=effort_val,
            verbosity=verbosity_val,
            auto_scale=auto_scale_val,
            overlap_llm_with_inpaint=bool(final_overlap_llm_with_inpaint),
        ),
        input_language=final_input_language,
        output_language=final_output_language,
        font_pack=final_font_pack,
        batch_input_language=batch_input_language,
        batch_output_language=batch_output_language,
        batch_font_pack=batch_font_dropdown,
        batch_parallel_requests=int(batch_parallel_requests_val),
        batch_parallel_within_pages=bool(batch_parallel_within_pages_val),
        batch_retry_failed_once=bool(batch_retry_failed_once_val),
        batch_previous_context_image_count=int(batch_previous_context_image_count_val),
        batch_previous_context_text_count=int(batch_previous_context_text_count_val),
    )


def _validate_ui_state(ui_state: UIConfigState) -> None:
    """Validate UI state including modes and API keys. Raises gr.Error on validation failure."""
    try:
        validate_mutually_exclusive_modes(
            ui_state.general.cleaning_only,
            ui_state.general.upscaling_only,
            ui_state.general.test_mode,
        )
    except ValidationError as e:
        raise gr.Error(f"{ERROR_PREFIX}{str(e)}")

    # Skip API key validation if in cleaning-only, upscaling-only, or test mode
    if (
        not ui_state.general.cleaning_only
        and not ui_state.general.upscaling_only
        and not ui_state.general.test_mode
    ):
        api_key_to_validate = ""
        provider_selector = ui_state.provider_settings.provider
        if provider_selector == "Google":
            api_key_to_validate = ui_state.provider_settings.google_api_key
        elif provider_selector == "OpenAI":
            api_key_to_validate = ui_state.provider_settings.openai_api_key
        elif provider_selector == "Anthropic":
            api_key_to_validate = ui_state.provider_settings.anthropic_api_key
        elif provider_selector == "SpaceXAI":
            api_key_to_validate = ui_state.provider_settings.xai_api_key
        elif provider_selector == "Meta Model":
            api_key_to_validate = ui_state.provider_settings.meta_api_key
        elif provider_selector == "DeepSeek":
            api_key_to_validate = ui_state.provider_settings.deepseek_api_key
        elif provider_selector == "Z.ai":
            api_key_to_validate = ui_state.provider_settings.zai_api_key
        elif provider_selector == "Moonshot AI":
            api_key_to_validate = ui_state.provider_settings.moonshot_api_key
        elif provider_selector == "Xiaomi MiMo":
            api_key_to_validate = ui_state.provider_settings.mimo_api_key
        elif provider_selector == "QwenCloud":
            api_key_to_validate = ui_state.provider_settings.qwencloud_api_key
        elif provider_selector == "OpenCode":
            api_key_to_validate = ui_state.provider_settings.opencode_api_key
        elif provider_selector == "OpenRouter":
            api_key_to_validate = ui_state.provider_settings.openrouter_api_key
        elif provider_selector == "OpenAI-Compatible":
            api_key_to_validate = ui_state.provider_settings.openai_compatible_api_key
        elif provider_selector == "OmniRoute":
            api_key_to_validate = ui_state.provider_settings.omniroute_api_key
        elif provider_selector == "9router":
            api_key_to_validate = ui_state.provider_settings.ninerouter_api_key

        api_valid, api_msg = utils.validate_api_key(
            api_key_to_validate, provider_selector
        )
        if not api_valid and not (
            provider_selector in ("OpenAI-Compatible", "OmniRoute", "9router") and not api_key_to_validate
        ):
            raise gr.Error(f"{ERROR_PREFIX}{api_msg}")

    if (
        ui_state.provider_settings.provider in ("OpenAI-Compatible", "OmniRoute", "9router")
        and not ui_state.general.cleaning_only
        and not ui_state.general.upscaling_only
        and not ui_state.general.test_mode
    ):
        p_name = ui_state.provider_settings.provider
        target_url = (
            ui_state.provider_settings.omniroute_url
            if p_name == "OmniRoute"
            else (
                ui_state.provider_settings.ninerouter_url
                if p_name == "9router"
                else ui_state.provider_settings.openai_compatible_url
            )
        )
        if not target_url:
            raise gr.Error(f"{ERROR_PREFIX}{p_name} URL is required.")
        if not target_url.startswith(("http://", "https://")):
            raise gr.Error(
                f"{ERROR_PREFIX}Invalid {p_name} URL format. Must start with http:// or https://",
            )

    if ui_state.outside_text.enabled:
        hf_valid, hf_msg = utils.validate_huggingface_token(
            ui_state.outside_text.huggingface_token
        )
        if not hf_valid:
            raise gr.Error(f"{ERROR_PREFIX}{hf_msg}")


def _format_single_success_message(
    result_image: Image.Image,
    backend_config: MangaTranslatorConfig,
    font_dir_path: Path,
    save_path: Path,
    processing_time: float,
) -> str:
    """Formats the success message for single image translation."""
    width, height = result_image.size
    provider = backend_config.translation.provider
    model_name = backend_config.translation.model_name
    thinking_status_str = utils.format_thinking_status(
        provider,
        model_name,
        backend_config.translation.reasoning_effort,
    )
    temp_val = backend_config.translation.temperature
    top_p_val = backend_config.translation.top_p
    top_k_val = backend_config.translation.top_k

    llm_params_str = f"• LLM Params: Temp={temp_val:.2f}, Top-P={top_p_val:.2f}"
    param_notes = ""
    if provider == "Google":
        llm_params_str += f", Top-K={top_k_val}"
    elif provider == "Anthropic":
        llm_params_str += f", Top-K={top_k_val}"
        param_notes = " (Temp clamped <= 1.0)"
    elif provider == "OpenAI":
        param_notes = " (Top-K N/A)"
    elif provider == "OpenRouter":
        if is_openai_model_family(model_name) or is_anthropic_model_family(model_name):
            param_notes = " (Temp clamped <= 1.0, Top-K N/A)"
        else:
            llm_params_str += f", Top-K={top_k_val}"
    elif provider == "OpenAI-Compatible":
        llm_params_str += f", Top-K={top_k_val}"
    llm_params_str += param_notes

    if backend_config.cleaning_only:
        processing_mode_str = "Cleaning Only"
    elif backend_config.upscaling_only:
        processing_mode_str = "Upscaling Only"
    else:
        processing_mode_str = "Translation"

    if backend_config.test_mode:
        model_display = "<Test Mode>"
    elif backend_config.cleaning_only:
        model_display = "<Cleaning-only Mode>"
    elif backend_config.upscaling_only:
        model_display = "<Upscaling-only Mode>"
    else:
        model_display = f"{model_name}{thinking_status_str}"

    msg_parts = [
        f"{SUCCESS_PREFIX}{processing_mode_str} completed!\n",
        f"• Image Size: {width}x{height} pixels\n",
        f"• Outside Text Detection: {'Enabled' if backend_config.outside_text.enabled else 'Disabled'}\n",
    ]

    if backend_config.outside_text.enabled and backend_config.outside_text.osb_font_dir:
        osb_font_pack_name = Path(backend_config.outside_text.osb_font_dir).name
        msg_parts.append(f"• OSB Font Pack: {osb_font_pack_name}\n")

    if not backend_config.cleaning_only and not backend_config.upscaling_only:
        msg_parts.append(
            f"• Full-Page Context: {'On' if backend_config.translation.send_full_page_context else 'Off'}\n"
        )
        previous_context_count = backend_config.translation.previous_context_image_count
        if previous_context_count:
            msg_parts.append(f"• Previous Context Images: {previous_context_count}\n")

    msg_parts.append(
        f"• Upscale Method: {backend_config.translation.upscale_method.title()}\n"
    )

    msg_parts.extend(
        [
            f"• Provider: {provider}\n",
            f"• Model: {model_display}\n",
            f"• Source Language: {backend_config.translation.input_language}\n",
            f"• Target Language: {backend_config.translation.output_language}\n",
            f"• Reading Direction: {backend_config.translation.reading_direction.upper()}\n",
            f"• Translation Mode: {backend_config.translation.translation_mode}\n",
        ]
    )

    if not backend_config.cleaning_only and not backend_config.upscaling_only:
        msg_parts.append(f"{llm_params_str}\n")

    msg_parts.append(f"• Font Pack: {font_dir_path.name}\n")

    if (
        getattr(backend_config, "preprocessing", None)
        and backend_config.preprocessing.enabled
    ):
        msg_parts.append(
            f"• Initial Image Upscaling: {backend_config.preprocessing.factor}x\n"
        )

    if backend_config.output.upscale_final_image:
        msg_parts.append(
            f"• Final Image Upscaling: {backend_config.output.image_upscale_factor}x\n"
        )

    if backend_config.output.output_format == "png":
        msg_parts.append("• Output Format: png\n")
        msg_parts.append(
            f"• PNG Compression: {backend_config.output.png_compression}\n"
        )
    elif backend_config.output.output_format == "jpeg":
        msg_parts.append("• Output Format: jpeg\n")
        msg_parts.append(f"• JPEG Quality: {backend_config.output.jpeg_quality}\n")
    else:
        try:
            ext = save_path.suffix.lower()
        except Exception:
            ext = ""
        if ext in {".jpg", ".jpeg"}:
            msg_parts.append("• Output Format: auto → jpeg\n")
            msg_parts.append(f"• JPEG Quality: {backend_config.output.jpeg_quality}\n")
        elif ext == ".png":
            msg_parts.append("• Output Format: auto → png\n")
            msg_parts.append(
                f"• PNG Compression: {backend_config.output.png_compression}\n"
            )
        elif ext == ".webp":
            msg_parts.append("• Output Format: auto → webp\n")
        else:
            msg_parts.append("• Output Format: auto\n")

    msg_parts.extend(
        [
            f"\n• Processing Time: {processing_time:.2f} seconds\n",
            f"• Saved to: {save_path}",
        ]
    )
    return "".join(msg_parts)


def _format_batch_success_message(
    results: Dict[str, Any],
    backend_config: MangaTranslatorConfig,
    font_dir_path: Path,
) -> str:
    """Formats the success message for batch processing."""
    success_count = results["success_count"]
    error_count = results["error_count"]
    total_images = success_count + error_count
    processing_time = results["processing_time"]
    output_path = results["output_path"]
    seconds_per_image = processing_time / success_count if success_count > 0 else 0

    provider = backend_config.translation.provider
    model_name = backend_config.translation.model_name
    thinking_status_str = utils.format_thinking_status(
        provider,
        model_name,
        backend_config.translation.reasoning_effort,
    )
    temp_val = backend_config.translation.temperature
    top_p_val = backend_config.translation.top_p
    top_k_val = backend_config.translation.top_k

    llm_params_str = f"• LLM Params: Temp={temp_val:.2f}, Top-P={top_p_val:.2f}"
    param_notes = ""
    if provider == "Google":
        llm_params_str += f", Top-K={top_k_val}"
    elif provider == "Anthropic":
        llm_params_str += f", Top-K={top_k_val}"
        param_notes = " (Temp clamped <= 1.0)"
    elif provider == "OpenAI":
        param_notes = " (Top-K N/A)"
    elif provider == "OpenRouter":
        if is_openai_model_family(model_name) or is_anthropic_model_family(model_name):
            param_notes = " (Temp clamped <= 1.0, Top-K N/A)"
        else:
            llm_params_str += f", Top-K={top_k_val}"
    elif provider == "OpenAI-Compatible":
        param_notes = " (Top-K N/A)"
    llm_params_str += param_notes

    if not backend_config.cleaning_only and not backend_config.upscaling_only:
        llm_params_str = (
            f"• Full-Page Context: {'On' if backend_config.translation.send_full_page_context else 'Off'}\n"
            + llm_params_str
        )

    failure_details = ""
    if error_count > 0:
        failed_pages = results.get("errors", {})
        failure_parts = [f"• Warning: {error_count} image(s) failed to process.\n"]
        if failed_pages:
            failure_parts.append("\nFailed pages:\n")
            failure_parts.extend(
                f"• {filename}: {error_msg}\n"
                for filename, error_msg in failed_pages.items()
            )
        failure_details = "".join(failure_parts)

    if backend_config.cleaning_only:
        processing_mode_str = "Cleaning Only"
    elif backend_config.upscaling_only:
        processing_mode_str = "Upscaling Only"
    else:
        processing_mode_str = "Translation"

    # Determine model display based on mode
    if backend_config.test_mode:
        model_display = "<Test Mode>"
    elif backend_config.cleaning_only:
        model_display = "<Cleaning-only Mode>"
    elif backend_config.upscaling_only:
        model_display = "<Upscaling-only Mode>"
    else:
        model_display = f"{model_name}{thinking_status_str}"

    msg_parts = [
        f"{SUCCESS_PREFIX}Batch {processing_mode_str.lower()} completed!\n",
        f"• Outside Text Detection: {'Enabled' if backend_config.outside_text.enabled else 'Disabled'}\n",
    ]

    if backend_config.outside_text.enabled and backend_config.outside_text.osb_font_dir:
        osb_font_pack_name = Path(backend_config.outside_text.osb_font_dir).name
        msg_parts.append(f"• OSB Font Pack: {osb_font_pack_name}\n")

    if not backend_config.cleaning_only and not backend_config.upscaling_only:
        msg_parts.append(
            f"• Full-Page Context: {'On' if backend_config.translation.send_full_page_context else 'Off'}\n"
        )
        previous_context_count = backend_config.translation.previous_context_image_count
        if previous_context_count:
            msg_parts.append(f"• Previous Context Images: {previous_context_count}\n")

    msg_parts.append(
        f"• Upscale Method: {backend_config.translation.upscale_method.title()}\n"
    )

    msg_parts.extend(
        [
            f"• Provider: {provider}\n",
            f"• Model: {model_display}\n",
            f"• Source Language: {backend_config.translation.input_language}\n",
            f"• Target Language: {backend_config.translation.output_language}\n",
            f"• Reading Direction: {backend_config.translation.reading_direction.upper()}\n",
            f"• Translation Mode: {backend_config.translation.translation_mode}\n",
        ]
    )

    if not backend_config.cleaning_only and not backend_config.upscaling_only:
        msg_parts.append(f"{llm_params_str}\n")

    msg_parts.append(f"• Font Pack: {font_dir_path.name}\n")

    if (
        getattr(backend_config, "preprocessing", None)
        and backend_config.preprocessing.enabled
    ):
        msg_parts.append(
            f"• Initial Image Upscaling: {backend_config.preprocessing.factor}x\n"
        )

    if backend_config.output.upscale_final_image:
        msg_parts.append(
            f"• Final Image Upscaling: {backend_config.output.image_upscale_factor}x\n"
        )

    if backend_config.output.output_format == "png":
        msg_parts.append("• Output Format: png\n")
        msg_parts.append(
            f"• PNG Compression: {backend_config.output.png_compression}\n"
        )
    elif backend_config.output.output_format == "jpeg":
        msg_parts.append("• Output Format: jpeg\n")
        msg_parts.append(f"• JPEG Quality: {backend_config.output.jpeg_quality}\n")
    else:
        msg_parts.append("• Output Format: auto\n")
        try:
            out_dir = Path(output_path)
            if out_dir.exists() and out_dir.is_dir():
                exts = {p.suffix.lower() for p in out_dir.glob("*.*") if p.is_file()}
                only_jpeg = exts and all(e in {".jpg", ".jpeg"} for e in exts)
                only_png = exts == {".png"}
                if only_png:
                    msg_parts.append(
                        f"• PNG Compression: {backend_config.output.png_compression}\n"
                    )
                elif only_jpeg:
                    msg_parts.append(
                        f"• JPEG Quality: {backend_config.output.jpeg_quality}\n"
                    )
                # If mixed or unknown, omit specific settings to avoid showing both
        except Exception:
            pass

    msg_parts.extend(
        [
            f"• Successful Translations: {success_count}/{total_images}\n",
            failure_details,
            f"\n• Total Processing Time: {processing_time:.2f} seconds ({seconds_per_image:.2f} seconds/image)\n",
            f"• Saved to: {output_path}",
        ]
    )
    failed_paths_file = results.get("failed_paths_file")
    if failed_paths_file:
        msg_parts.append(f"\n• Failed paths list: {failed_paths_file}")
    retry_attempted = results.get("retry_attempted_count")
    if retry_attempted:
        retry_ok = results.get("retry_success_count", 0)
        retry_fail = results.get("retry_failed_count", 0)
        msg_parts.append(
            f"\n• Retry pass: {retry_ok} recovered, {retry_fail} still failed "
            f"(of {retry_attempted} attempted)"
        )
    return "".join(msg_parts)


def handle_translate_click(
    *args: Any,
    models_dir: Path,
    fonts_base_dir: Path,
    target_device: Optional[torch.device],
) -> Tuple[Optional[Image.Image], str, Any]:
    """Callback for the 'Translate' button click. Uses dataclasses for config."""
    manual_txt_val = args[-1]
    ui_args = args[:-1]
    input_image_path = ui_args[0]
    start_time = time.time()
    global CANCELLATION_MANAGER
    CANCELLATION_MANAGER = CancellationManager()
    try:
        img_valid, img_msg = utils.validate_image(input_image_path)
        if not img_valid:
            raise gr.Error(f"{ERROR_PREFIX}{img_msg}")

        ui_state = _build_ui_state_from_args(ui_args[1:], is_batch=False)

        _validate_ui_state(ui_state)

        # Auto-save special instructions to settings
        try:
            settings = settings_manager.get_saved_settings()
            settings["special_instructions"] = ui_state.llm_settings.special_instructions
            settings_manager.save_config(settings)
        except Exception:
            pass

        backend_config = map_ui_to_backend_config(
            ui_state=ui_state,
            fonts_base_dir=fonts_base_dir,
            target_device=target_device,
            is_batch=False,
        )
        
        # Check mismatch for manual txt
        if manual_txt_val and isinstance(manual_txt_val, str) and manual_txt_val.strip() and backend_config.translation.manual_json_translation:
            _json_base = Path(input_image_path)
            _base_name = _json_base.name.split('_translated_')[0] if '_translated_' in _json_base.name else _json_base.stem
            
            import hashlib
            import tempfile
            import json
            import re
            
            _path_hash = hashlib.md5(Path(input_image_path).name.encode('utf-8')).hexdigest()
            _meta_path = Path(tempfile.gettempdir()) / f"manga_tr_meta_{_path_hash}.json"
            
            mismatch = False
            if not _meta_path.exists():
                mismatch = True
            else:
                try:
                    with open(_meta_path, "r", encoding="utf-8") as f:
                        _meta_data = json.load(f)
                    _meta_map = {int(e["index"]): e for e in _meta_data}
                    
                    # Check if the indexes belong to this image's metadata
                    for _line in manual_txt_val.splitlines():
                        _m = re.match(r"^(?:\[(?:Bubble|OSB)\]\s*)?(\d+):\s*(?:\[(?:Bubble|OSB)\]\s*)?(.*?)(?:\s*\|\|\s*(.*))?$", _line.strip())
                        if _m:
                            _idx = int(_m.group(1))
                            if _idx not in _meta_map:
                                mismatch = True
                                break
                except Exception:
                    mismatch = True
            
            if mismatch:
                # Force Lần 1 generation
                backend_config.translation.manual_translation_text = ""
            else:
                # Set the translation text in config in-memory
                backend_config.translation.manual_translation_text = manual_txt_val.strip()
                # Write updated box content to backup folder
                try:
                    backup_dir = Path(r"C:\Users\Hung\Documents\Selenium\MangaTranslator\Translation_backup")
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    backup_path = backup_dir / f"{_base_name}_translation.txt"
                    with open(backup_path, "w", encoding="utf-8") as f:
                        f.write(manual_txt_val.strip())
                except Exception:
                    pass
        else:
            backend_config.translation.manual_translation_text = ""
        selected_font_pack_name = ui_state.font_pack

        result_image, save_path = logic.translate_manga_logic(
            image=input_image_path,
            config=backend_config,
            selected_font_pack_name=selected_font_pack_name,
            models_dir=models_dir,
            fonts_base_dir=fonts_base_dir,
            cancellation_manager=CANCELLATION_MANAGER,
        )

        # Re-get font path for success message (logic already validated it)
        font_dir_path = (
            fonts_base_dir / selected_font_pack_name
            if selected_font_pack_name
            else Path(".")
        )

        processing_time = time.time() - start_time
        status_msg = _format_single_success_message(
            result_image,
            backend_config,
            font_dir_path,
            save_path,
            processing_time,
        )

        # Clean up the temp working txt file in output/
        return result_image.copy(), _status_update(status_msg), gr.update()

    except gr.Error as e:
        cleaned = _clean_error_message(e)
        gr.Error(cleaned)
        return None, _status_update(cleaned), gr.update()
    except CancellationError:
        return None, _status_update("Translation cancelled by user."), gr.update()
    except (ValidationError, FileNotFoundError, ValueError, logic.LogicError, TranslationError) as e:
        cleaned = _clean_error_message(e)
        gr.Error(cleaned)
        
        # Check if error is related to TranslationError of manual txt
        txt_content = gr.update()
        if "--- CONTENT START ---" in cleaned:
            import re
            m = re.search(r"--- CONTENT START ---\n(.*?)\n--- CONTENT END ---", cleaned, re.DOTALL)
            if m:
                _txt_content_str = m.group(1).strip()
                txt_content = gr.update(value=_txt_content_str)
                # Lần 1: Không lưu vào Translation_backup, chỉ nạp vào Textbox để sửa
                # Simplify status message to be clean and user friendly
                cleaned = "Template generated. Edit text after '||' then click Translate again to render."
                
        return None, _status_update(cleaned), txt_content
    except Exception as e:
        import traceback

        traceback.print_exc()
        cleaned = _clean_error_message(f"An unexpected error occurred: {str(e)}")
        gr.Error(cleaned)
        return None, _status_update(cleaned), gr.update()


def handle_zip_or_failed_paths_upload(
    file_path: Optional[str],
) -> Tuple[Any, Any]:
    """Expand a failed-paths .txt into the images uploader; leave ZIPs unchanged."""
    if not file_path:
        return gr.update(), gr.update()

    path = Path(file_path)
    if path.suffix.lower() != ".txt":
        # ZIP or other: leave both components as Gradio already set them.
        return gr.update(), gr.update()

    images = [str(p) for p in read_image_paths_from_txt(path)]
    if not images:
        gr.Warning("No existing image paths found in the path list file.")
        return gr.update(), gr.update(value=None)

    # Populate image preview; clear the zip/txt slot so batch uses the file list.
    return gr.update(value=images), gr.update(value=None)


def handle_batch_click(
    *args: Any,
    models_dir: Path,
    fonts_base_dir: Path,
    target_device: Optional[torch.device],
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[List[str], str]:
    """Callback for the 'Start Batch Translating' button click. Uses dataclasses."""
    input_files = args[0]
    input_zip = args[1] if len(args) > 1 else None
    progress(0, desc="Starting batch process...")
    global CANCELLATION_MANAGER
    CANCELLATION_MANAGER = CancellationManager()
    try:
        zip_file_path = None
        if input_zip:
            zip_path = Path(input_zip)
            if zip_path.suffix.lower() == ".txt":
                # Defensive: expand path list if change-handler did not already
                expanded = [str(p) for p in read_image_paths_from_txt(zip_path)]
                if not expanded:
                    raise gr.Error(
                        f"{ERROR_PREFIX}No existing image paths found in the path list file."
                    )
                if input_files and not isinstance(input_files, list):
                    raise gr.Error(
                        f"{ERROR_PREFIX}Invalid input format. Expected a list of files."
                    )
                input_files = list(input_files or []) + expanded
            else:
                try:
                    zip_file_path = validate_zip_file(input_zip)
                except (ValidationError, FileNotFoundError) as e:
                    raise gr.Error(f"{ERROR_PREFIX}{str(e)}")

        if input_files:
            if not isinstance(input_files, list):
                raise gr.Error(
                    f"{ERROR_PREFIX}Invalid input format. Expected a list of files."
                )

        if not zip_file_path and not input_files:
            raise gr.Error(
                f"{ERROR_PREFIX}Please upload images, a folder, a ZIP archive, "
                "or a failed-paths .txt file."
            )

        if zip_file_path and input_files:
            input_to_process = {
                "zip": str(zip_file_path),
                "files": input_files,
            }
        elif zip_file_path:
            input_to_process = str(zip_file_path)
        else:
            input_to_process = input_files

        batch_manual_txt_val = args[-1]
        ui_args = args[:-1]
        ui_state = _build_ui_state_from_args(ui_args[2:], is_batch=True)

        _validate_ui_state(ui_state)

        # Auto-save batch special instructions to settings
        try:
            settings = settings_manager.get_saved_settings()
            settings["batch_special_instructions"] = ui_state.llm_settings.special_instructions
            settings_manager.save_config(settings)
        except Exception:
            pass

        backend_config = map_ui_to_backend_config(
            ui_state=ui_state,
            fonts_base_dir=fonts_base_dir,
            target_device=target_device,
            is_batch=True,
        )
        selected_font_pack_name = ui_state.batch_font_pack

        parsed_batch_translations: Optional[Dict[str, str]] = None
        if batch_manual_txt_val and isinstance(batch_manual_txt_val, str) and batch_manual_txt_val.strip() and backend_config.translation.manual_json_translation:
            parsed_batch_translations = {}
            current_filename = None
            current_lines = []
            for line in batch_manual_txt_val.splitlines():
                line_str = line.strip()
                m = re.match(r"^(.+?\.(?:png|jpg|jpeg|webp)):$", line_str, re.IGNORECASE)
                if m:
                    if current_filename and current_lines:
                        parsed_batch_translations[current_filename] = "\n".join(current_lines).strip()
                    current_filename = Path(m.group(1).strip()).name
                    current_lines = []
                else:
                    if current_filename is not None:
                        current_lines.append(line)
            if current_filename and current_lines:
                parsed_batch_translations[current_filename] = "\n".join(current_lines).strip()

            if parsed_batch_translations:
                try:
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    backup_dir = Path(r"C:\Users\Hung\Documents\Selenium\MangaTranslator\Translation_backup")
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    backup_path = backup_dir / f"batch_translation_{timestamp}.txt"
                    with open(backup_path, "w", encoding="utf-8") as f:
                        f.write(batch_manual_txt_val.strip())
                except Exception:
                    pass

        import queue
        import threading

        template_queue = queue.Queue()
        progress_queue = queue.Queue()

        def _on_template_ready(fname: str, template_str: str):
            template_queue.put((fname, template_str))

        def _on_batch_progress(val: float, desc: str = "Processing..."):
            progress_queue.put((val, desc))

        batch_thread_exception = []
        results = {}

        def _run_batch_worker():
            try:
                res = logic.process_batch_logic(
                    input_dir_or_files=input_to_process,
                    config=backend_config,
                    selected_font_pack_name=selected_font_pack_name,
                    models_dir=models_dir,
                    fonts_base_dir=fonts_base_dir,
                    gradio_progress=_on_batch_progress,
                    cancellation_manager=CANCELLATION_MANAGER,
                    batch_manual_translations=parsed_batch_translations,
                    on_template_ready=_on_template_ready,
                )
                results.update(res)
            except Exception as ex:
                batch_thread_exception.append(ex)

        t = threading.Thread(target=_run_batch_worker)
        t.start()

        collected_blocks = []
        seen_filenames = set()
        last_progress_text = None

        while t.is_alive() or not template_queue.empty() or not progress_queue.empty():
            got_progress = False
            while not progress_queue.empty():
                try:
                    pval, pdesc = progress_queue.get_nowait()
                    try:
                        progress(pval, desc=pdesc)
                    except Exception:
                        pass
                    last_progress_text = f"{pdesc} ({int(pval * 100)}%)"
                    got_progress = True
                except queue.Empty:
                    break

            got_new = False
            while not template_queue.empty():
                try:
                    fname, template_str = template_queue.get_nowait()
                    if fname not in seen_filenames:
                        seen_filenames.add(fname)
                        collected_blocks.append(f"{fname}:\n\n{template_str}")
                        got_new = True
                except queue.Empty:
                    break
            
            if got_new and backend_config.translation.manual_json_translation and not parsed_batch_translations:
                status_txt = last_progress_text or "Generated translation templates for batch images..."
                yield [], _status_update(status_txt), gr.update(value="\n\n".join(collected_blocks))
            elif got_progress and last_progress_text:
                yield gr.update(), _status_update(last_progress_text), gr.update()
            
            time.sleep(0.15)

        t.join()

        if batch_thread_exception:
            raise batch_thread_exception[0]

        output_path = results.get("output_path")
        gallery_images = []
        if output_path and output_path.exists():
            # Use rglob to recursively find all images when structure is preserved
            processed_files = list(output_path.rglob("*.*"))
            image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
            processed_files = [
                f
                for f in processed_files
                if f.is_file() and f.suffix.lower() in image_extensions
            ]
            processed_files.sort(  # Sort naturally (e.g., page_1, page_2, page_10)
                key=lambda x: tuple(
                    int(part) if part.isdigit() else part
                    for part in re.split(r"(\d+)", x.stem)
                )
            )
            gallery_images = [str(file_path) for file_path in processed_files]

        # Collect any remaining Lần 1 manual templates if present
        batch_txt_update = gr.update()
        if backend_config.translation.manual_json_translation and not parsed_batch_translations:
            # Lần 1: Chỉ trích xuất bản dịch mẫu, xóa thư mục output để tránh sinh folder thừa
            if output_path and output_path.exists():
                try:
                    import shutil
                    shutil.rmtree(output_path, ignore_errors=True)
                except Exception:
                    pass
            gallery_images = []

            errors_dict = results.get("errors", {})
            for err_key, err_val in errors_dict.items():
                if "--- CONTENT START ---" in str(err_val):
                    m_str = re.search(r"--- CONTENT START ---\n(.*?)\n--- CONTENT END ---", str(err_val), re.DOTALL)
                    if m_str:
                        template_str = m_str.group(1).strip()
                        filename = Path(err_key).name
                        if filename not in seen_filenames:
                            seen_filenames.add(filename)
                            collected_blocks.append(f"{filename}:\n\n{template_str}")
            
            if collected_blocks:
                full_batch_txt = "\n\n".join(collected_blocks)
                batch_txt_update = gr.update(value=full_batch_txt)
                # Lần 1: Không lưu vào Translation_backup, chỉ nạp vào Textbox để người dùng xem và chỉnh sửa

        # Re-get font path for success message (logic already validated it)
        font_dir_path = (
            fonts_base_dir / selected_font_pack_name
            if selected_font_pack_name
            else Path(".")
        )

        status_msg = _format_batch_success_message(
            results, backend_config, font_dir_path
        )
        if backend_config.translation.manual_json_translation and not parsed_batch_translations and collected_blocks:
            status_msg = "Batch translation templates generated. Edit content in 'Batch Manual Translation Box' then click Start Batch Translating again to render."
            
        progress(1.0, desc="Batch complete!")
        yield gallery_images, _status_update(status_msg), batch_txt_update

    except gr.Error as e:
        progress(1.0, desc="Error occurred")
        cleaned = _clean_error_message(e)
        gr.Error(cleaned)
        return None, _status_update(cleaned), gr.update()
    except CancellationError:
        progress(1.0, desc="Cancelled")
        return None, _status_update("Batch process cancelled by user."), gr.update()
    except (ValidationError, FileNotFoundError, ValueError, logic.LogicError, TranslationError) as e:
        progress(1.0, desc="Error occurred")
        cleaned = _clean_error_message(e)
        gr.Error(cleaned)
        return None, _status_update(cleaned), gr.update()
    except Exception as e:
        progress(1.0, desc="Error occurred")
        import traceback

        traceback.print_exc()
        cleaned = _clean_error_message(
            f"An unexpected error occurred during batch processing: {str(e)}"
        )
        gr.Error(cleaned)
        return None, _status_update(cleaned), gr.update()


def handle_save_config_click(*args: Any) -> str:
    """Callback for the 'Save Config' button. Uses dataclasses."""
    (
        conf,
        conjoined_conf,
        panel_conf,
        seg_model,
        bubble_detector_model,
        conjoined_detection,
        osb_text_verification,
        use_panel_sorting,
        rd,
        thresholding_val,
        otsu,
        inpaint_colored_bubbles,
        roi_shrink_px,
        prov,
        gem_key,
        oai_key,
        ant_key,
        xai_key,
        meta_key,
        deepseek_key,
        zai_key,
        moonshot_key,
        mimo_key,
        qwencloud_key,
        opencode_key,
        opencode_tier_val,
        or_key,
        comp_url,
        comp_key,
        model,
        temp,
        tp,
        tk,
        use_custom_sampling_val,
        max_tokens,
        trans_mode,
        ocr_method_val,
        max_fs,
        min_fs,
        ls,
        subpix,
        hint,
        liga,
        out_fmt,
        jq,
        pngc,
        verb,
        cleaning_only_val,
        upscaling_only_val,
        test_mode_val,
        s_in_lang,
        s_out_lang,
        s_font,
        b_in_lang,
        b_out_lang,
        b_font,
        enable_web_search_val,
        enable_code_execution_val,
        image_detail_val,
        media_resolution_val,
        media_resolution_bubbles_val,
        media_resolution_context_val,
        reasoning_effort_val,
        effort_val,
        verbosity_val,
        send_full_page_context_val,
        whiteout_conjoined_bubbles_val,
        upscale_method_val,
        bubble_min_side_pixels_val,
        context_image_max_side_pixels_val,
        osb_min_side_pixels_val,
        hyphenate_before_scaling_val,
        detach_trailing_punctuation_val,
        auto_vertical_text_val,
        vertical_line_spacing_mult_val,
        vertical_font_size_mult_val,
        auto_rotate_text_val,
        auto_rotate_confidence_threshold_val,
        special_instructions_val,
        batch_special_instructions_val,
        hyphen_penalty_val,
        hyphenation_min_word_length_val,
        badness_exponent_val,
        padding_pixels_val,
        supersampling_factor_val,
        outside_text_enabled_val,
        outside_text_seed_val,
        outside_text_inpainting_method_val,
        outside_text_flux_backend_val,
        outside_text_cloudflare_accounts_val,
        outside_text_colab_server_url_val,
        outside_text_flux_low_vram_val,
        outside_text_flux_sdcpp_cache_mode_val,
        outside_text_flux_sdcpp_diffusion_quant_val,
        outside_text_flux_sdcpp_text_encoder_quant_val,
        outside_text_flux_num_inference_steps_val,
        outside_text_flux_luminance_correction_val,
        outside_text_flux_upscale_small_crops_val,
        outside_text_flux_group_regions_val,
        outside_text_flux_residual_diff_threshold_val,
        outside_text_osb_confidence_val,
        outside_text_osb_text_free_only_val,
        outside_text_min_area_ignore_ratio_percent_val,
        outside_text_enable_page_number_filtering_val,
        outside_text_page_filter_margin_threshold_val,
        outside_text_page_filter_min_area_ratio_val,
        outside_text_huggingface_token_val,
        outside_text_osb_font_pack_val,
        outside_text_osb_max_font_size_val,
        outside_text_osb_min_font_size_val,
        outside_text_osb_use_ligatures_val,
        outside_text_osb_outline_width_val,
        outside_text_osb_line_spacing_val,
        outside_text_osb_padding_pixels_val,
        outside_text_osb_auto_vertical_text_val,
        outside_text_osb_vertical_line_spacing_mult_val,
        outside_text_osb_vertical_font_size_mult_val,
        outside_text_osb_use_subpixel_rendering_val,
        outside_text_osb_font_hinting_val,
        outside_text_bbox_expansion_percent_width_val,
        outside_text_bbox_expansion_percent_height_val,
        outside_text_osb_render_expansion_narrow_multiplier_val,
        outside_text_osb_render_expansion_aspect_ratio_threshold_val,
        outside_text_osb_render_expansion_tiny_multiplier_val,
        outside_text_osb_render_expansion_area_threshold_percent_val,
        outside_text_text_box_proximity_ratio_val,
        image_upscale_mode_val,
        image_upscale_factor_val,
        image_upscale_model_val,
        auto_scale_val,
        batch_parallel_requests_val,
        batch_parallel_within_pages_val,
        overlap_llm_with_inpaint_val,
        batch_overlap_llm_with_inpaint_val,
        batch_retry_failed_once_val,
        batch_previous_context_image_count_val,
        batch_previous_context_text_count_val,
        manual_json_translation_val,
    ) = args
    saved_settings = settings_manager.get_saved_settings()
    ui_state = UIConfigState(
        detection=UIDetectionSettings(
            confidence=conf,
            conjoined_confidence=conjoined_conf,
            panel_confidence=panel_conf,
            seg_model=seg_model,
            bubble_detector_model=bubble_detector_model,
            conjoined_detection=conjoined_detection,
            use_osb_text_verification=osb_text_verification,
            use_panel_sorting=use_panel_sorting,
        ),
        cleaning=UICleaningSettings(
            thresholding_value=thresholding_val,
            use_otsu_threshold=otsu,
            inpaint_colored_bubbles=inpaint_colored_bubbles,
            roi_shrink_px=int(max(0, min(8, roi_shrink_px))),
        ),
        outside_text=UIOutsideTextSettings(
            enabled=outside_text_enabled_val,
            seed=int(outside_text_seed_val),
            huggingface_token=outside_text_huggingface_token_val,
            inpainting_method=outside_text_inpainting_method_val,
            flux_backend=outside_text_flux_backend_val,
            cloudflare_accounts=outside_text_cloudflare_accounts_val,
            colab_server_url=outside_text_colab_server_url_val,
            flux_low_vram=outside_text_flux_low_vram_val,
            flux_sdcpp_cache_mode=outside_text_flux_sdcpp_cache_mode_val,
            flux_sdcpp_diffusion_quant=outside_text_flux_sdcpp_diffusion_quant_val,
            flux_sdcpp_text_encoder_quant=outside_text_flux_sdcpp_text_encoder_quant_val,
            flux_num_inference_steps=int(outside_text_flux_num_inference_steps_val),
            flux_luminance_correction=outside_text_flux_luminance_correction_val,
            flux_upscale_small_crops=outside_text_flux_upscale_small_crops_val,
            flux_group_regions=outside_text_flux_group_regions_val,
            flux_residual_diff_threshold=float(
                outside_text_flux_residual_diff_threshold_val
            ),
            osb_confidence=float(outside_text_osb_confidence_val),
            osb_text_free_only=bool(outside_text_osb_text_free_only_val),
            min_area_ignore_ratio=float(outside_text_min_area_ignore_ratio_percent_val)
            / 100.0,
            enable_page_number_filtering=outside_text_enable_page_number_filtering_val,
            page_filter_margin_threshold=float(
                outside_text_page_filter_margin_threshold_val
            ),
            page_filter_min_area_ratio=float(
                outside_text_page_filter_min_area_ratio_val
            ),
            osb_font_dir=outside_text_osb_font_pack_val,
            osb_max_font_size=int(outside_text_osb_max_font_size_val),
            osb_min_font_size=int(outside_text_osb_min_font_size_val),
            osb_use_ligatures=outside_text_osb_use_ligatures_val,
            osb_outline_width=float(outside_text_osb_outline_width_val),
            osb_line_spacing=float(outside_text_osb_line_spacing_val),
            osb_padding_pixels=float(outside_text_osb_padding_pixels_val),
            osb_auto_vertical_text=outside_text_osb_auto_vertical_text_val,
            osb_vertical_line_spacing_mult=float(
                outside_text_osb_vertical_line_spacing_mult_val
            ),
            osb_vertical_font_size_mult=float(
                outside_text_osb_vertical_font_size_mult_val
            ),
            osb_use_subpixel_rendering=outside_text_osb_use_subpixel_rendering_val,
            osb_font_hinting=outside_text_osb_font_hinting_val,
            bbox_expansion_percent_width=float(
                outside_text_bbox_expansion_percent_width_val
            ),
            bbox_expansion_percent_height=float(
                outside_text_bbox_expansion_percent_height_val
            ),
            osb_render_expansion_narrow_multiplier=float(
                outside_text_osb_render_expansion_narrow_multiplier_val
            ),
            osb_render_expansion_tiny_multiplier=float(
                outside_text_osb_render_expansion_tiny_multiplier_val
            ),
            osb_render_expansion_aspect_ratio_threshold=float(
                outside_text_osb_render_expansion_aspect_ratio_threshold_val
            ),
            osb_render_expansion_area_ratio_threshold=float(
                outside_text_osb_render_expansion_area_threshold_percent_val
            )
            / 100.0,
            text_box_proximity_ratio=float(outside_text_text_box_proximity_ratio_val),
        ),
        provider_settings=UITranslationProviderSettings(
            provider=prov,
            google_api_key=gem_key,
            openai_api_key=oai_key,
            anthropic_api_key=ant_key,
            xai_api_key=xai_key,
            meta_api_key=meta_key,
            deepseek_api_key=deepseek_key,
            zai_api_key=zai_key,
            moonshot_api_key=moonshot_key,
            mimo_api_key=mimo_key,
            qwencloud_api_key=qwencloud_key,
            opencode_api_key=opencode_key,
            opencode_tier=("go" if "go" in str(opencode_tier_val).lower() else "zen"),
            openrouter_api_key=or_key,
            openai_compatible_url=comp_url if prov == "OpenAI-Compatible" else saved_settings.get("openai_compatible_url", "http://localhost:8080/v1"),
            openai_compatible_api_key=comp_key if prov == "OpenAI-Compatible" else saved_settings.get("openai_compatible_api_key", ""),
            omniroute_url=comp_url if prov == "OmniRoute" else saved_settings.get("omniroute_url", "http://127.0.0.1:20128/v1"),
            omniroute_api_key=comp_key if prov == "OmniRoute" else saved_settings.get("omniroute_api_key", ""),
            ninerouter_url=comp_url if prov == "9router" else saved_settings.get("ninerouter_url", "http://127.0.0.1:20807/v1"),
            ninerouter_api_key=comp_key if prov == "9router" else saved_settings.get("ninerouter_api_key", ""),
            manual_json_translation=manual_json_translation_val,
        ),
        llm_settings=UITranslationLLMSettings(
            model_name=model,
            temperature=temp,
            top_p=tp,
            top_k=tk,
            max_tokens=max_tokens,
            reading_direction=rd,
            translation_mode=trans_mode,
            ocr_method=ocr_method_val,
            send_full_page_context=send_full_page_context_val,
            whiteout_conjoined_bubbles=whiteout_conjoined_bubbles_val,
            upscale_method=upscale_method_val,
            bubble_min_side_pixels=bubble_min_side_pixels_val,
            context_image_max_side_pixels=context_image_max_side_pixels_val,
            osb_min_side_pixels=osb_min_side_pixels_val,
            special_instructions=special_instructions_val,
        ),
        rendering=UIRenderingSettings(
            max_font_size=max_fs,
            min_font_size=min_fs,
            line_spacing_mult=ls,
            use_subpixel_rendering=subpix,
            font_hinting=hint,
            use_ligatures=liga,
            hyphenate_before_scaling=hyphenate_before_scaling_val,
            detach_trailing_punctuation=detach_trailing_punctuation_val,
            auto_vertical_text=auto_vertical_text_val,
            vertical_line_spacing_mult=float(vertical_line_spacing_mult_val),
            vertical_font_size_mult=float(vertical_font_size_mult_val),
            auto_rotate_text=bool(auto_rotate_text_val),
            auto_rotate_confidence_threshold=float(auto_rotate_confidence_threshold_val),
            hyphen_penalty=hyphen_penalty_val,
            hyphenation_min_word_length=hyphenation_min_word_length_val,
            badness_exponent=badness_exponent_val,
            padding_pixels=padding_pixels_val,
            supersampling_factor=int(supersampling_factor_val),
        ),
        output=UIOutputSettings(
            output_format=out_fmt,
            jpeg_quality=jq,
            png_compression=pngc,
            image_upscale_mode=image_upscale_mode_val,
            image_upscale_factor=image_upscale_factor_val,
            image_upscale_model=image_upscale_model_val,
        ),
        general=UIGeneralSettings(
            verbose=verb,
            cleaning_only=cleaning_only_val,
            upscaling_only=upscaling_only_val,
            test_mode=test_mode_val,
            enable_web_search=enable_web_search_val,
            enable_code_execution=enable_code_execution_val,
            use_custom_sampling=use_custom_sampling_val,
            image_detail=image_detail_val,
            media_resolution=media_resolution_val,
            media_resolution_bubbles=media_resolution_bubbles_val,
            media_resolution_context=media_resolution_context_val,
            reasoning_effort=reasoning_effort_val,
            effort=effort_val,
            verbosity=verbosity_val,
            auto_scale=auto_scale_val,
            overlap_llm_with_inpaint=bool(overlap_llm_with_inpaint_val),
        ),
        input_language=s_in_lang,
        output_language=s_out_lang,
        font_pack=s_font,
        batch_input_language=b_in_lang,
        batch_output_language=b_out_lang,
        batch_font_pack=b_font,
        batch_special_instructions=batch_special_instructions_val,
        batch_parallel_requests=int(batch_parallel_requests_val),
        batch_parallel_within_pages=bool(batch_parallel_within_pages_val),
        batch_overlap_llm_with_inpaint=bool(batch_overlap_llm_with_inpaint_val),
        batch_retry_failed_once=bool(batch_retry_failed_once_val),
        batch_previous_context_image_count=int(batch_previous_context_image_count_val),
        batch_previous_context_text_count=int(batch_previous_context_text_count_val),
    )

    if gem_key is not None:
        raw_lines = [l.strip() for l in str(gem_key).splitlines() if l.strip()]
        is_masked = len(raw_lines) > 0 and all(set(l) <= {"•", "*", "."} for l in raw_lines)
        if not is_masked:
            from core.services.credentials import save_secure_api_keys, clear_secure_api_keys
            import re

            tokens = [
                t.strip()
                for t in re.split(r"[\r\n,;]+", str(gem_key))
                if t.strip() and not t.strip().startswith("#")
            ]
            if tokens:
                save_secure_api_keys(tokens, "Google")
            else:
                clear_secure_api_keys("Google")
            global BATCH_GEMINI_SESSION
            if BATCH_GEMINI_SESSION:
                BATCH_GEMINI_SESSION.reload_keys()

    if outside_text_cloudflare_accounts_val is not None:
        raw_cf = [l.strip() for l in str(outside_text_cloudflare_accounts_val).splitlines() if l.strip()]
        is_cf_masked = len(raw_cf) > 0 and all(set(l) <= {"•", "*", "."} for l in raw_cf)
        if not is_cf_masked:
            from core.services.credentials import save_secure_cloudflare_accounts, clear_secure_cloudflare_accounts
            if raw_cf:
                save_secure_cloudflare_accounts(str(outside_text_cloudflare_accounts_val))
            else:
                clear_secure_cloudflare_accounts()

    # Convert UI state to dictionary for saving
    settings_dict = ui_state.to_save_dict()
    settings_dict["google_api_key"] = ""
    settings_dict["outside_text_cloudflare_accounts"] = ""
    settings_dict["send_full_page_context"] = bool(send_full_page_context_val)
    message = settings_manager.save_config(settings_dict)
    return message


def handle_reset_defaults_click(fonts_base_dir: Path) -> List[gr.update]:
    """Callback for the 'Reset Defaults' button. Uses dataclasses."""

    default_settings_dict = settings_manager.reset_to_defaults()
    default_ui_state = UIConfigState.from_dict(default_settings_dict)

    available_fonts, _ = utils.get_available_font_packs(fonts_base_dir)
    reset_single_font = default_ui_state.font_pack
    if reset_single_font not in available_fonts:
        reset_single_font = available_fonts[0] if available_fonts else None
    default_ui_state.font_pack = reset_single_font

    batch_reset_font = default_ui_state.batch_font_pack
    if batch_reset_font not in available_fonts:
        batch_reset_font = available_fonts[0] if available_fonts else None
    default_ui_state.batch_font_pack = batch_reset_font

    default_provider = default_ui_state.provider_settings.provider
    default_model_name = default_ui_state.llm_settings.model_name
    if not default_model_name:
        default_model_name = settings_manager.DEFAULT_SETTINGS["provider_models"].get(
            default_provider
        )
        default_ui_state.llm_settings.model_name = default_model_name

    if (
        default_provider == "OpenRouter"
        or default_provider == "OpenAI-Compatible"
        or default_provider == "OpenCode"
    ):
        default_models_choices = [default_model_name] if default_model_name else []
    else:
        default_models_choices = settings_manager.PROVIDER_MODELS.get(
            default_provider, []
        )

    gemini_visible = default_provider == "Google"
    openai_visible = default_provider == "OpenAI"
    anthropic_visible = default_provider == "Anthropic"
    xai_visible = default_provider == "SpaceXAI"
    meta_visible = default_provider == "Meta Model"
    deepseek_visible = default_provider == "DeepSeek"
    zai_visible = default_provider == "Z.ai"
    moonshot_visible = default_provider == "Moonshot AI"
    mimo_visible = default_provider == "Xiaomi MiMo"
    qwencloud_visible = default_provider == "QwenCloud"
    opencode_visible = default_provider == "OpenCode"
    opencode_tier_visible = default_provider == "OpenCode"
    openrouter_visible = default_provider == "OpenRouter"
    compatible_visible = default_provider == "OpenAI-Compatible"
    (
        temp_update,
        top_p_update,
        top_k_update,
        _,  # max_tokens_update - unused (using saved default instead)
        use_custom_sampling_update,
        enable_web_search_update,
        enable_code_execution_update,
        image_detail_update,
        media_resolution_update,
        media_resolution_bubbles_update,
        media_resolution_context_update,
        reasoning_effort_update,
        effort_update,
        verbosity_update,
    ) = utils.update_params_for_model(
        default_provider, default_model_name, default_ui_state.llm_settings.temperature
    )
    temp_val = temp_update.get("value", default_ui_state.llm_settings.temperature)
    temp_max = temp_update.get("maximum", 2.0)
    top_p_interactive = top_p_update.get("interactive", True)
    top_k_interactive = top_k_update.get("interactive", True)
    top_k_val = top_k_update.get("value", default_ui_state.llm_settings.top_k)
    use_custom_sampling_visible = use_custom_sampling_update.get("visible", True)
    is_reasoning = utils.is_reasoning_model(default_provider, default_model_name)
    max_tokens_val = 16384 if is_reasoning else 4096
    enable_web_search_visible = enable_web_search_update.get("visible", False)
    enable_code_execution_visible = enable_code_execution_update.get("visible", False)
    image_detail_visible = image_detail_update.get("visible", False)
    media_resolution_visible = media_resolution_update.get("visible", False)
    media_resolution_bubbles_visible = media_resolution_bubbles_update.get(
        "visible", False
    )
    media_resolution_context_visible = media_resolution_context_update.get(
        "visible", False
    )
    reasoning_visible = reasoning_effort_update.get("visible", False)
    reasoning_effort_val = reasoning_effort_update.get(
        "value", default_ui_state.general.reasoning_effort
    )
    effort_visible = effort_update.get("visible", False)
    effort_val = effort_update.get("value", default_ui_state.general.effort)
    verbosity_visible = verbosity_update.get("visible", False)
    verbosity_val = verbosity_update.get("value", default_ui_state.general.verbosity)
    text_encoder_quant = flux_sdcpp_valid_text_encoder_quant(
        default_ui_state.outside_text.inpainting_method,
        default_ui_state.outside_text.flux_sdcpp_text_encoder_quant,
    )
    text_encoder_quants = flux_sdcpp_text_encoder_quants(
        default_ui_state.outside_text.inpainting_method
    )
    text_layout_flags = text_layout_control_interactivity(
        default_ui_state.output_language,
        default_ui_state.batch_output_language,
        default_ui_state.rendering.hyphenate_before_scaling,
    )
    reset_hyphenate_relevant = text_layout_flags["hyphenate_before_scaling"]
    reset_hyphenate_value = (
        default_ui_state.rendering.hyphenate_before_scaling
        if reset_hyphenate_relevant
        else False
    )
    text_layout_flags = text_layout_control_interactivity(
        default_ui_state.output_language,
        default_ui_state.batch_output_language,
        reset_hyphenate_value,
    )

    return [
        default_ui_state.detection.confidence,
        default_ui_state.detection.conjoined_confidence,
        gr.update(
            value=default_ui_state.detection.panel_confidence,
            interactive=default_ui_state.detection.use_panel_sorting,
        ),
        default_ui_state.detection.seg_model,
        default_ui_state.detection.bubble_detector_model,
        default_ui_state.detection.conjoined_detection,
        default_ui_state.detection.use_osb_text_verification,
        default_ui_state.detection.use_panel_sorting,
        default_ui_state.llm_settings.reading_direction,
        default_ui_state.cleaning.thresholding_value,
        default_ui_state.cleaning.use_otsu_threshold,
        default_ui_state.cleaning.inpaint_colored_bubbles,
        default_ui_state.cleaning.roi_shrink_px,
        gr.update(value=default_provider),
        gr.update(
            value=get_masked_google_keys(),
            visible=gemini_visible,
        ),
        gr.update(
            value=default_ui_state.provider_settings.openai_api_key,
            visible=openai_visible,
        ),
        gr.update(
            value=default_ui_state.provider_settings.anthropic_api_key,
            visible=anthropic_visible,
        ),
        gr.update(
            value=default_ui_state.provider_settings.xai_api_key, visible=xai_visible
        ),
        gr.update(
            value=default_ui_state.provider_settings.meta_api_key,
            visible=meta_visible,
        ),
        gr.update(
            value=default_ui_state.provider_settings.deepseek_api_key,
            visible=deepseek_visible,
        ),
        gr.update(
            value=default_ui_state.provider_settings.zai_api_key,
            visible=zai_visible,
        ),
        gr.update(
            value=default_ui_state.provider_settings.moonshot_api_key,
            visible=moonshot_visible,
        ),
        gr.update(
            value=default_ui_state.provider_settings.mimo_api_key,
            visible=mimo_visible,
        ),
        gr.update(
            value=default_ui_state.provider_settings.qwencloud_api_key,
            visible=qwencloud_visible,
        ),
        gr.update(
            value=default_ui_state.provider_settings.opencode_api_key,
            visible=opencode_visible,
        ),
        gr.update(
            value=(
                "OpenCode Go"
                if (default_ui_state.provider_settings.opencode_tier or "").lower()
                == "go"
                else "OpenCode Zen"
            ),
            visible=opencode_tier_visible,
        ),
        gr.update(
            value=default_ui_state.provider_settings.openrouter_api_key,
            visible=openrouter_visible,
        ),
        gr.update(
            value=default_ui_state.provider_settings.openai_compatible_url,
            visible=compatible_visible,
        ),
        gr.update(
            value=default_ui_state.provider_settings.openai_compatible_api_key,
            visible=compatible_visible,
        ),
        gr.update(choices=default_models_choices, value=default_model_name),
        gr.update(
            value=temp_val,
            maximum=temp_max,
            visible=use_custom_sampling_visible,
        ),
        gr.update(
            value=default_ui_state.llm_settings.top_p,
            interactive=top_p_interactive,
            visible=use_custom_sampling_visible,
        ),
        gr.update(
            value=top_k_val,
            interactive=top_k_interactive,
            visible=use_custom_sampling_visible,
        ),
        gr.update(
            value=default_ui_state.general.use_custom_sampling,
            visible=use_custom_sampling_visible,
        ),
        gr.update(value=max_tokens_val),
        gr.update(value=default_ui_state.llm_settings.translation_mode),
        gr.update(value=default_ui_state.llm_settings.ocr_method),
        default_ui_state.rendering.max_font_size,
        default_ui_state.rendering.min_font_size,
        default_ui_state.rendering.line_spacing_mult,
        default_ui_state.rendering.use_subpixel_rendering,
        default_ui_state.rendering.font_hinting,
        default_ui_state.rendering.use_ligatures,
        default_ui_state.output.output_format,
        default_ui_state.output.jpeg_quality,
        default_ui_state.output.png_compression,
        default_ui_state.general.verbose,
        default_ui_state.general.cleaning_only,
        default_ui_state.general.upscaling_only,
        default_ui_state.general.test_mode,
        default_ui_state.input_language,
        default_ui_state.output_language,
        gr.update(value=default_ui_state.font_pack),
        default_ui_state.batch_input_language,
        default_ui_state.batch_output_language,
        gr.update(value=default_ui_state.batch_font_pack),
        gr.update(
            value=default_ui_state.general.enable_web_search,
            visible=enable_web_search_visible,
        ),
        gr.update(
            value=default_ui_state.general.enable_code_execution,
            visible=enable_code_execution_visible,
        ),
        gr.update(
            value=default_ui_state.general.image_detail,
            visible=image_detail_visible,
            choices=image_detail_update.get("choices"),
            info=image_detail_update.get("info"),
        ),
        gr.update(
            value=default_ui_state.general.media_resolution,
            visible=media_resolution_visible,
        ),
        gr.update(
            value=default_ui_state.general.media_resolution_bubbles,
            visible=media_resolution_bubbles_visible,
        ),
        gr.update(
            value=default_ui_state.general.media_resolution_context,
            visible=media_resolution_context_visible,
        ),
        gr.update(value=reasoning_effort_val, visible=reasoning_visible),
        gr.update(value=effort_val, visible=effort_visible),
        gr.update(value=verbosity_val, visible=verbosity_visible),
        "Settings reset to defaults (API keys preserved).",
        gr.update(value=default_ui_state.llm_settings.send_full_page_context),
        gr.update(value=default_ui_state.llm_settings.whiteout_conjoined_bubbles),
        gr.update(value=default_ui_state.llm_settings.upscale_method),
        default_ui_state.llm_settings.bubble_min_side_pixels,
        default_ui_state.llm_settings.context_image_max_side_pixels,
        default_ui_state.llm_settings.osb_min_side_pixels,
        gr.update(
            value=reset_hyphenate_value,
            interactive=reset_hyphenate_relevant,
        ),
        gr.update(value=default_ui_state.rendering.detach_trailing_punctuation),
        gr.update(value=default_ui_state.rendering.auto_vertical_text),
        default_ui_state.rendering.vertical_line_spacing_mult,
        default_ui_state.rendering.vertical_font_size_mult,
        gr.update(value=getattr(default_ui_state.rendering, "auto_rotate_text", True)),
        gr.update(
            value=getattr(
                default_ui_state.rendering, "auto_rotate_confidence_threshold", 0.5
            ),
            visible=getattr(default_ui_state.rendering, "auto_rotate_text", True),
        ),
        gr.update(
            value=default_ui_state.rendering.hyphen_penalty,
            interactive=text_layout_flags["hyphen_penalty"],
        ),
        gr.update(
            value=default_ui_state.rendering.hyphenation_min_word_length,
            interactive=text_layout_flags["hyphenation_min_word_length"],
        ),
        default_ui_state.llm_settings.special_instructions or "",
        default_ui_state.batch_special_instructions or "",
        default_ui_state.outside_text.enabled,
        default_ui_state.outside_text.seed,
        gr.update(
            value=default_ui_state.outside_text.inpainting_method,
            choices=_inpainting_method_choices(
                default_ui_state.outside_text.flux_backend
            ),
        ),
        gr.update(
            value=default_ui_state.outside_text.flux_backend,
            choices=_flux_backend_choices(
                default_ui_state.outside_text.inpainting_method
            ),
        ),
        default_ui_state.outside_text.flux_backend,
        gr.update(value=get_masked_cloudflare_accounts()),
        gr.update(value=default_ui_state.outside_text.colab_server_url),
        default_ui_state.outside_text.flux_low_vram,
        default_ui_state.outside_text.flux_sdcpp_cache_mode,
        gr.update(
            value=default_ui_state.outside_text.flux_sdcpp_diffusion_quant,
            choices=_radio_choices(FLUX_SDCPP_DIFFUSION_QUANTS),
        ),
        default_ui_state.outside_text.flux_sdcpp_diffusion_quant,
        gr.update(
            value=text_encoder_quant,
            choices=_radio_choices(text_encoder_quants),
        ),
        text_encoder_quant,
        default_ui_state.outside_text.flux_num_inference_steps,
        default_ui_state.outside_text.flux_luminance_correction,
        default_ui_state.outside_text.flux_upscale_small_crops,
        default_ui_state.outside_text.flux_group_regions,
        default_ui_state.outside_text.flux_residual_diff_threshold,
        default_ui_state.outside_text.osb_confidence,
        default_ui_state.outside_text.osb_text_free_only,
        default_ui_state.outside_text.min_area_ignore_ratio * 100.0,
        gr.update(value=default_ui_state.outside_text.enable_page_number_filtering),
        gr.update(
            value=default_ui_state.outside_text.page_filter_margin_threshold,
            interactive=default_ui_state.outside_text.enable_page_number_filtering,
        ),
        gr.update(
            value=default_ui_state.outside_text.page_filter_min_area_ratio,
            interactive=default_ui_state.outside_text.enable_page_number_filtering,
        ),
        default_ui_state.outside_text.huggingface_token,
        gr.update(value=default_ui_state.outside_text.osb_font_dir),
        default_ui_state.outside_text.osb_max_font_size,
        default_ui_state.outside_text.osb_min_font_size,
        default_ui_state.outside_text.osb_use_ligatures,
        default_ui_state.outside_text.osb_outline_width,
        default_ui_state.outside_text.osb_line_spacing,
        default_ui_state.outside_text.osb_padding_pixels,
        default_ui_state.outside_text.osb_auto_vertical_text,
        default_ui_state.outside_text.osb_vertical_line_spacing_mult,
        default_ui_state.outside_text.osb_vertical_font_size_mult,
        default_ui_state.outside_text.osb_use_subpixel_rendering,
        default_ui_state.outside_text.osb_font_hinting,
        default_ui_state.outside_text.bbox_expansion_percent_width,
        default_ui_state.outside_text.bbox_expansion_percent_height,
        default_ui_state.outside_text.osb_render_expansion_narrow_multiplier,
        default_ui_state.outside_text.osb_render_expansion_aspect_ratio_threshold,
        default_ui_state.outside_text.osb_render_expansion_tiny_multiplier,
        default_ui_state.outside_text.osb_render_expansion_area_ratio_threshold * 100.0,
        default_ui_state.outside_text.text_box_proximity_ratio,
        gr.update(value=default_ui_state.output.image_upscale_mode),
        gr.update(
            value=default_ui_state.output.image_upscale_factor,
            interactive=default_ui_state.output.image_upscale_mode != "off",
        ),
        gr.update(
            value=default_ui_state.output.image_upscale_model,
            interactive=default_ui_state.output.image_upscale_mode != "off",
        ),
        default_ui_state.general.auto_scale,
        default_ui_state.batch_parallel_requests,
        default_ui_state.batch_parallel_within_pages,
        default_ui_state.general.overlap_llm_with_inpaint,
        default_ui_state.batch_overlap_llm_with_inpaint,
        default_ui_state.batch_retry_failed_once,
        default_ui_state.batch_previous_context_image_count,
        default_ui_state.batch_previous_context_text_count,
    ]


def handle_provider_change(
    provider: str,
    ocr_method: str = "LLM",
    use_custom_sampling: bool = True,
    opencode_tier: Optional[str] = None,
):
    """Handles changes in the provider selector."""
    from core.caching import get_cache

    cache = get_cache()
    cache.clear_translation_cache()
    cache.clear_manga_ocr_cache()
    return utils.update_translation_ui(
        provider, ocr_method, use_custom_sampling, opencode_tier
    )


def handle_opencode_tier_change(
    opencode_tier: str,
    current_model: Optional[str],
    current_temp: float,
    use_custom_sampling: bool = True,
    ocr_method: str = "LLM",
):
    """Handles switching between OpenCode Zen and OpenCode Go endpoint tiers."""
    from core.caching import get_cache

    cache = get_cache()
    cache.clear_translation_cache()
    cache.clear_manga_ocr_cache()

    is_go = "go" in (opencode_tier or "").lower()
    tier_key = "go" if is_go else "zen"

    saved_settings = settings_manager.get_saved_settings()
    provider_models_dict = saved_settings.get(
        "provider_models", settings_manager.DEFAULT_SETTINGS["provider_models"]
    )

    cached_models = utils.OPENCODE_MODEL_CACHE.get(tier_key) or []
    if ocr_method == "LLM":
        available_choices = [
            m for m in cached_models if is_opencode_multimodal_model(m)
        ]
    else:
        available_choices = list(cached_models)

    remembered_model = provider_models_dict.get("OpenCode")
    selected_model = (
        remembered_model
        if remembered_model in available_choices
        else (
            available_choices[0]
            if available_choices
            else (remembered_model or current_model)
        )
    )

    model_update = gr.update(
        choices=available_choices
        if available_choices
        else ([selected_model] if selected_model else []),
        value=selected_model,
    )
    param_updates = utils.update_params_for_model(
        "OpenCode", selected_model, current_temp, use_custom_sampling
    )

    return (model_update, *param_updates)


def handle_output_format_change(output_format_value: str):
    """Handles changes in the output format radio button."""
    is_jpeg = output_format_value == "jpeg"
    is_png = output_format_value == "png"

    jpeg_interactive = not is_png
    png_interactive = not is_jpeg

    return gr.update(interactive=jpeg_interactive), gr.update(
        interactive=png_interactive
    )


def handle_refresh_resources_click(fonts_base_dir: Path):
    """Callback for the 'Refresh Models / Fonts' button."""
    return utils.refresh_models_and_fonts(fonts_base_dir)


def handle_unload_models_click():
    """Callback for the 'Unload Models' button."""
    try:
        from core.ml.model_manager import get_model_manager

        model_manager = get_model_manager()
        model_manager.unload_all()
        gr.Info("All models unloaded from memory successfully.")
    except Exception as e:
        gr.Error(f"Error unloading models: {str(e)}")


def handle_model_change(
    provider: str,
    model_name: Optional[str],
    current_temp: float,
    use_custom_sampling: bool = True,
):
    """Handles changes in the model name dropdown."""
    from core.caching import get_cache

    cache = get_cache()
    cache.clear_translation_cache()
    cache.clear_manga_ocr_cache()
    return utils.update_params_for_model(
        provider, model_name, current_temp, use_custom_sampling
    )


def handle_reasoning_effort_change(
    provider: str,
    model_name: Optional[str],
    reasoning_effort: Optional[str],
    use_custom_sampling: bool = True,
):
    """Updates temp/top_p slider interactivity when reasoning effort changes."""
    temp_ok, top_p_ok, top_k_ok = utils.get_sampling_slider_interactivity(
        provider,
        model_name,
        reasoning_effort,
        use_custom_sampling=use_custom_sampling,
    )
    visible = utils.is_use_custom_sampling_visible(
        provider, model_name, reasoning_effort
    )
    return (
        gr.update(interactive=temp_ok, visible=visible),
        gr.update(interactive=top_p_ok, visible=visible),
        gr.update(interactive=top_k_ok, visible=visible),
        gr.update(visible=visible),
    )


def handle_use_custom_sampling_change(
    provider: str,
    model_name: Optional[str],
    current_temp: float,
    use_custom_sampling: bool,
):
    """Updates sampling slider interactivity when custom sampling is toggled."""
    updates = utils.update_params_for_model(
        provider, model_name, current_temp, use_custom_sampling
    )
    return updates[0], updates[1], updates[2]


def handle_app_load(
    provider: str,
    url: str,
    key: Optional[str],
    opencode_tier: Optional[str] = None,
    opencode_key: Optional[str] = None,
    ocr_method: str = "LLM",
):
    """Callback for the app.load event to fetch dynamic models."""
    return utils.initial_dynamic_fetch(
        provider,
        url,
        key,
        opencode_tier=opencode_tier,
        opencode_key=opencode_key,
        ocr_method=ocr_method,
    )


def update_process_buttons(
    processing: bool, button_text_processing: str, button_text_idle: str
):
    """Toggle visibility and interactivity of processing buttons."""
    global CANCELLATION_MANAGER
    if not processing:
        CANCELLATION_MANAGER = None
    return (
        gr.update(
            interactive=not processing,
            value=button_text_processing if processing else button_text_idle,
        ),
        gr.update(visible=not processing),  # Clear button
        gr.update(visible=processing),  # Cancel button
        # Also update the other tab's buttons to prevent concurrent runs
        gr.update(interactive=not processing),
        gr.update(visible=not processing),
        gr.update(visible=processing),
    )


def cancel_process():
    """Signal that the current process should be cancelled."""
    if CANCELLATION_MANAGER:
        CANCELLATION_MANAGER.cancel()


def handle_thresholding_change(use_otsu_threshold: bool):
    """Handles changes in the Otsu thresholding checkbox to enable/disable thresholding_value slider."""
    return gr.update(interactive=not use_otsu_threshold)


def handle_text_layout_language_change(
    output_language: str,
    batch_output_language: str,
):
    """Enable/disable Text Layout hyphenation controls for the target language(s).

    Gradio checkboxes do not grey out clearly when only interactive=False, so when
    long-word breaking is irrelevant the checkbox is forced off. When it becomes
    relevant again, restore from saved config (same pattern as OCR-gated checkboxes
    in handle_ocr_method_change).
    """
    saved_settings = settings_manager.get_saved_settings()
    restored_hyphenate = saved_settings.get("hyphenate_before_scaling", True)
    flags = text_layout_control_interactivity(
        output_language, batch_output_language, restored_hyphenate
    )
    hyphenate_relevant = flags["hyphenate_before_scaling"]
    hyphenate_value = restored_hyphenate if hyphenate_relevant else False
    return (
        gr.update(interactive=hyphenate_relevant, value=hyphenate_value),
        gr.update(interactive=flags["hyphen_penalty"]),
        gr.update(interactive=flags["hyphenation_min_word_length"]),
    )


def handle_hyphenation_change(
    hyphenate_before_scaling: bool,
    output_language: str,
    batch_output_language: str,
):
    """Enable/disable hyphenation-related sliders from checkbox + target language(s)."""
    flags = text_layout_control_interactivity(
        output_language, batch_output_language, hyphenate_before_scaling
    )
    return (
        gr.update(interactive=flags["hyphen_penalty"]),
        gr.update(interactive=flags["hyphenation_min_word_length"]),
    )


def handle_cleaning_only_change(cleaning_only: bool):
    """Handles changes in the cleaning-only checkbox to disable upscaling-only and test mode if enabled."""
    return (
        gr.update(
            interactive=not cleaning_only, value=False if cleaning_only else None
        ),
        gr.update(
            interactive=not cleaning_only, value=False if cleaning_only else None
        ),
    )


def handle_upscaling_only_change(upscaling_only: bool):
    """Handles changes in the upscaling-only checkbox to disable cleaning-only and test mode if enabled."""
    return (
        gr.update(
            interactive=not upscaling_only, value=False if upscaling_only else None
        ),
        gr.update(
            interactive=not upscaling_only, value=False if upscaling_only else None
        ),
    )


def handle_test_mode_change(test_mode: bool):
    """Handles changes in the test mode checkbox to disable cleaning-only and upscaling-only if enabled."""
    return (
        gr.update(interactive=not test_mode, value=False if test_mode else None),
        gr.update(interactive=not test_mode, value=False if test_mode else None),
    )


def handle_conjoined_detection_change(_conjoined_detection: bool):
    """Handles changes in the conjoined detection checkbox to clear SAM cache."""
    from core.caching import get_cache

    cache = get_cache()
    cache.clear_sam_cache()
    return gr.update(interactive=_conjoined_detection)


def handle_confidence_threshold_change(_confidence: float):
    """Handles changes in confidence threshold settings to clear YOLO cache."""
    from core.caching import get_cache

    cache = get_cache()
    cache.clear_yolo_cache()
    return None


def handle_luminance_correction_change(_enabled: bool):
    """Handles changes in the flux luminance correction checkbox to clear inpaint cache."""
    from core.caching import get_cache

    cache = get_cache()
    cache.clear_inpaint_cache()
    return None


def handle_ocr_method_change(
    ocr_method: str,
    input_language: str,
    original_language_state: str,
    batch_input_language: str,
    batch_original_language_state: str,
    provider: str,
    current_model: Optional[str],
    openai_compatible_url: str,
    openai_compatible_api_key: Optional[str],
    opencode_tier: Optional[str] = None,
    opencode_api_key: Optional[str] = None,
):
    """Handles changes in OCR method selection."""
    import gradio as gr

    from core.caching import get_cache

    from . import layout, utils

    cache = get_cache()
    cache.clear_translation_cache()
    cache.clear_manga_ocr_cache()

    updates = []

    # Filter provider selector based on OCR method
    available_providers = utils.get_available_providers(ocr_method)
    current_provider = (
        provider
        if provider in available_providers
        else available_providers[0]
        if available_providers
        else "Google"
    )
    provider_selector_update = gr.update(
        choices=available_providers, value=current_provider
    )
    updates.append(provider_selector_update)

    if ocr_method == "manga-ocr":
        if input_language != "Japanese":
            saved_language = input_language
        else:
            saved_language = original_language_state

        updates.append(
            gr.update(value="Japanese", choices=["Japanese"], interactive=False)
        )
        updates.append(saved_language)

        if batch_input_language != "Japanese":
            batch_saved_language = batch_input_language
        else:
            batch_saved_language = batch_original_language_state

        updates.append(
            gr.update(value="Japanese", choices=["Japanese"], interactive=False)
        )
        updates.append(batch_saved_language)

        updates.append(gr.update(value=False, interactive=False))
        updates.append(gr.update(value=0, interactive=False))
        updates.append(gr.update())
        # Disable code execution checkbox (Gemini Flash only, disabled in text-only mode)
        updates.append(gr.update(value=False, interactive=False))
        updates.append(gr.update(interactive=False))
        # Disable media resolution dropdowns (no images sent to LLM in text-only mode)
        updates.append(gr.update(interactive=False))
        updates.append(gr.update(interactive=False))
        updates.append(gr.update(interactive=False))

        # Trigger model list refresh for providers with dynamic model lists
        if current_provider == "OpenRouter":
            model_update = utils.fetch_and_update_openrouter_models(
                ocr_method="manga-ocr", current_model=current_model
            )
            updates.append(model_update)
        elif current_provider in ("OpenAI-Compatible", "OmniRoute", "9router"):
            model_update = utils.fetch_and_update_compatible_models(
                openai_compatible_url, openai_compatible_api_key, current_model
            )
            updates.append(model_update)
        elif current_provider == "Z.ai":
            # For manga-ocr mode, show all Z.ai models (text-only models work)
            models = settings_manager.PROVIDER_MODELS.get("Z.ai", [])
            saved_settings = settings_manager.get_saved_settings()
            provider_models_dict = saved_settings.get(
                "provider_models", settings_manager.DEFAULT_SETTINGS["provider_models"]
            )
            remembered_model = provider_models_dict.get("Z.ai")
            selected_model = (
                remembered_model
                if remembered_model in models
                else (models[0] if models else None)
            )
            updates.append(gr.update(choices=models, value=selected_model))
        elif current_provider == "Moonshot AI":
            models = settings_manager.PROVIDER_MODELS.get("Moonshot AI", [])
            saved_settings = settings_manager.get_saved_settings()
            provider_models_dict = saved_settings.get(
                "provider_models", settings_manager.DEFAULT_SETTINGS["provider_models"]
            )
            remembered_model = provider_models_dict.get("Moonshot AI")
            selected_model = (
                remembered_model
                if remembered_model in models
                else (models[0] if models else None)
            )
            updates.append(gr.update(choices=models, value=selected_model))
        elif current_provider == "Xiaomi MiMo":
            models = settings_manager.PROVIDER_MODELS.get("Xiaomi MiMo", [])
            saved_settings = settings_manager.get_saved_settings()
            provider_models_dict = saved_settings.get(
                "provider_models", settings_manager.DEFAULT_SETTINGS["provider_models"]
            )
            remembered_model = provider_models_dict.get("Xiaomi MiMo")
            selected_model = (
                remembered_model
                if remembered_model in models
                else (models[0] if models else None)
            )
            updates.append(gr.update(choices=models, value=selected_model))
        elif current_provider == "Meta Model":
            models = settings_manager.PROVIDER_MODELS.get("Meta Model", [])
            saved_settings = settings_manager.get_saved_settings()
            provider_models_dict = saved_settings.get(
                "provider_models", settings_manager.DEFAULT_SETTINGS["provider_models"]
            )
            remembered_model = provider_models_dict.get("Meta Model")
            selected_model = (
                remembered_model
                if remembered_model in models
                else (models[0] if models else None)
            )
            updates.append(gr.update(choices=models, value=selected_model))
        elif current_provider == "QwenCloud":
            models = settings_manager.PROVIDER_MODELS.get("QwenCloud", [])
            saved_settings = settings_manager.get_saved_settings()
            provider_models_dict = saved_settings.get(
                "provider_models", settings_manager.DEFAULT_SETTINGS["provider_models"]
            )
            remembered_model = provider_models_dict.get("QwenCloud")
            selected_model = (
                remembered_model
                if remembered_model in models
                else (models[0] if models else None)
            )
            updates.append(gr.update(choices=models, value=selected_model))
        elif current_provider == "OpenCode":
            saved_settings = settings_manager.get_saved_settings()
            tier = opencode_tier or saved_settings.get("opencode_tier") or "zen"
            key = opencode_api_key or saved_settings.get("opencode_api_key") or ""
            model_update = utils.fetch_and_update_opencode_models(
                tier=tier,
                api_key=key,
                current_model=current_model,
                ocr_method=ocr_method,
            )
            updates.append(model_update)
        else:
            updates.append(gr.update())
    elif ocr_method == "paddleocr-vl-1.6":
        # Restrict to PaddleOCR-VL-1.6 supported languages; keep current if supported
        paddle_langs = layout.PADDLE_OCR_VL_LANGUAGES
        if input_language in paddle_langs:
            saved_language = original_language_state
            selected_lang = input_language
        else:
            saved_language = input_language
            selected_lang = (
                "Japanese" if "Japanese" in paddle_langs else paddle_langs[0]
            )

        updates.append(
            gr.update(value=selected_lang, choices=paddle_langs, interactive=True)
        )
        updates.append(saved_language)

        if batch_input_language in paddle_langs:
            batch_saved_language = batch_original_language_state
            batch_selected_lang = batch_input_language
        else:
            batch_saved_language = batch_input_language
            batch_selected_lang = (
                "Japanese" if "Japanese" in paddle_langs else paddle_langs[0]
            )

        updates.append(
            gr.update(value=batch_selected_lang, choices=paddle_langs, interactive=True)
        )
        updates.append(batch_saved_language)

        updates.append(gr.update(value=False, interactive=False))
        updates.append(gr.update(value=0, interactive=False))
        updates.append(gr.update())
        updates.append(gr.update(value=False, interactive=False))
        updates.append(gr.update(interactive=False))
        updates.append(gr.update(interactive=False))
        updates.append(gr.update(interactive=False))
        updates.append(gr.update(interactive=False))

        # Model list refresh — same as manga-ocr (enables text-only providers)
        if current_provider == "OpenRouter":
            model_update = utils.fetch_and_update_openrouter_models(
                ocr_method="manga-ocr", current_model=current_model
            )
            updates.append(model_update)
        elif current_provider in ("OpenAI-Compatible", "OmniRoute", "9router"):
            model_update = utils.fetch_and_update_compatible_models(
                openai_compatible_url, openai_compatible_api_key, current_model
            )
            updates.append(model_update)
        elif current_provider == "Z.ai":
            models = settings_manager.PROVIDER_MODELS.get("Z.ai", [])
            saved_settings = settings_manager.get_saved_settings()
            provider_models_dict = saved_settings.get(
                "provider_models", settings_manager.DEFAULT_SETTINGS["provider_models"]
            )
            remembered_model = provider_models_dict.get("Z.ai")
            selected_model = (
                remembered_model
                if remembered_model in models
                else (models[0] if models else None)
            )
            updates.append(gr.update(choices=models, value=selected_model))
        elif current_provider == "Moonshot AI":
            models = settings_manager.PROVIDER_MODELS.get("Moonshot AI", [])
            saved_settings = settings_manager.get_saved_settings()
            provider_models_dict = saved_settings.get(
                "provider_models", settings_manager.DEFAULT_SETTINGS["provider_models"]
            )
            remembered_model = provider_models_dict.get("Moonshot AI")
            selected_model = (
                remembered_model
                if remembered_model in models
                else (models[0] if models else None)
            )
            updates.append(gr.update(choices=models, value=selected_model))
        elif current_provider == "Xiaomi MiMo":
            models = settings_manager.PROVIDER_MODELS.get("Xiaomi MiMo", [])
            saved_settings = settings_manager.get_saved_settings()
            provider_models_dict = saved_settings.get(
                "provider_models", settings_manager.DEFAULT_SETTINGS["provider_models"]
            )
            remembered_model = provider_models_dict.get("Xiaomi MiMo")
            selected_model = (
                remembered_model
                if remembered_model in models
                else (models[0] if models else None)
            )
            updates.append(gr.update(choices=models, value=selected_model))
        elif current_provider == "Meta Model":
            models = settings_manager.PROVIDER_MODELS.get("Meta Model", [])
            saved_settings = settings_manager.get_saved_settings()
            provider_models_dict = saved_settings.get(
                "provider_models", settings_manager.DEFAULT_SETTINGS["provider_models"]
            )
            remembered_model = provider_models_dict.get("Meta Model")
            selected_model = (
                remembered_model
                if remembered_model in models
                else (models[0] if models else None)
            )
            updates.append(gr.update(choices=models, value=selected_model))
        elif current_provider == "QwenCloud":
            models = settings_manager.PROVIDER_MODELS.get("QwenCloud", [])
            saved_settings = settings_manager.get_saved_settings()
            provider_models_dict = saved_settings.get(
                "provider_models", settings_manager.DEFAULT_SETTINGS["provider_models"]
            )
            remembered_model = provider_models_dict.get("QwenCloud")
            selected_model = (
                remembered_model
                if remembered_model in models
                else (models[0] if models else None)
            )
            updates.append(gr.update(choices=models, value=selected_model))
        elif current_provider == "OpenCode":
            saved_settings = settings_manager.get_saved_settings()
            tier = opencode_tier or saved_settings.get("opencode_tier") or "zen"
            key = opencode_api_key or saved_settings.get("opencode_api_key") or ""
            model_update = utils.fetch_and_update_opencode_models(
                tier=tier,
                api_key=key,
                current_model=current_model,
                ocr_method=ocr_method,
            )
            updates.append(model_update)
        else:
            updates.append(gr.update())
    else:
        restored_language = (
            original_language_state if original_language_state else input_language
        )
        updates.append(
            gr.update(
                value=restored_language,
                choices=layout.SOURCE_LANGUAGES,
                interactive=True,
            )
        )
        updates.append(original_language_state)

        batch_restored_language = (
            batch_original_language_state
            if batch_original_language_state
            else batch_input_language
        )
        updates.append(
            gr.update(
                value=batch_restored_language,
                choices=layout.SOURCE_LANGUAGES,
                interactive=True,
            )
        )
        updates.append(batch_original_language_state)

        saved_settings = settings_manager.get_saved_settings()
        restored_send_full_page_context = saved_settings.get(
            "send_full_page_context", True
        )
        updates.append(
            gr.update(value=restored_send_full_page_context, interactive=True)
        )
        restored_previous_context_count = saved_settings.get(
            "batch_previous_context_image_count", 0
        )
        updates.append(
            gr.update(
                value=(
                    restored_previous_context_count
                    if restored_send_full_page_context
                    else 0
                ),
                interactive=restored_send_full_page_context,
            )
        )
        restored_whiteout = saved_settings.get("whiteout_conjoined_bubbles", True)
        updates.append(gr.update(value=restored_whiteout, interactive=True))
        updates.append(gr.update(interactive=True))
        updates.append(gr.update(interactive=True))
        updates.append(gr.update(interactive=True))
        updates.append(gr.update(interactive=True))
        updates.append(gr.update(interactive=True))

        # Trigger model list refresh for providers with dynamic or filtered model lists
        if current_provider == "OpenRouter":
            model_update = utils.fetch_and_update_openrouter_models(
                ocr_method="LLM", current_model=current_model
            )
            updates.append(model_update)
        elif current_provider in ("OpenAI-Compatible", "OmniRoute", "9router"):
            model_update = utils.fetch_and_update_compatible_models(
                openai_compatible_url, openai_compatible_api_key, current_model
            )
            updates.append(model_update)
        elif current_provider == "Z.ai":
            # For LLM OCR mode, only show Z.ai vision models
            models = [
                m for m in settings_manager.PROVIDER_MODELS.get("Z.ai", []) if "v" in m
            ]
            saved_settings = settings_manager.get_saved_settings()
            provider_models_dict = saved_settings.get(
                "provider_models", settings_manager.DEFAULT_SETTINGS["provider_models"]
            )
            remembered_model = provider_models_dict.get("Z.ai")
            selected_model = (
                remembered_model
                if remembered_model in models
                else (models[0] if models else None)
            )
            updates.append(gr.update(choices=models, value=selected_model))
        elif current_provider == "Moonshot AI":
            models = settings_manager.PROVIDER_MODELS.get("Moonshot AI", [])
            saved_settings = settings_manager.get_saved_settings()
            provider_models_dict = saved_settings.get(
                "provider_models", settings_manager.DEFAULT_SETTINGS["provider_models"]
            )
            remembered_model = provider_models_dict.get("Moonshot AI")
            selected_model = (
                remembered_model
                if remembered_model in models
                else (models[0] if models else None)
            )
            updates.append(gr.update(choices=models, value=selected_model))
        elif current_provider == "Xiaomi MiMo":
            models = [
                m
                for m in settings_manager.PROVIDER_MODELS.get("Xiaomi MiMo", [])
                if m.lower() == "mimo-v2.5"
            ]
            saved_settings = settings_manager.get_saved_settings()
            provider_models_dict = saved_settings.get(
                "provider_models", settings_manager.DEFAULT_SETTINGS["provider_models"]
            )
            remembered_model = provider_models_dict.get("Xiaomi MiMo")
            selected_model = (
                remembered_model
                if remembered_model in models
                else (models[0] if models else None)
            )
            updates.append(gr.update(choices=models, value=selected_model))
        elif current_provider == "Meta Model":
            models = settings_manager.PROVIDER_MODELS.get("Meta Model", [])
            saved_settings = settings_manager.get_saved_settings()
            provider_models_dict = saved_settings.get(
                "provider_models", settings_manager.DEFAULT_SETTINGS["provider_models"]
            )
            remembered_model = provider_models_dict.get("Meta Model")
            selected_model = (
                remembered_model
                if remembered_model in models
                else (models[0] if models else None)
            )
            updates.append(gr.update(choices=models, value=selected_model))
        elif current_provider == "QwenCloud":
            models = settings_manager.PROVIDER_MODELS.get("QwenCloud", [])
            saved_settings = settings_manager.get_saved_settings()
            provider_models_dict = saved_settings.get(
                "provider_models", settings_manager.DEFAULT_SETTINGS["provider_models"]
            )
            remembered_model = provider_models_dict.get("QwenCloud")
            selected_model = (
                remembered_model
                if remembered_model in models
                else (models[0] if models else None)
            )
            updates.append(gr.update(choices=models, value=selected_model))
        elif current_provider == "OpenCode":
            saved_settings = settings_manager.get_saved_settings()
            tier = opencode_tier or saved_settings.get("opencode_tier") or "zen"
            key = opencode_api_key or saved_settings.get("opencode_api_key") or ""
            model_update = utils.fetch_and_update_opencode_models(
                tier=tier,
                api_key=key,
                current_model=current_model,
                ocr_method=ocr_method,
            )
            updates.append(model_update)
        else:
            updates.append(gr.update())

    updates.append(current_provider)

    return updates


def handle_translation_mode_change(translation_mode: str, current_ocr_method: str):
    """Handles changes in translation mode to enable/disable OCR method selection."""
    import gradio as gr

    if translation_mode == "one-step":
        if current_ocr_method in ("manga-ocr", "paddleocr-vl-1.6"):
            return gr.update(value="LLM", interactive=False)
        else:
            return gr.update(interactive=False)
    else:
        return gr.update(interactive=True)


def handle_input_image_change(image_path: Any) -> Any:
    """Loads existing translation file if it exists, otherwise clears the textbox."""
    import gradio as gr
    from pathlib import Path
    if not image_path or not isinstance(image_path, (str, Path)):
        return gr.update(value="")
    try:
        _json_base = Path(image_path)
        _base_name = _json_base.name.split('_translated_')[0] if '_translated_' in _json_base.name else _json_base.stem
        _txt_path = Path("./output") / f"{_base_name}_translation.txt"
        if _txt_path.exists():
            with open(_txt_path, "r", encoding="utf-8") as f:
                return gr.update(value=f.read())
    except Exception:
        pass
    return gr.update(value="")


# --- Batch Interactive Gemini Chat Handlers ---
from core.services.gemini_session_manager import GeminiChatSession, AVAILABLE_CHAT_MODELS
from core.services.batch_translation_parser import (
    parse_chat_response_to_batch_manual,
    parse_and_validate_batch_manual,
    extract_file_names_from_batch_manual,
    extract_ocr_blocks_for_files,
    find_problematic_lines,
    validate_batch_translations_stats,
)
import os

def get_masked_google_keys() -> str:
    from core.services.credentials import get_secure_api_keys

    keys = get_secure_api_keys("Google")
    if not keys:
        return ""
    return "\n".join(["•" * 36 for _ in keys])


def handle_toggle_google_key(revealed: bool):
    import gradio as gr
    from core.services.credentials import get_secure_api_keys

    keys = get_secure_api_keys("Google")
    if not revealed:
        return "\n".join(keys), gr.update(value="Show/Hide"), True
    else:
        return (
            get_masked_google_keys(),
            gr.update(value="Show/Hide"),
            False,
        )


def get_masked_cloudflare_accounts() -> str:
    from core.services.credentials import get_secure_cloudflare_accounts

    accounts = get_secure_cloudflare_accounts()
    if not accounts:
        return ""
    return "\n".join(["•" * 36 for _ in accounts])


def handle_toggle_cloudflare_accounts(revealed: bool):
    import gradio as gr
    from core.services.credentials import get_secure_cloudflare_accounts

    accounts = get_secure_cloudflare_accounts()
    if not revealed:
        return "\n".join(accounts), gr.update(value="Show/Hide"), True
    else:
        return (
            get_masked_cloudflare_accounts(),
            gr.update(value="Show/Hide"),
            False,
        )


BATCH_GEMINI_SESSION: Optional[GeminiChatSession] = None


def get_batch_gemini_session() -> GeminiChatSession:
    global BATCH_GEMINI_SESSION
    settings = settings_manager.get_saved_settings()
    api_key = (
        settings.get("google_api_key")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or ""
    )
    if not BATCH_GEMINI_SESSION:
        BATCH_GEMINI_SESSION = GeminiChatSession(fallback_api_key=api_key)
    else:
        BATCH_GEMINI_SESSION.reload_keys(fallback_key=api_key)
    return BATCH_GEMINI_SESSION


def handle_batch_chat_model_change(selected_model: str):
    session = get_batch_gemini_session()
    session.set_model(selected_model)
    return f"**Active Model:** `{selected_model}`"


def handle_batch_chat_cancel(current_history: List[Any]):
    session = get_batch_gemini_session()
    next_key = session.cancel_and_rotate_key()
    key_info = f" Switched to {next_key}." if next_key else ""
    status_msg = f"Request cancelled on `{session.current_model}`.{key_info} You can send another request."

    hist = list(current_history or [])
    if hist and isinstance(hist[-1], (list, tuple)) and len(hist[-1]) > 1:
        last_bot_msg = str(hist[-1][1] or "")
        if "Thinking..." in last_bot_msg or "Analyzing..." in last_bot_msg or "Translating..." in last_bot_msg or "Fixing..." in last_bot_msg or "⏳" in last_bot_msg or not last_bot_msg:
            hist[-1] = (hist[-1][0], "*(Request cancelled)*")
    return hist, status_msg


def handle_batch_chat_init(manual_txt: str, current_history: List[Any], selected_model: str):
    session = get_batch_gemini_session()
    if selected_model:
        session.set_model(selected_model)

    if not session.has_keys:
        error_msg = "No API Key found in vault or configuration."
        hist = list(current_history or [])
        hist.append(("Load OCR", error_msg))
        yield hist, "**Error:** Missing API Key"
        return

    txt_content = (manual_txt or "").strip()
    if not txt_content:
        error_msg = "Batch manual box is empty! Please load files or run OCR first."
        hist = list(current_history or [])
        hist.append(("Load OCR", error_msg))
        yield hist, "**Warning:** Batch box empty"
        return

    session.reset()
    user_prompt = (
        "Dưới đây là toàn bộ văn bản OCR của tập truyện theo từng file ảnh:\n\n"
        f"{txt_content}\n\n"
        "Hãy đọc toàn bộ câu thoại, phân tích bối cảnh câu chuyện, nhận diện nhân vật, mối quan hệ và đề xuất Bảng quy ước xưng hô phù hợp cho truyện 18+. "
        "Chưa cần dịch vội, hãy tóm tắt và chờ tôi xác nhận hoặc yêu cầu dịch từng cụm trang cụ thể."
    )

    init_prompt_label = "Load OCR to AI"
    # Hiển thị ngay lập tức tin nhắn của user và trạng thái đang xử lý
    yield [(init_prompt_label, "Analyzing context and addressing rules...")], f"Sending request to `{session.current_model}`..."

    try:
        status_info, current_model, model_text = session.send_message(user_prompt)
        yield [(init_prompt_label, model_text)], f"**Model:** `{current_model}` | {status_info}"
    except Exception as e:
        yield [(init_prompt_label, f"Error: {str(e)}")], f"**Error:** {str(e)}"


def handle_batch_chat_send(user_msg: str, current_history: List[Any], selected_model: str):
    msg = (user_msg or "").strip()
    if not msg:
        yield current_history or [], "", "**Message is empty**"
        return

    session = get_batch_gemini_session()
    if selected_model:
        session.set_model(selected_model)

    hist = list(current_history or [])
    if not session.has_keys:
        error_msg = "No API Key found in vault or configuration."
        hist.append((msg, error_msg))
        yield hist, "", "**Error:** Missing API Key"
        return

    # Lượt 1: Hiện ngay tin nhắn người dùng và làm sạch ô nhập liệu
    hist.append([msg, "Thinking..."])
    yield hist, "", f"Sending request to `{session.current_model}`..."

    # Lượt 2: Gọi LLM và cập nhật câu trả lời khi có kết quả
    try:
        status_info, current_model, model_text = session.send_message(msg)
        hist[-1] = [msg, model_text]
        yield hist, "", f"**Model:** `{current_model}` | {status_info}"
    except Exception as e:
        hist[-1] = [msg, f"Error: {str(e)}"]
        yield hist, "", f"**Error:** {str(e)}"


def handle_batch_chat_auto_translate_all(
    manual_txt: str, current_history: List[Any], selected_model: str
):
    session = get_batch_gemini_session()
    if selected_model:
        session.set_model(selected_model)

    hist = list(current_history or [])
    if not session.has_keys:
        error_msg = "No API Key found in vault or configuration."
        hist.append(("Auto Translate All", error_msg))
        yield hist, "", "**Error:** Missing API Key"
        return

    txt_content = (manual_txt or "").strip()
    if not txt_content:
        error_msg = "Batch manual box is empty! Please load files or run OCR first."
        hist.append(("Auto Translate All", error_msg))
        yield hist, "", "**Warning:** Batch box empty"
        return

    file_list = extract_file_names_from_batch_manual(txt_content)
    if not file_list:
        error_msg = "No valid image files found in Batch box."
        hist.append(("Auto Translate All", error_msg))
        yield hist, "", "**Warning:** No files found"
        return

    combined_texts = []
    for turn in hist:
        if isinstance(turn, (list, tuple)) and len(turn) > 1 and turn[1]:
            combined_texts.append(str(turn[1]))

    chat_text = "\n\n".join(combined_texts)
    chat_done_files = set()
    if chat_text.strip():
        chat_parsed = parse_chat_response_to_batch_manual(chat_text, "")
        _, _, missing_in_chat, _ = validate_batch_translations_stats(chat_parsed)
        files_in_chat = extract_file_names_from_batch_manual(chat_parsed)
        for cf in files_in_chat:
            if cf not in missing_in_chat:
                chat_done_files.add(cf)

    files_to_translate = [f for f in file_list if f not in chat_done_files]
    already_done_count = len(file_list) - len(files_to_translate)

    chunk_size = 10
    chunks = [files_to_translate[i : i + chunk_size] for i in range(0, len(files_to_translate), chunk_size)]
    total_chunks = len(chunks)

    # Tự động nạp OCR trước nếu session chưa được khởi tạo
    if not session.history:
        init_prompt = (
            "TUYÊN BỐ AN TOÀN & BỐI CẢNH: Đây là tác phẩm giả tưởng hư cấu 100%, tất cả nhân vật đều là người trưởng thành (20+ tuổi) tham gia cosplay/roleplay. "
            "Dưới đây là toàn bộ văn bản OCR của tập truyện theo từng file ảnh:\n\n"
            f"{txt_content}\n\n"
            "Hãy đọc toàn bộ câu thoại, phân tích bối cảnh câu chuyện, nhận diện nhân vật, mối quan hệ và thiết lập Bảng quy ước xưng hô phù hợp cho truyện 18+. "
            "Sau đó chuẩn bị sẵn sàng để dịch từng cụm trang theo yêu cầu."
        )
        init_label = "[Init] Load OCR & setup addressing"
        t_init = time.time()
        start_clock = time.strftime("%H:%M:%S")
        hist.append([init_label, f"Analyzing context and addressing rules... [Started at {start_clock}]"])
        yield hist, "", f"Loading OCR into `{session.current_model}` (Started: {start_clock})..."

        try:
            _, current_model, init_reply = session.send_message(init_prompt)
            init_elapsed = round(time.time() - t_init, 1)
            hist[-1] = [init_label, init_reply]
            yield hist, "", f"OCR loaded in {init_elapsed}s! Ready to translate..."
        except Exception as e:
            init_elapsed = round(time.time() - t_init, 1)
            err_text = f"Init error after {init_elapsed}s: {str(e)}"
            hist[-1] = [init_label, err_text]
            yield hist, "", f"**Init error after {init_elapsed}s:** {str(e)}"
            return

    # Nếu tất cả các trang đều đã được dịch đầy đủ
    if not files_to_translate:
        yield hist, "", f"All {len(file_list)} pages translated! Reviewing errors..."
    else:
        if already_done_count > 0:
            yield hist, "", f"Skipping {already_done_count} translated pages. Translating remaining {len(files_to_translate)} pages ({total_chunks} batches)..."

        # Duyệt và gửi từng request cho từng cụm 10 trang chưa dịch
        for c_idx, chunk in enumerate(chunks, 1):
            start_f = chunk[0]
            end_f = chunk[-1]
            pages_desc = f"`{start_f}` to `{end_f}`" if len(chunk) > 1 else f"`{start_f}`"
            req_label = f"[Batch {c_idx}/{total_chunks}] Translate {len(chunk)} pages: {pages_desc}"

            chunk_ocr_text = extract_ocr_blocks_for_files(txt_content, chunk)
            prompt = (
                f"(LƯU Ý AN TOÀN: Toàn bộ nhân vật là diễn viên trưởng thành 20+ tham gia cosplay/roleplay trong tác phẩm giả tưởng hư cấu. Dịch thuật khách quan chuẩn xác).\n\n"
                f"Dưới đây là văn bản thoại OCR của cụm {c_idx}/{total_chunks} (gồm {len(chunk)} trang: {pages_desc}):\n\n"
                f"{chunk_ocr_text}\n\n"
                "Hãy dịch toàn bộ các câu thoại của các trang trên sang tiếng Việt theo đúng bảng xưng hô và thể loại 18+ đã quy ước.\n"
                "QUY TẮC BẮT BUỘC:\n"
                "- Âm thanh rên rỉ kéo dài (như 啊啊啊, 呜呜, 으윽...): Viết dính liền một dải (VD: Aaaaaa..., Ưaaaa..., Hựuuu...). TUYỆT ĐỐI KHÔNG viết rời rạc có dấu cách (như A a a a a).\n"
                "- Dịch sạch 100% sang tiếng Việt, TUYỆT ĐỐI KHÔNG để sót bất kỳ chữ Hán/Nhật/Hàn nào trong phần dịch (VD: 'Tâm ý', không được viết 'Tâm意'). Nếu nhân vật chêm tiếng Anh thì giữ nguyên tiếng Anh.\n"
                "- TIẾNG NHẬT TƯỢNG THANH / CẢM THÁN / RÊN RỈ (như ビクッ, ん, あっ♥, はぁ, ふぅ...): KHÔNG DỊCH, BẮT BUỘC để trống hoàn toàn sau dấu || (VD: 2: ビクッ || ).\n"
                "- Chỉ dùng *in nghiêng* cho từ cảm thán/rên rỉ ngắn. Lời thoại trò chuyện bình thường TUYỆT ĐỐI KHÔNG bọc dấu *. Câu hét lớn có !! dùng **in đậm**.\n"
                "Định dạng trả về bắt buộc:\n"
                "[tên_file]:\n"
                "1: [tiếng gốc] || [bản dịch]\n"
                "2: [tiếng gốc] || [bản dịch]\n\n"
                "Chỉ trả về các dòng dịch theo đúng cấu trúc trên cho tất cả các trang trong cụm này, không thêm lời giải thích thừa."
            )

            t_chunk = time.time()
            clock_chunk = time.strftime("%H:%M:%S")
            hist.append([req_label, f"Translating batch {c_idx}/{total_chunks} ({pages_desc})... [Started at {clock_chunk}]"])
            yield hist, "", f"Translating batch {c_idx}/{total_chunks} ({pages_desc}) on `{session.current_model}` (Started: {clock_chunk})..."

            try:
                status_info, current_model, model_text = session.send_message(prompt)
                chunk_elapsed = round(time.time() - t_chunk, 1)
                hist[-1] = [req_label, model_text]
                yield hist, "", f"Finished batch {c_idx}/{total_chunks} ({pages_desc}) in {chunk_elapsed}s. Model: `{current_model}` | {status_info}"
            except Exception as e:
                chunk_elapsed = round(time.time() - t_chunk, 1)
                err_msg = f"Error at batch {c_idx}/{total_chunks} ({pages_desc}) after {chunk_elapsed}s: {str(e)}"
                hist[-1] = [req_label, err_msg]
                yield hist, "", f"**{err_msg}**"
                return

    # Tự động rà soát và sửa lỗi (Auto-Fix Request) nếu phát hiện lỗi sót chữ Hán/Hàn hoặc thiếu dòng
    combined_texts = []
    for turn in hist:
        if isinstance(turn, (list, tuple)) and len(turn) > 1 and turn[1]:
            combined_texts.append(str(turn[1]))
    current_merged_txt = parse_chat_response_to_batch_manual("\n\n".join(combined_texts), txt_content)
    problems = find_problematic_lines(current_merged_txt)

    if problems:
        problem_count = len(problems)
        auto_fix_items = problems[:20]  # Giới hạn tối đa 20 câu lỗi mỗi đợt để phản hồi tức thì
        lines_to_fix = [f"{item['file']}: {item['idx']}: {item['orig']}" for item in auto_fix_items]

        fix_req_label = f"[Auto-Fix] Found {problem_count} untranslated or Han/Hangul lines"
        fix_prompt = (
            "Dưới đây là các câu thoại bị sót chữ Hán/Hàn hoặc chưa được dịch hoàn chỉnh:\n\n"
            + "\n".join(lines_to_fix) + "\n\n"
            "Hãy dịch lại hoàn toàn sạch sẽ sang tiếng Việt cho các câu trên.\n"
            "QUY TẮC QUAN TRỌNG:\n"
            "- BẮT BUỘC dịch sạch 100% chữ Hán/Nhật/Hàn, không để sót chữ nào, tuân thủ đúng bảng xưng hô, tiếng rên kéo dài viết dính liền.\n"
            "- TUYỆT ĐỐI BỎ QUA từ tượng thanh / cảm thán / rên rỉ tiếng Nhật (như ビクッ, ん, あっ, はぁ...), để trống sau dấu ||, KHÔNG dịch các từ này thành tiếng Việt.\n"
            "- Quy tắc font: chỉ dùng *in nghiêng* cho tiếng rên ngắn, không bọc * cả câu dài. Nếu có tiếng Anh thì giữ nguyên tiếng Anh.\n\n"
            "Định dạng trả về bắt buộc:\n"
            "[tên_file]:\n"
            "[số_dòng]: [tiếng gốc] || [bản dịch]\n\n"
            "Chỉ trả về các dòng sửa trên, không thêm lời giải thích thừa."
        )

        t_fix = time.time()
        clock_fix = time.strftime("%H:%M:%S")
        hist.append([fix_req_label, f"Sending AI fix request for {len(auto_fix_items)} lines... [Started at {clock_fix}]"])
        yield hist, "", f"Fixing {len(auto_fix_items)} lines on `{session.current_model}` (Started: {clock_fix})..."

        try:
            status_info, current_model, fix_model_text = session.send_message(fix_prompt)
            fix_elapsed = round(time.time() - t_fix, 1)
            hist[-1] = [fix_req_label, fix_model_text]
            yield hist, "", f"Fixed {len(auto_fix_items)} lines in {fix_elapsed}s. Model: `{current_model}` | {status_info}"
        except Exception as e:
            fix_elapsed = round(time.time() - t_fix, 1)
            hist[-1] = [fix_req_label, f"Error fixing lines after {fix_elapsed}s: {str(e)}"]

    yield hist, "", f"Finished translating all {len(file_list)} pages! Click 'Apply to Translation Box' to apply."


def handle_batch_chat_quick_check_role(current_history: List[Any], selected_model: str):
    prompt = (
        "Hãy kiểm tra và tóm tắt lại bảng quy ước xưng hô của các nhân vật đã xuất hiện trong truyện (Ai xưng gì với ai). "
        "Đánh giá xem các đoạn thoại vừa dịch có câu nào bị xưng hô nhầm ngôi hoặc không tự nhiên không?"
    )
    yield from handle_batch_chat_send(prompt, current_history, selected_model)


def handle_batch_chat_apply(current_history: List[Any], current_manual_txt: str):
    if not current_history:
        return current_manual_txt or "", "No chat history to apply."

    combined_model_text = []
    for turn in current_history:
        if isinstance(turn, (list, tuple)) and len(turn) > 1 and turn[1]:
            combined_model_text.append(str(turn[1]))

    all_ai_text = "\n\n".join(combined_model_text)
    updated_txt, status_md = parse_and_validate_batch_manual(
        all_ai_text, current_manual_txt or ""
    )
    return updated_txt, status_md


def handle_batch_chat_reset(selected_model: str):
    global BATCH_GEMINI_SESSION
    if BATCH_GEMINI_SESSION:
        BATCH_GEMINI_SESSION.reset()
        if selected_model:
            BATCH_GEMINI_SESSION.set_model(selected_model)
    model = selected_model or "gemini-3.8-flash"
    return [], f"**Active Model:** `{model}` (Session reset)"


