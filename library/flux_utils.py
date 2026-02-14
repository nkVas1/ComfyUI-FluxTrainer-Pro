from dataclasses import replace
import json
import os
import re
from typing import List, Optional, Tuple, Union
import einops
import torch
import torch.nn as nn

from safetensors import safe_open
from accelerate import init_empty_weights
from transformers import CLIPTextModel, CLIPConfig, T5EncoderModel, T5Config
try:
    from transformers import Qwen2Config, Qwen2Model
except Exception:
    Qwen2Config = None
    Qwen2Model = None

from .flux_models import Flux, AutoEncoder, configs
from .utils import setup_logging, load_safetensors

setup_logging()
import logging

logger = logging.getLogger(__name__)


class QwenLikeTextEncoderAdapter(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @property
    def dtype(self):
        param = next(self.parameters(), None)
        return param.dtype if param is not None else torch.float32

    @property
    def device(self):
        param = next(self.parameters(), None)
        return param.device if param is not None else torch.device("cpu")

    def forward(self, input_ids, attention_mask=None, return_dict=False, output_hidden_states=True):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=output_hidden_states,
            use_cache=False,
        )
        if return_dict:
            return outputs
        return outputs.last_hidden_state, None


def _infer_qwen2_config_from_state_dict(sd: dict) -> "Qwen2Config":
    embed_key = "model.embed_tokens.weight"
    if embed_key not in sd:
        raise ValueError("Qwen-like state dict missing model.embed_tokens.weight")

    vocab_size, hidden_size = sd[embed_key].shape

    layer_pattern = re.compile(r"^model\.layers\.(\d+)\.")
    layer_indices = []
    for key in sd.keys():
        match = layer_pattern.match(key)
        if match:
            layer_indices.append(int(match.group(1)))
    num_hidden_layers = max(layer_indices) + 1 if layer_indices else 32

    gate_proj_key = next((k for k in sd.keys() if k.endswith(".mlp.gate_proj.weight")), None)
    intermediate_size = sd[gate_proj_key].shape[0] if gate_proj_key else hidden_size * 4

    q_proj_key = next((k for k in sd.keys() if k.endswith(".self_attn.q_proj.weight")), None)
    k_proj_key = next((k for k in sd.keys() if k.endswith(".self_attn.k_proj.weight")), None)
    q_norm_key = next((k for k in sd.keys() if k.endswith(".self_attn.q_norm.weight")), None)

    q_out = sd[q_proj_key].shape[0] if q_proj_key else hidden_size
    k_out = sd[k_proj_key].shape[0] if k_proj_key else hidden_size
    head_dim = sd[q_norm_key].shape[0] if q_norm_key and sd[q_norm_key].ndim == 1 else 128
    if head_dim <= 0:
        head_dim = 128

    num_attention_heads = max(1, q_out // head_dim)
    num_key_value_heads = max(1, k_out // head_dim)

    return Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=32768,
        tie_word_embeddings=False,
        rms_norm_eps=1e-6,
    )


def _load_qwen_like_text_encoder(sd: dict) -> QwenLikeTextEncoderAdapter:
    if Qwen2Model is None or Qwen2Config is None:
        raise ValueError("transformers package in this environment does not provide Qwen2Model")

    config = _infer_qwen2_config_from_state_dict(sd)
    qwen = Qwen2Model(config)

    model_keys = set(qwen.state_dict().keys())

    filtered_sd = {}
    for key, value in sd.items():
        if key.startswith("lm_head."):
            continue
        if key.endswith(".weight_scale") or key.endswith(".weight_scale_2") or key.endswith(".comfy_quant"):
            continue

        normalized_key = key[6:] if key.startswith("model.") else key
        if normalized_key in model_keys:
            filtered_sd[normalized_key] = value

    info = qwen.load_state_dict(filtered_sd, strict=False, assign=True)
    logger.info(f"Loaded Qwen-like text encoder: {info}")

    qwen.eval()
    return QwenLikeTextEncoderAdapter(qwen)

# Model version constants
MODEL_VERSION_FLUX_V1 = "flux1"
MODEL_VERSION_FLUX_V2 = "flux2"
MODEL_NAME_DEV = "dev"
MODEL_NAME_SCHNELL = "schnell"
MODEL_NAME_FLUX2_KLEIN_9B = "flux2_klein_9b"
MODEL_NAME_FLUX2_DEV = "flux2_dev"

# bypass guidance
def bypass_flux_guidance(transformer):
    transformer.params.guidance_embed = False

# restore the forward function
def restore_flux_guidance(transformer):
    transformer.params.guidance_embed = True


def detect_flux_version(keys: List[str], num_double_blocks: int, num_single_blocks: int) -> Tuple[str, str]:
    """
    Определяет версию модели Flux на основе ключей и количества блоков.
    
    Returns:
        Tuple[str, str]: (model_version, model_name)
            - model_version: "flux1" или "flux2"
            - model_name: "dev", "schnell", "flux2_klein_9b", "flux2_dev"
    """
    # Проверяем наличие специфичных ключей Flux.2
    # Flux.2 может иметь другие ключи или структуру
    
    has_guidance = "guidance_in.in_layer.bias" in keys or "time_text_embed.guidance_embedder.linear_1.bias" in keys
    
    # Flux.2 Klein 9B: примерно 19 double blocks, 38 single blocks (как dev)
    # Flux.2 Dev: больше блоков (28 double, 56 single для 32B модели)
    
    # Эвристика для определения версии:
    # 1. Если блоков значительно больше стандартных flux1 (19/38) - это flux2_dev
    # 2. Если параметр hidden_size можно определить - используем его
    
    if num_double_blocks > 24 or num_single_blocks > 50:
        # Это вероятно Flux.2 Dev (32B модель)
        return MODEL_VERSION_FLUX_V2, MODEL_NAME_FLUX2_DEV
    
    # Проверяем по размеру hidden_size (если можно определить из ключей)
    # Для стандартных моделей используем старую логику
    if not has_guidance:
        return MODEL_VERSION_FLUX_V1, MODEL_NAME_SCHNELL
    
    # По умолчанию - flux1 dev (или flux2_klein который совместим)
    return MODEL_VERSION_FLUX_V1, MODEL_NAME_DEV
    

def analyze_checkpoint_state(ckpt_path: str) -> Tuple[bool, bool, Tuple[int, int], List[str]]:
    """
    チェックポイントの状態を分析し、DiffusersかBFLか、devかschnellか、ブロック数を計算して返す。
    Extended to support Flux.2 models.

    Args:
        ckpt_path (str): チェックポイントファイルまたはディレクトリのパス。

    Returns:
        Tuple[bool, bool, Tuple[int, int], List[str]]:
            - bool: Diffusersかどうかを示すフラグ。
            - bool: Schnellかどうかを示すフラグ。
            - Tuple[int, int]: ダブルブロックとシングルブロックの数。
            - List[str]: チェックポイントに含まれるキーのリスト。
    """
    # check the state dict: Diffusers or BFL, dev or schnell, number of blocks
    logger.info(f"Checking the state dict: Diffusers or BFL, dev or schnell, Flux version")

    if os.path.isdir(ckpt_path):  # if ckpt_path is a directory, it is Diffusers
        ckpt_path = os.path.join(ckpt_path, "transformer", "diffusion_pytorch_model-00001-of-00003.safetensors")
    if "00001-of-00003" in ckpt_path:
        ckpt_paths = [ckpt_path.replace("00001-of-00003", f"0000{i}-of-00003") for i in range(1, 4)]
    else:
        ckpt_paths = [ckpt_path]

    keys = []
    for ckpt_path in ckpt_paths:
        with safe_open(ckpt_path, framework="pt") as f:
            keys.extend(f.keys())

    if keys[0].startswith("model.diffusion_model."):
        keys = [key.replace("model.diffusion_model.", "") for key in keys]

    is_diffusers = any(k.startswith("transformer_blocks.") for k in keys)
    is_schnell = not (
        "guidance_in.in_layer.bias" in keys
        or "guidance_in.in_layer.weight" in keys
        or "time_text_embed.guidance_embedder.linear_1.bias" in keys
        or "time_text_embed.guidance_embedder.linear_1.weight" in keys
    )

    # check number of double and single blocks
    if not is_diffusers:
        double_indices = [
            int(key.split(".")[1])
            for key in keys
            if key.startswith("double_blocks.")
            and (
                key.endswith(".img_attn.proj.weight")
                or key.endswith(".img_attn.proj.bias")
                or key.endswith(".img_mlp.0.weight")
            )
        ]
        single_indices = [
            int(key.split(".")[1])
            for key in keys
            if key.startswith("single_blocks.")
            and (
                key.endswith(".linear1.weight")
                or key.endswith(".linear2.weight")
                or key.endswith(".modulation.lin.bias")
            )
        ]
    else:
        double_indices = [
            int(key.split(".")[1])
            for key in keys
            if key.startswith("transformer_blocks.")
            and (
                key.endswith(".attn.add_k_proj.weight")
                or key.endswith(".attn.add_k_proj.bias")
            )
        ]
        single_indices = [
            int(key.split(".")[1])
            for key in keys
            if key.startswith("single_transformer_blocks.")
            and (
                key.endswith(".attn.to_k.weight")
                or key.endswith(".attn.to_k.bias")
            )
        ]

    if not double_indices or not single_indices:
        logger.warning(
            f"Could not detect block structure from checkpoint keys. "
            f"double_blocks found: {len(double_indices)}, single_blocks found: {len(single_indices)}. "
            f"Falling back to default Flux Dev layout (19 double, 38 single)."
        )
        max_double_block_index = max(double_indices) if double_indices else 18  # default 19 blocks (0-18)
        max_single_block_index = max(single_indices) if single_indices else 37  # default 38 blocks (0-37)
    else:
        max_double_block_index = max(double_indices)
        max_single_block_index = max(single_indices)

    num_double_blocks = max_double_block_index + 1
    num_single_blocks = max_single_block_index + 1

    return is_diffusers, is_schnell, (num_double_blocks, num_single_blocks), ckpt_paths


def _infer_flux_params_from_state_dict(base_params, sd: dict):
    """Инференс ключевых архитектурных параметров Flux по формам тензоров в state_dict."""
    params = base_params

    img_in_weight = sd.get("img_in.weight")
    txt_in_weight = sd.get("txt_in.weight")
    ds_mlp_in_proj_weight = sd.get("double_blocks.0.img_mlp.0.weight")
    ds_mlp_out_proj_weight = sd.get("double_blocks.0.img_mlp.2.weight")
    ss_in_proj_weight = sd.get("single_blocks.0.linear1.weight")
    ss_out_proj_weight = sd.get("single_blocks.0.linear2.weight")

    inferred = {}

    if img_in_weight is not None and img_in_weight.ndim == 2:
        inferred_hidden_size = int(img_in_weight.shape[0])
        inferred_in_channels = int(img_in_weight.shape[1])
        inferred["hidden_size"] = inferred_hidden_size
        inferred["in_channels"] = inferred_in_channels

    if txt_in_weight is not None and txt_in_weight.ndim == 2:
        inferred["context_in_dim"] = int(txt_in_weight.shape[1])

    hidden_size = inferred.get("hidden_size", params.hidden_size)

    mlp_hidden_dim = None
    if ds_mlp_out_proj_weight is not None and ds_mlp_out_proj_weight.ndim == 2 and hidden_size > 0:
        # img_mlp.2: Linear(mlp_hidden_dim -> hidden_size), weight shape = [hidden_size, mlp_hidden_dim]
        mlp_hidden_dim = int(ds_mlp_out_proj_weight.shape[1])
    elif ss_out_proj_weight is not None and ss_out_proj_weight.ndim == 2 and hidden_size > 0:
        # single_blocks.linear2: Linear(hidden_size + mlp_hidden_dim -> hidden_size)
        candidate = int(ss_out_proj_weight.shape[1]) - int(hidden_size)
        if candidate > 0:
            mlp_hidden_dim = candidate

    if mlp_hidden_dim is not None and hidden_size > 0:
        inferred["mlp_ratio"] = float(mlp_hidden_dim) / float(hidden_size)

    # infer gated mlp architecture (GEGLU-like)
    gated_votes = []
    if mlp_hidden_dim is not None and mlp_hidden_dim > 0:
        if ds_mlp_in_proj_weight is not None and ds_mlp_in_proj_weight.ndim == 2:
            ds_in = int(ds_mlp_in_proj_weight.shape[0])
            if ds_in == mlp_hidden_dim * 2:
                gated_votes.append(True)
            elif ds_in == mlp_hidden_dim:
                gated_votes.append(False)

        if ss_in_proj_weight is not None and ss_in_proj_weight.ndim == 2:
            ss_in = int(ss_in_proj_weight.shape[0])
            if ss_in == (3 * hidden_size + 2 * mlp_hidden_dim):
                gated_votes.append(True)
            elif ss_in == (3 * hidden_size + mlp_hidden_dim):
                gated_votes.append(False)

    if gated_votes:
        inferred["mlp_gated"] = sum(1 for v in gated_votes if v) >= sum(1 for v in gated_votes if not v)

    # depth inference from block keys
    double_indices = [
        int(k.split(".")[1])
        for k in sd.keys()
        if k.startswith("double_blocks.")
        and (
            k.endswith(".img_attn.proj.weight")
            or k.endswith(".img_attn.proj.bias")
            or k.endswith(".img_mlp.0.weight")
        )
    ]
    single_indices = [
        int(k.split(".")[1])
        for k in sd.keys()
        if k.startswith("single_blocks.")
        and (
            k.endswith(".linear1.weight")
            or k.endswith(".linear2.weight")
            or k.endswith(".modulation.lin.bias")
        )
    ]
    if double_indices:
        inferred["depth"] = max(double_indices) + 1
    if single_indices:
        inferred["depth_single_blocks"] = max(single_indices) + 1

    # infer guidance support
    has_guidance = any(k.startswith("guidance_in.") for k in sd.keys())
    inferred["guidance_embed"] = has_guidance

    # keep RoPE axes consistent: sum(axes_dim)=hidden/num_heads
    axes_sum = sum(params.axes_dim)
    if hidden_size % axes_sum == 0:
        inferred["num_heads"] = hidden_size // axes_sum

    # apply inferred values only when valid
    if "num_heads" in inferred and inferred["num_heads"] <= 0:
        inferred.pop("num_heads", None)
    if "mlp_ratio" in inferred and inferred["mlp_ratio"] <= 0:
        inferred.pop("mlp_ratio", None)

    if inferred:
        logger.info(
            "Inferred Flux params from checkpoint: "
            f"in_channels={inferred.get('in_channels', params.in_channels)}, "
            f"context_in_dim={inferred.get('context_in_dim', params.context_in_dim)}, "
            f"hidden_size={inferred.get('hidden_size', params.hidden_size)}, "
            f"mlp_ratio={inferred.get('mlp_ratio', params.mlp_ratio):.3f}, "
            f"mlp_gated={inferred.get('mlp_gated', getattr(params, 'mlp_gated', False))}, "
            f"num_heads={inferred.get('num_heads', params.num_heads)}, "
            f"depth={inferred.get('depth', params.depth)}, "
            f"depth_single_blocks={inferred.get('depth_single_blocks', params.depth_single_blocks)}, "
            f"guidance_embed={inferred.get('guidance_embed', params.guidance_embed)}"
        )
        params = replace(params, **inferred)

    return params


def load_flow_model(
    ckpt_path: str, dtype: Optional[torch.dtype], device: Union[str, torch.device], disable_mmap: bool = False
) -> Tuple[bool, Flux]:
    is_diffusers, is_schnell, (num_double_blocks, num_single_blocks), ckpt_paths = analyze_checkpoint_state(ckpt_path)
    # load_sft doesn't support torch.device
    logger.info(f"Loading state dict from {ckpt_path}")
    sd = {}
    for ckpt_path in ckpt_paths:
        sd.update(load_safetensors(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype))

    # convert Diffusers to BFL
    if is_diffusers:
        logger.info("Converting Diffusers to BFL")
        sd = convert_diffusers_sd_to_bfl(sd, num_double_blocks, num_single_blocks)
        logger.info("Converted Diffusers to BFL")

    for key in list(sd.keys()):
        new_key = key.replace("model.diffusion_model.", "")
        if new_key == key:
            break
        sd[new_key] = sd.pop(key)

    # choose base config and infer params from checkpoint tensors
    if not is_schnell:
        base_name = MODEL_NAME_DEV
    else:
        base_name = MODEL_NAME_SCHNELL

    if "img_in.weight" in sd and sd["img_in.weight"].shape[0] >= 4096:
        base_name = MODEL_NAME_FLUX2_KLEIN_9B

    params = configs[base_name].params

    # set block counts from checkpoint analysis as initial values
    if params.depth != num_double_blocks:
        params = replace(params, depth=num_double_blocks)
    if params.depth_single_blocks != num_single_blocks:
        params = replace(params, depth_single_blocks=num_single_blocks)

    # infer architecture from state_dict to avoid size mismatch
    params = _infer_flux_params_from_state_dict(params, sd)

    # align is_schnell with inferred guidance support
    is_schnell = not params.guidance_embed

    def _try_build_and_load(candidate_params):
        logger.info(
            f"Building Flux model {base_name} from {'Diffusers' if is_diffusers else 'BFL'} checkpoint "
            f"(hidden={candidate_params.hidden_size}, in={candidate_params.in_channels}, "
            f"context={candidate_params.context_in_dim}, depth={candidate_params.depth}/{candidate_params.depth_single_blocks}, "
            f"heads={candidate_params.num_heads}, mlp_ratio={candidate_params.mlp_ratio}, "
            f"mlp_gated={getattr(candidate_params, 'mlp_gated', False)})"
        )
        with torch.device("meta"):
            candidate_model = Flux(candidate_params)
            if dtype is not None:
                candidate_model = candidate_model.to(dtype)
        candidate_info = candidate_model.load_state_dict(sd, strict=False, assign=True)
        return candidate_model, candidate_info

    candidates = [params]
    seen = {
        (
            params.hidden_size,
            params.in_channels,
            params.context_in_dim,
            params.depth,
            params.depth_single_blocks,
            params.num_heads,
            float(params.mlp_ratio),
            bool(getattr(params, "mlp_gated", False)),
        )
    }

    # Fallback candidates for inconsistent Flux.2 checkpoints
    for fallback in [
        replace(params, mlp_gated=not getattr(params, "mlp_gated", False)),
        replace(params, mlp_ratio=float(params.mlp_ratio) * 2.0, mlp_gated=False),
        replace(params, mlp_ratio=max(float(params.mlp_ratio) / 2.0, 0.5), mlp_gated=True),
    ]:
        key = (
            fallback.hidden_size,
            fallback.in_channels,
            fallback.context_in_dim,
            fallback.depth,
            fallback.depth_single_blocks,
            fallback.num_heads,
            float(fallback.mlp_ratio),
            bool(getattr(fallback, "mlp_gated", False)),
        )
        if key not in seen:
            seen.add(key)
            candidates.append(fallback)

    last_error = None
    for idx, candidate in enumerate(candidates):
        try:
            model, info = _try_build_and_load(candidate)
            if idx > 0:
                logger.warning(
                    "Loaded Flux after fallback candidate #%d (mlp_ratio=%s, mlp_gated=%s)",
                    idx,
                    candidate.mlp_ratio,
                    getattr(candidate, "mlp_gated", False),
                )
            logger.info(f"Loaded Flux: {info}")
            return is_schnell, model
        except RuntimeError as e:
            last_error = e
            logger.warning(
                "Flux candidate #%d failed (mlp_ratio=%s, mlp_gated=%s): %s",
                idx,
                candidate.mlp_ratio,
                getattr(candidate, "mlp_gated", False),
                str(e).splitlines()[0] if str(e) else "runtime error",
            )

    raise last_error if last_error is not None else RuntimeError("Failed to load Flux model from checkpoint")


def load_ae(
    ckpt_path: str, dtype: torch.dtype, device: Union[str, torch.device], disable_mmap: bool = False
) -> AutoEncoder:
    logger.info(f"Loading state dict from {ckpt_path}")
    sd = load_safetensors(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)

    base_params = configs[MODEL_NAME_DEV].ae_params
    inferred = {
        "ch": base_params.ch,
        "z_channels": base_params.z_channels,
    }

    encoder_conv_in = sd.get("encoder.conv_in.weight")
    if encoder_conv_in is not None and len(encoder_conv_in.shape) >= 1:
        inferred["ch"] = int(encoder_conv_in.shape[0])

    z_from_encoder = None
    enc_conv_out = sd.get("encoder.conv_out.weight")
    if enc_conv_out is not None and len(enc_conv_out.shape) >= 1:
        z_from_encoder = int(enc_conv_out.shape[0]) // 2

    z_from_decoder = None
    dec_conv_in = sd.get("decoder.conv_in.weight")
    if dec_conv_in is not None and len(dec_conv_in.shape) >= 2:
        z_from_decoder = int(dec_conv_in.shape[1])

    if z_from_encoder is not None and z_from_decoder is not None and z_from_encoder != z_from_decoder:
        logger.warning(
            "AutoEncoder z_channels mismatch in checkpoint (encoder=%s, decoder=%s). Using decoder value.",
            z_from_encoder,
            z_from_decoder,
        )

    inferred_z = z_from_decoder if z_from_decoder is not None else z_from_encoder
    if inferred_z is not None and inferred_z > 0:
        inferred["z_channels"] = inferred_z

    ae_params = replace(base_params, ch=inferred["ch"], z_channels=inferred["z_channels"])
    logger.info(
        "Building AutoEncoder (inferred): ch=%d, z_channels=%d, resolution=%d",
        ae_params.ch,
        ae_params.z_channels,
        ae_params.resolution,
    )
    with torch.device("meta"):
        ae = AutoEncoder(ae_params).to(dtype)

    info = ae.load_state_dict(sd, strict=False, assign=True)
    logger.info(f"Loaded AE: {info}")
    return ae


def load_clip_l(
    ckpt_path: Optional[str],
    dtype: torch.dtype,
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    state_dict: Optional[dict] = None,
) -> CLIPTextModel:
    logger.info("Building CLIP-L")
    CLIPL_CONFIG = {
        "_name_or_path": "clip-vit-large-patch14/",
        "architectures": ["CLIPModel"],
        "initializer_factor": 1.0,
        "logit_scale_init_value": 2.6592,
        "model_type": "clip",
        "projection_dim": 768,
        # "text_config": {
        "_name_or_path": "",
        "add_cross_attention": False,
        "architectures": None,
        "attention_dropout": 0.0,
        "bad_words_ids": None,
        "bos_token_id": 0,
        "chunk_size_feed_forward": 0,
        "cross_attention_hidden_size": None,
        "decoder_start_token_id": None,
        "diversity_penalty": 0.0,
        "do_sample": False,
        "dropout": 0.0,
        "early_stopping": False,
        "encoder_no_repeat_ngram_size": 0,
        "eos_token_id": 2,
        "finetuning_task": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "hidden_act": "quick_gelu",
        "hidden_size": 768,
        "id2label": {"0": "LABEL_0", "1": "LABEL_1"},
        "initializer_factor": 1.0,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "is_decoder": False,
        "is_encoder_decoder": False,
        "label2id": {"LABEL_0": 0, "LABEL_1": 1},
        "layer_norm_eps": 1e-05,
        "length_penalty": 1.0,
        "max_length": 20,
        "max_position_embeddings": 77,
        "min_length": 0,
        "model_type": "clip_text_model",
        "no_repeat_ngram_size": 0,
        "num_attention_heads": 12,
        "num_beam_groups": 1,
        "num_beams": 1,
        "num_hidden_layers": 12,
        "num_return_sequences": 1,
        "output_attentions": False,
        "output_hidden_states": False,
        "output_scores": False,
        "pad_token_id": 1,
        "prefix": None,
        "problem_type": None,
        "projection_dim": 768,
        "pruned_heads": {},
        "remove_invalid_values": False,
        "repetition_penalty": 1.0,
        "return_dict": True,
        "return_dict_in_generate": False,
        "sep_token_id": None,
        "task_specific_params": None,
        "temperature": 1.0,
        "tie_encoder_decoder": False,
        "tie_word_embeddings": True,
        "tokenizer_class": None,
        "top_k": 50,
        "top_p": 1.0,
        "torch_dtype": None,
        "torchscript": False,
        "transformers_version": "4.16.0.dev0",
        "use_bfloat16": False,
        "vocab_size": 49408,
        "hidden_act": "gelu",
        "hidden_size": 1280,
        "intermediate_size": 5120,
        "num_attention_heads": 20,
        "num_hidden_layers": 32,
        # },
        # "text_config_dict": {
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "projection_dim": 768,
        # },
        # "torch_dtype": "float32",
        # "transformers_version": None,
    }
    config = CLIPConfig(**CLIPL_CONFIG)
    with init_empty_weights():
        clip = CLIPTextModel._from_config(config)

    if state_dict is not None:
        sd = state_dict
    else:
        logger.info(f"Loading state dict from {ckpt_path}")
        sd = load_safetensors(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)
    info = clip.load_state_dict(sd, strict=False, assign=True)
    logger.info(f"Loaded CLIP-L: {info}")
    return clip


def load_t5xxl(
    ckpt_path: str,
    dtype: Optional[torch.dtype],
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    state_dict: Optional[dict] = None,
) -> T5EncoderModel:
    T5_CONFIG_JSON = """
{
  "architectures": [
    "T5EncoderModel"
  ],
  "classifier_dropout": 0.0,
  "d_ff": 10240,
  "d_kv": 64,
  "d_model": 4096,
  "decoder_start_token_id": 0,
  "dense_act_fn": "gelu_new",
  "dropout_rate": 0.1,
  "eos_token_id": 1,
  "feed_forward_proj": "gated-gelu",
  "initializer_factor": 1.0,
  "is_encoder_decoder": true,
  "is_gated_act": true,
  "layer_norm_epsilon": 1e-06,
  "model_type": "t5",
  "num_decoder_layers": 24,
  "num_heads": 64,
  "num_layers": 24,
  "output_past": true,
  "pad_token_id": 0,
  "relative_attention_max_distance": 128,
  "relative_attention_num_buckets": 32,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.41.2",
  "use_cache": true,
  "vocab_size": 32128
}
"""
    config = json.loads(T5_CONFIG_JSON)
    config = T5Config(**config)
    with init_empty_weights():
        t5xxl = T5EncoderModel._from_config(config)

    if state_dict is not None:
        sd = state_dict
    else:
        logger.info(f"Loading state dict from {ckpt_path}")
        sd = load_safetensors(ckpt_path, device=str(device), disable_mmap=disable_mmap, dtype=dtype)

    # Flux.2 Klein workflows in ComfyUI use Qwen text encoders.
    # Для совместимости не падаем здесь фатально, а продолжаем с подробным предупреждением.
    is_qwen_like = "model.embed_tokens.weight" in sd and "encoder.block.0.layer.0.SelfAttention.q.weight" not in sd
    if is_qwen_like:
        logger.warning(
            "Detected Qwen/Llama-style text encoder keys in %s. "
            "Continuing in compatibility mode for Flux.2 Klein. "
            "Recommended checkpoint for Klein 9B: qwen_3_8b_fp8mixed.safetensors",
            ckpt_path,
        )
        return _load_qwen_like_text_encoder(sd)

    info = t5xxl.load_state_dict(sd, strict=False, assign=True)
    logger.info(f"Loaded T5xxl: {info}")
    return t5xxl


def get_t5xxl_actual_dtype(t5xxl: T5EncoderModel) -> torch.dtype:
    if hasattr(t5xxl, "encoder") and hasattr(t5xxl.encoder, "block"):
        # nn.Embedding is the first layer, but it could be casted to bfloat16 or float32
        return t5xxl.encoder.block[0].layer[0].SelfAttention.q.weight.dtype

    first_param = next(t5xxl.parameters(), None)
    if first_param is not None:
        return first_param.dtype
    return torch.float32


def prepare_img_ids(batch_size: int, packed_latent_height: int, packed_latent_width: int):
    img_ids = torch.zeros(packed_latent_height, packed_latent_width, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(packed_latent_height)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(packed_latent_width)[None, :]
    img_ids = einops.repeat(img_ids, "h w c -> b (h w) c", b=batch_size)
    return img_ids


def unpack_latents(x: torch.Tensor, packed_latent_height: int, packed_latent_width: int) -> torch.Tensor:
    """
    x: [b (h w) (c ph pw)] -> [b c (h ph) (w pw)], ph=2, pw=2
    """
    x = einops.rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=packed_latent_height, w=packed_latent_width, ph=2, pw=2)
    return x


def pack_latents(x: torch.Tensor) -> torch.Tensor:
    """
    x: [b c (h ph) (w pw)] -> [b (h w) (c ph pw)], ph=2, pw=2
    """
    x = einops.rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    return x


# region Diffusers

NUM_DOUBLE_BLOCKS = 19
NUM_SINGLE_BLOCKS = 38

BFL_TO_DIFFUSERS_MAP = {
    "time_in.in_layer.weight": ["time_text_embed.timestep_embedder.linear_1.weight"],
    "time_in.in_layer.bias": ["time_text_embed.timestep_embedder.linear_1.bias"],
    "time_in.out_layer.weight": ["time_text_embed.timestep_embedder.linear_2.weight"],
    "time_in.out_layer.bias": ["time_text_embed.timestep_embedder.linear_2.bias"],
    "vector_in.in_layer.weight": ["time_text_embed.text_embedder.linear_1.weight"],
    "vector_in.in_layer.bias": ["time_text_embed.text_embedder.linear_1.bias"],
    "vector_in.out_layer.weight": ["time_text_embed.text_embedder.linear_2.weight"],
    "vector_in.out_layer.bias": ["time_text_embed.text_embedder.linear_2.bias"],
    "guidance_in.in_layer.weight": ["time_text_embed.guidance_embedder.linear_1.weight"],
    "guidance_in.in_layer.bias": ["time_text_embed.guidance_embedder.linear_1.bias"],
    "guidance_in.out_layer.weight": ["time_text_embed.guidance_embedder.linear_2.weight"],
    "guidance_in.out_layer.bias": ["time_text_embed.guidance_embedder.linear_2.bias"],
    "txt_in.weight": ["context_embedder.weight"],
    "txt_in.bias": ["context_embedder.bias"],
    "img_in.weight": ["x_embedder.weight"],
    "img_in.bias": ["x_embedder.bias"],
    "double_blocks.().img_mod.lin.weight": ["norm1.linear.weight"],
    "double_blocks.().img_mod.lin.bias": ["norm1.linear.bias"],
    "double_blocks.().txt_mod.lin.weight": ["norm1_context.linear.weight"],
    "double_blocks.().txt_mod.lin.bias": ["norm1_context.linear.bias"],
    "double_blocks.().img_attn.qkv.weight": ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"],
    "double_blocks.().img_attn.qkv.bias": ["attn.to_q.bias", "attn.to_k.bias", "attn.to_v.bias"],
    "double_blocks.().txt_attn.qkv.weight": ["attn.add_q_proj.weight", "attn.add_k_proj.weight", "attn.add_v_proj.weight"],
    "double_blocks.().txt_attn.qkv.bias": ["attn.add_q_proj.bias", "attn.add_k_proj.bias", "attn.add_v_proj.bias"],
    "double_blocks.().img_attn.norm.query_norm.scale": ["attn.norm_q.weight"],
    "double_blocks.().img_attn.norm.key_norm.scale": ["attn.norm_k.weight"],
    "double_blocks.().txt_attn.norm.query_norm.scale": ["attn.norm_added_q.weight"],
    "double_blocks.().txt_attn.norm.key_norm.scale": ["attn.norm_added_k.weight"],
    "double_blocks.().img_mlp.0.weight": ["ff.net.0.proj.weight"],
    "double_blocks.().img_mlp.0.bias": ["ff.net.0.proj.bias"],
    "double_blocks.().img_mlp.2.weight": ["ff.net.2.weight"],
    "double_blocks.().img_mlp.2.bias": ["ff.net.2.bias"],
    "double_blocks.().txt_mlp.0.weight": ["ff_context.net.0.proj.weight"],
    "double_blocks.().txt_mlp.0.bias": ["ff_context.net.0.proj.bias"],
    "double_blocks.().txt_mlp.2.weight": ["ff_context.net.2.weight"],
    "double_blocks.().txt_mlp.2.bias": ["ff_context.net.2.bias"],
    "double_blocks.().img_attn.proj.weight": ["attn.to_out.0.weight"],
    "double_blocks.().img_attn.proj.bias": ["attn.to_out.0.bias"],
    "double_blocks.().txt_attn.proj.weight": ["attn.to_add_out.weight"],
    "double_blocks.().txt_attn.proj.bias": ["attn.to_add_out.bias"],
    "single_blocks.().modulation.lin.weight": ["norm.linear.weight"],
    "single_blocks.().modulation.lin.bias": ["norm.linear.bias"],
    "single_blocks.().linear1.weight": ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight", "proj_mlp.weight"],
    "single_blocks.().linear1.bias": ["attn.to_q.bias", "attn.to_k.bias", "attn.to_v.bias", "proj_mlp.bias"],
    "single_blocks.().linear2.weight": ["proj_out.weight"],
    "single_blocks.().norm.query_norm.scale": ["attn.norm_q.weight"],
    "single_blocks.().norm.key_norm.scale": ["attn.norm_k.weight"],
    "single_blocks.().linear2.weight": ["proj_out.weight"],
    "single_blocks.().linear2.bias": ["proj_out.bias"],
    "final_layer.linear.weight": ["proj_out.weight"],
    "final_layer.linear.bias": ["proj_out.bias"],
    "final_layer.adaLN_modulation.1.weight": ["norm_out.linear.weight"],
    "final_layer.adaLN_modulation.1.bias": ["norm_out.linear.bias"],
}


def make_diffusers_to_bfl_map(num_double_blocks: int, num_single_blocks: int) -> dict[str, tuple[int, str]]:
    # make reverse map from diffusers map
    diffusers_to_bfl_map = {}  # key: diffusers_key, value: (index, bfl_key)
    for b in range(num_double_blocks):
        for key, weights in BFL_TO_DIFFUSERS_MAP.items():
            if key.startswith("double_blocks."):
                block_prefix = f"transformer_blocks.{b}."
                for i, weight in enumerate(weights):
                    diffusers_to_bfl_map[f"{block_prefix}{weight}"] = (i, key.replace("()", f"{b}"))
    for b in range(num_single_blocks):
        for key, weights in BFL_TO_DIFFUSERS_MAP.items():
            if key.startswith("single_blocks."):
                block_prefix = f"single_transformer_blocks.{b}."
                for i, weight in enumerate(weights):
                    diffusers_to_bfl_map[f"{block_prefix}{weight}"] = (i, key.replace("()", f"{b}"))
    for key, weights in BFL_TO_DIFFUSERS_MAP.items():
        if not (key.startswith("double_blocks.") or key.startswith("single_blocks.")):
            for i, weight in enumerate(weights):
                diffusers_to_bfl_map[weight] = (i, key)
    return diffusers_to_bfl_map


def convert_diffusers_sd_to_bfl(
    diffusers_sd: dict[str, torch.Tensor], num_double_blocks: int = NUM_DOUBLE_BLOCKS, num_single_blocks: int = NUM_SINGLE_BLOCKS
) -> dict[str, torch.Tensor]:
    diffusers_to_bfl_map = make_diffusers_to_bfl_map(num_double_blocks, num_single_blocks)

    # iterate over three safetensors files to reduce memory usage
    flux_sd = {}
    for diffusers_key, tensor in diffusers_sd.items():
        if diffusers_key in diffusers_to_bfl_map:
            index, bfl_key = diffusers_to_bfl_map[diffusers_key]
            if bfl_key not in flux_sd:
                flux_sd[bfl_key] = []
            flux_sd[bfl_key].append((index, tensor))
        else:
            logger.error(f"Error: Key not found in diffusers_to_bfl_map: {diffusers_key}")
            raise KeyError(f"Key not found in diffusers_to_bfl_map: {diffusers_key}")

    # concat tensors if multiple tensors are mapped to a single key, sort by index
    for key, values in flux_sd.items():
        if len(values) == 1:
            flux_sd[key] = values[0][1]
        else:
            flux_sd[key] = torch.cat([value[1] for value in sorted(values, key=lambda x: x[0])])

    # special case for final_layer.adaLN_modulation.1.weight and final_layer.adaLN_modulation.1.bias
    def swap_scale_shift(weight):
        shift, scale = weight.chunk(2, dim=0)
        new_weight = torch.cat([scale, shift], dim=0)
        return new_weight

    if "final_layer.adaLN_modulation.1.weight" in flux_sd:
        flux_sd["final_layer.adaLN_modulation.1.weight"] = swap_scale_shift(flux_sd["final_layer.adaLN_modulation.1.weight"])
    if "final_layer.adaLN_modulation.1.bias" in flux_sd:
        flux_sd["final_layer.adaLN_modulation.1.bias"] = swap_scale_shift(flux_sd["final_layer.adaLN_modulation.1.bias"])

    return flux_sd


# endregion