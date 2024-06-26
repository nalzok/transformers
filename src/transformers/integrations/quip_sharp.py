# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"QuIP# integration file"


from functools import partial
from ..utils import is_accelerate_available, is_quip_sharp_available, is_torch_available


if is_torch_available():
    import torch
    import torch.nn as nn


def replace_with_quip_sharp_linear(
    model,
    model_config,
    quantization_config,
    unquantized_linear_layers,
    _current_path=None,
):
    """
    Public method that recursively replaces the Linear layers of the given model with QuIP# quantized layers.
    `accelerate` is needed to use this method. Returns the converted model.

    Args:
        model (`torch.nn.Module`):
            The model to convert, can be any `torch.nn.Module` instance.
        model_config (`PretrainedConfig`):
            The config object that contains the model configurations.
        quantization_config (`QuIPSharpConfig`):
            The quantization config object that contains the quantization parameters.
        unquantized_linear_layers (`set[str]`):
            A set of nn.Linear weights to not convert. If a parameter path is in the set (e.g. `model.layers.31.post_attention_layernorm`), the corresponding module will not be converted.
        _current_path (`list`, *optional*):
            A list that contains the current path. This is used for DFS traversal and should not be passed by the user.
    """
    from ..models.llama.modeling_llama import LlamaAttention, LlamaMLP
    from ..models.mistral.modeling_mistral import MistralAttention, MistralMLP

    if not torch.cuda.is_available():
        raise NotImplementedError("At the time, QuIP# only supports GPU inference")

    if not is_accelerate_available():
        raise ImportError("Using `quip-sharp` quantization requires Accelerate: `pip install accelerate`")
    else:
        from accelerate import init_empty_weights

    if not is_quip_sharp_available():
        raise ImportError("Using `quip-sharp` quantization requires quip-sharp: `pip install quip-sharp`")
    else:
        from quip_sharp.lib.linear.quantized_linear import QuantizedLinear
        from quip_sharp.lib.linear.fused_quantized_linear import FusedQuantizedLinear

    if _current_path is None:
        _current_path = []

    to_purge = set()

    for name, module in model.named_children():
        _current_path.append(name)

        hidden_size = model_config.hidden_size
        num_heads = model_config.num_attention_heads
        head_dim = hidden_size // num_heads
        num_key_value_heads = model_config.num_key_value_heads
        intermediate_size = model_config.intermediate_size

        if isinstance(module, nn.Linear):
            full_name = ".".join(_current_path)
            if full_name in unquantized_linear_layers:
                pass
            elif model_config.model_type in {"llama", "mistral"} \
                    and name in {"q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"}:
                to_purge.add(name)
            else:
                with init_empty_weights():
                    model._modules[name] = QuantizedLinear(
                        in_features=module.in_features,
                        out_features=module.out_features,
                        codesz=quantization_config.codesz,
                        packsz=quantization_config.packsz,
                        pack_out=quantization_config.pack_out,
                        idx_dtype=quantization_config.idx_dtype,
                        codebook_version=quantization_config.codebook_version,
                        rank=quantization_config.lora_rank,
                        rescale_WH=quantization_config.rescale_WH,
                        resid_scale_override=quantization_config.resid_scale_override,
                        train_mode=quantization_config.get("train_mode", False),
                    )

                # Store the module class in case we need to transpose the weight later
                model._modules[name].source_cls = type(module)
                # Force requires grad to False to avoid unexpected errors
                model._modules[name].requires_grad_(False)

        elif isinstance(module, (LlamaAttention, MistralAttention)):
            module.qkv_proj = FusedQuantizedLinear(
                fuse_dim=-1,
                fuse_sizes=(num_heads*head_dim, num_key_value_heads*head_dim, num_key_value_heads*head_dim),
                in_features=hidden_size,
                out_features=(num_heads*head_dim) + (num_key_value_heads*head_dim) + (num_key_value_heads*head_dim),
                codesz=quantization_config.codesz,
                packsz=quantization_config.packsz,
                pack_out=quantization_config.pack_out,
                idx_dtype=quantization_config.idx_dtype,
                codebook_version=quantization_config.codebook_version,
                rank=quantization_config.lora_rank,
                rescale_WH=quantization_config.rescale_WH,
                resid_scale_override=quantization_config.resid_scale_override,
                train_mode=quantization_config.get('train_mode', False),
            )

        elif isinstance(module, (LlamaMLP, MistralMLP)):
            module.upgate_proj = FusedQuantizedLinear(
                fuse_dim=-1,
                fuse_sizes=(intermediate_size, intermediate_size),
                in_features=hidden_size,
                out_features=intermediate_size*2,
                codesz=quantization_config.codesz,
                packsz=quantization_config.packsz,
                pack_out=quantization_config.pack_out,
                idx_dtype=quantization_config.idx_dtype,
                codebook_version=quantization_config.codebook_version,
                rank=quantization_config.lora_rank,
                rescale_WH=quantization_config.rescale_WH,
                resid_scale_override=quantization_config.resid_scale_override,
                train_mode=quantization_config.get('train_mode', False),
            )

        if model._modules[name]._modules:
            replace_with_quip_sharp_linear(
                module,
                model_config,
                quantization_config,
                unquantized_linear_layers,
                _current_path,
            )

        _current_path.pop()

    for name in to_purge:
        del model._modules[name]

    return model



def unpack_quip_sharp_fused(model):
    """
    Public method that unpack the fused Linear layers for the given model quantized with QuIP#.
    Returns the converted model, or the original model when there is no fused quantized layers.

    Args:
        model (`torch.nn.Module`):
            The model to convert, can be any `torch.nn.Module` instance.
    """
    from ..models.llama.modeling_llama import LlamaAttention, LlamaMLP
    from ..models.mistral.modeling_mistral import MistralAttention, MistralMLP

    if not is_quip_sharp_available():
        raise ImportError("Using `quip-sharp` quantization requires quip-sharp: `pip install quip-sharp`")
    else:
        from quip_sharp.lib.linear.fused_quantized_linear import FusedQuantizedLinear
        from quip_sharp.lib.codebook import cache_permute_set

    for module in model.modules():
        if isinstance(module, FusedQuantizedLinear):
            if module.codebook_id in cache_permute_set:
                m, n = module.Qidxs.shape
                module.Qidxs.copy_(module.Qidxs
                                   .view(m, n // model.config.codesz, model.config.codesz)
                                   .permute(1, 0, 2)
                                   .reshape(m, n)
                                   .contiguous())

    for module in model.modules():
        if isinstance(module, (LlamaAttention, MistralAttention)):
            def forward_factory_attn():
                q_output = k_output = v_output = None

                def forward_q(*args, **kwargs):
                    nonlocal q_output, k_output, v_output
                    module = kwargs.pop("module")
                    if q_output is None:
                        q_output, k_output, v_output = module.qkv_proj.forward(*args, **kwargs)
                    rv, q_output = q_output, None
                    return rv

                def forward_k(*args, **kwargs):
                    nonlocal q_output, k_output, v_output
                    module = kwargs.pop("module")
                    if k_output is None:
                        q_output, k_output, v_output = module.qkv_proj.forward(*args, **kwargs)
                    rv, k_output = k_output, None
                    return rv

                def forward_v(*args, **kwargs):
                    nonlocal q_output, k_output, v_output
                    module = kwargs.pop("module")
                    if v_output is None:
                        q_output, k_output, v_output = module.qkv_proj.forward(*args, **kwargs)
                    rv, v_output = v_output, None
                    return rv

                return forward_q, forward_k, forward_v

            forward_q, forward_k, forward_v = forward_factory_attn()

            # https://stackoverflow.com/a/3431699
            setattr(module, "q_proj", partial(forward_q, module=module))
            setattr(module, "k_proj", partial(forward_k, module=module))
            setattr(module, "v_proj", partial(forward_v, module=module))

        elif isinstance(module, (LlamaMLP, MistralMLP)):
            def forward_factory_mlp():
                up_output = gate_output = None

                def forward_up(*args, **kwargs):
                    nonlocal up_output, gate_output
                    module = kwargs.pop("module")
                    if up_output is None:
                        up_output, gate_output = module.upgate_proj.forward(*args, **kwargs)
                    rv, up_output = up_output, None
                    return rv

                def forward_gate(*args, **kwargs):
                    nonlocal up_output, gate_output
                    module = kwargs.pop("module")
                    if gate_output is None:
                        up_output, gate_output = module.upgate_proj.forward(*args, **kwargs)
                    rv, gate_output = gate_output, None
                    return rv

                return forward_up, forward_gate

            forward_up, forward_gate = forward_factory_mlp()

            # https://stackoverflow.com/a/3431699
            setattr(module, "up_proj", partial(forward_up, module=module))
            setattr(module, "gate_proj", partial(forward_gate, module=module))
