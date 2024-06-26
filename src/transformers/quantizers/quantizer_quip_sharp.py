# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import importlib
from typing import TYPE_CHECKING, Optional

from packaging import version

from .base import HfQuantizer


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from ..integrations import replace_with_quip_sharp_linear, unpack_quip_sharp_fused
from ..utils import is_accelerate_available, is_quip_sharp_available, is_torch_available, logging
from ..utils.quantization_config import QuantizationConfigMixin


if is_torch_available():
    import torch

logger = logging.get_logger(__name__)


class QuipSharpHfQuantizer(HfQuantizer):
    """
    Quantizer of the QuIP# method. Enables the loading of prequantized models.
    """

    requires_calibration = True
    required_packages = ["quip-sharp"]
    optimum_quantizer = None

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def validate_environment(self, *args, **kwargs):
        if not torch.cuda.is_available():
            raise NotImplementedError("At the time, QuIP# only supports GPU inference")

        if not is_accelerate_available():
            raise ImportError("Using `quip-sharp` quantization requires Accelerate: `pip install accelerate`")

        if not is_quip_sharp_available():
            raise ImportError("Using `quip-sharp` quantization requires quip-sharp: `pip install quip-sharp`")

    def update_torch_dtype(self, torch_dtype: "torch.dtype") -> "torch.dtype":
        if torch_dtype is None:
            torch_dtype = torch.float16
            logger.info(
                "Assuming QuIP# inference on GPU and loading the model in `torch.float16`. To overwrite it, set `torch_dtype` manually."
            )
        return torch_dtype

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ):
        del kwargs

        model_type = model.config.model_type
        if model_type == "llama":
            unquantized_linear_layers = {"lm_head"}
        elif model_type == "mistral":
            unquantized_linear_layers = {"lm_head"}
        else:
            raise NotImplementedError(f"Unsupported model type {model_type}")

        replace_with_quip_sharp_linear(
            model,
            model.config,
            self.quantization_config,
            unquantized_linear_layers
        )

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        del kwargs
        unpack_quip_sharp_fused(model)
        return model

    @property
    def is_trainable(self, model: Optional["PreTrainedModel"] = None):
        return False

    @property
    def is_serializable(self):
        return True
