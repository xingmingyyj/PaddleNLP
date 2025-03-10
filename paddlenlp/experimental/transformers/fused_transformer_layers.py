# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import paddle
import paddle.distributed as dist
from paddle.framework import in_dynamic_mode
from paddle.incubate.nn.functional import (
    fused_bias_act,
    fused_layer_norm,
    fused_moe,
    fused_rms_norm,
    masked_multihead_attention,
    variable_length_memory_efficient_attention,
)
from paddle.nn import Layer
from paddle.nn.initializer import Constant
from paddle.nn.quant import weight_only_linear

from paddlenlp.utils.import_utils import is_paddlenlp_ops_available
from paddlenlp.utils.log import logger

if not is_paddlenlp_ops_available():
    logger.warning(
        "The paddlenlp_ops package is not installed. you can read the docs and install it by hand, "
        "you can refer to: https://github.com/PaddlePaddle/PaddleNLP/blob/develop/csrc/README.md"
    )

if (
    paddle.device.get_all_custom_device_type() is not None and len(paddle.device.get_all_custom_device_type()) > 0
) or paddle.is_compiled_with_cuda():
    from paddlenlp_ops import rebuild_padding_v2


def use_cutlass_fp8_gemm():
    return os.getenv("FLAGS_CUTLASS_FP8_GEMM", "False") in ["True", "1", "true"]


if paddle.is_compiled_with_cuda():
    if use_cutlass_fp8_gemm():
        logger.info("cutlass fp8 gemm is used. you can turn it off by setting FLAGS_CUTLASS_FP8_GEMM to False.")
        from paddlenlp_ops import (
            cutlass_fp8_fp8_fp8_dual_gemm_fused as fp8_dual_gemm_fused,
        )
        from paddlenlp_ops import cutlass_fp8_fp8_half_gemm_fused as fp8_gemm_fused
    else:
        from paddle.linalg import fp8_fp8_half_gemm_fused as fp8_gemm_fused
    try:
        from paddlenlp_ops import (
            dequant_int8,
            encode_rotary_qk,
            qkv_transpose_split,
            quant_int8,
            rebuild_padding,
            transpose_remove_padding,
            write_cache_kv,
        )

    except:
        pass

__all__ = [
    "MoeConfig",
    "MLAConfig",
    "FusedMultiTransformerConfig",
    "FusedMultiTransformerBase",
    "FusedMultiTransformerPostLayernorm",
    "FusedMultiTransformerWeightOnly",
    "FusedMultiTransformerWeightOnlyPostLayernorm",
    "FusedBlockMultiTransformer",
    "FusedBlockMultiTransformerWeightOnly",
    "FusedBlockMultiTransformerA8W8",
    "FusedBlockMultiTransformerFP8",
    "FusedBlockMultiTransformerFP8DynamicQuant",
]


# for distributed tensor model parallel
def _set_var_distributed(var):
    if var is None:
        return

    var.is_distributed = True

    if not in_dynamic_mode():
        # NOTE: use current_block and find_var_recursive to support while_loop
        startup_block = paddle.static.default_startup_program().current_block()
        main_block = paddle.static.default_main_program().current_block()
        startup_block._find_var_recursive(var.name).is_distributed = True
        main_block._find_var_recursive(var.name).is_distributed = True


@dataclass
class MoeConfig:
    num_experts: int = 0
    top_k: int = 0
    topk_method: Optional[str] = None
    num_expert_group: int = 1
    topk_group: Optional[int] = None
    norm_topk_prob: bool = True
    moe_every2: bool = False
    first_k_dense_replace: int = 0
    moe_intermediate_size: int = 0
    routed_scaling_factor: float = 1.0

    shared_expert_with_gate: bool = True

    shared_expert_intermediate_size: int = 0
    shared_expert_ffn1_weight_attrs: Optional[List[paddle.ParamAttr]] = None
    shared_expert_ffn1_weight_scale_attrs: Optional[List[paddle.ParamAttr]] = None
    shared_expert_ffn2_weight_attrs: Optional[List[paddle.ParamAttr]] = None
    shared_expert_ffn2_weight_scale_attrs: Optional[List[paddle.ParamAttr]] = None
    shared_expert_gate_weight_attrs: Optional[List[paddle.ParamAttr]] = None

    def has_moe(self) -> bool:
        return self.num_experts > 1

    def use_moe(self, i: int) -> bool:
        return (
            self.has_moe()
            and (self.moe_every2 is False or (self.moe_every2 and i % 2 == 1))
            and i >= self.first_k_dense_replace
        )

    def has_shared_expert(self) -> bool:
        return self.has_moe() and self.shared_expert_intermediate_size > 0

    def use_shared_expert(self, i: int) -> bool:
        return self.use_moe(i) and self.shared_expert_intermediate_size > 0


@dataclass
class AvxConfig:
    max_position_embeddings: int = 0
    compute_type: str = "fp16"
    cache_dtype: str = "fp16"


@dataclass
class SpeculateConfig:
    speculate_max_draft_token_num: int = 5
    speculate_method: str = None
    return_full_hidden_states: bool = False


@dataclass
class MLAConfig:
    use_matrix_absorption: bool = False

    q_lora_rank: int = None
    kv_lora_rank: int = None
    qk_nope_head_dim: int = None
    qk_rope_head_dim: int = None
    v_head_dim: int = None

    mscale: float = 1.0

    q_proj_weight_attrs: Optional[List[paddle.ParamAttr]] = None
    q_proj_weight_scale_attrs: Optional[List[paddle.ParamAttr]] = None

    q_a_proj_weight_attrs: Optional[List[paddle.ParamAttr]] = None
    q_a_proj_weight_scale_attrs: Optional[List[paddle.ParamAttr]] = None
    q_a_layernorm_weight_attrs: Optional[List[paddle.ParamAttr]] = None
    q_b_proj_weight_attrs: Optional[List[paddle.ParamAttr]] = None
    q_b_proj_weight_scale_attrs: Optional[List[paddle.ParamAttr]] = None
    kv_a_proj_with_mqa_weight_attrs: Optional[List[paddle.ParamAttr]] = None
    kv_a_proj_with_mqa_weight_scale_attrs: Optional[List[paddle.ParamAttr]] = None
    kv_a_layernorm_weight_attrs: Optional[List[paddle.ParamAttr]] = None
    kv_b_proj_weight_attrs: Optional[List[paddle.ParamAttr]] = None
    kv_b_proj_weight_scale_attrs: Optional[List[paddle.ParamAttr]] = None

    q_nope_k_b_proj_weight_attrs: Optional[List[paddle.ParamAttr]] = None
    q_nope_k_b_proj_weight_scale_attrs: Optional[List[paddle.ParamAttr]] = None
    q_rope_proj_weight_attrs: Optional[List[paddle.ParamAttr]] = None
    q_rope_proj_weight_scale_attrs: Optional[List[paddle.ParamAttr]] = None
    v_b_o_proj_weight_attrs: Optional[List[paddle.ParamAttr]] = None
    v_b_o_proj_weight_scale_attrs: Optional[List[paddle.ParamAttr]] = None

    def use_mla(self) -> bool:
        return self.kv_lora_rank is not None

    def use_absorb(self) -> bool:
        return self.use_mla() and self.use_matrix_absorption

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim


class FusedMultiTransformerConfig:
    def __init__(
        self,
        embed_dim,
        num_heads,
        intermediate_size,
        quant_type="",
        weight_block_size=[0, 0],
        moe_quant_type="",
        weightonly_group_size=-1,
        dropout_rate=0.0,
        activation="gelu",
        norm_type="layernorm",
        use_neox_rotary_style=False,
        rope_theta=10000.0,
        rotary_emb=None,
        normalize_before=True,
        ln_scale_attrs=None,
        ln_bias_attrs=None,
        qkv_weight_attrs=None,
        qkv_weight_scale_attrs=None,
        qkv_bias_attrs=None,
        linear_weight_attrs=None,
        linear_weight_scale_attrs=None,
        linear_bias_attrs=None,
        ffn_ln_scale_attrs=None,
        ffn_ln_bias_attrs=None,
        gate_weight_attrs=None,
        gate_bias_attrs=None,
        up_weight_attrs=None,
        up_bias_attrs=None,
        ffn1_weight_attrs=None,
        ffn1_weight_scale_attrs=None,
        ffn1_bias_attrs=None,
        ffn1_0_weight_attrs=None,
        ffn1_1_weight_attrs=None,
        ffn1_0_bias_attrs=None,
        ffn1_1_bias_attrs=None,
        ffn2_weight_attrs=None,
        ffn2_weight_scale_attrs=None,
        ffn2_bias_attrs=None,
        e_score_correction_bias_attrs=None,
        qkv_out_scale_attrs=None,
        linear_out_scale_attrs=None,
        ffn1_out_scale_attrs=None,
        ffn2_out_scale_attrs=None,
        linear_shift_attrs=None,
        linear_smooth_attrs=None,
        ffn2_shift_attrs=None,
        ffn2_smooth_attrs=None,
        cache_k_scale_attrs=None,
        cache_v_scale_attrs=None,
        cache_k_out_scale_attrs=None,
        cache_v_out_scale_attrs=None,
        quant_round_type=0,
        quant_max_bound=127.0,
        quant_min_bound=-127.0,
        epsilon=1e-5,
        residual_alpha=1.0,
        num_layers=-1,
        nranks=1,
        trans_qkvw=True,
        ring_id=-1,
        kv_num_heads=-1,
        cachekv_int8_type=None,
        rank_id=-1,
        append_attn=False,
        moe_config=MoeConfig(),
        avx_config=AvxConfig(),
        speculate_config=SpeculateConfig(),
        mla_config=MLAConfig(),
    ):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if kv_num_heads > 0:
            self.kv_num_heads = kv_num_heads
        else:
            self.kv_num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.norm_type = norm_type
        self.rope_theta = rope_theta

        self.rotary_emb = rotary_emb

        self.use_neox_rotary_style = use_neox_rotary_style
        self.normalize_before = normalize_before
        self.ln_scale_attrs = ln_scale_attrs
        self.ln_bias_attrs = ln_bias_attrs
        self.qkv_weight_attrs = qkv_weight_attrs
        self.qkv_weight_scale_attrs = qkv_weight_scale_attrs
        self.qkv_bias_attrs = qkv_bias_attrs
        self.linear_weight_attrs = linear_weight_attrs
        self.linear_weight_scale_attrs = linear_weight_scale_attrs
        self.linear_bias_attrs = linear_bias_attrs
        self.ffn_ln_scale_attrs = ffn_ln_scale_attrs
        self.ffn_ln_bias_attrs = ffn_ln_bias_attrs
        self.gate_weight_attrs = gate_weight_attrs
        self.gate_bias_attrs = gate_bias_attrs
        self.up_weight_attrs = up_weight_attrs
        self.up_bias_attrs = up_bias_attrs
        self.ffn1_weight_attrs = ffn1_weight_attrs
        self.ffn1_weight_scale_attrs = ffn1_weight_scale_attrs
        self.ffn1_bias_attrs = ffn1_bias_attrs

        # FP8 attrs
        self.ffn1_0_weight_attrs = ffn1_0_weight_attrs
        self.ffn1_1_weight_attrs = ffn1_1_weight_attrs
        self.ffn1_0_bias_attrs = ffn1_0_bias_attrs
        self.ffn1_1_bias_attrs = ffn1_1_bias_attrs

        self.ffn2_weight_attrs = ffn2_weight_attrs
        self.ffn2_weight_scale_attrs = ffn2_weight_scale_attrs
        self.ffn2_bias_attrs = ffn2_bias_attrs

        self.e_score_correction_bias_attrs = e_score_correction_bias_attrs

        self.qkv_out_scale_attrs = qkv_out_scale_attrs
        self.linear_out_scale_attrs = linear_out_scale_attrs
        self.ffn1_out_scale_attrs = ffn1_out_scale_attrs
        self.ffn2_out_scale_attrs = ffn2_out_scale_attrs
        self.linear_shift_attrs = linear_shift_attrs
        self.linear_smooth_attrs = linear_smooth_attrs
        self.ffn2_shift_attrs = ffn2_shift_attrs
        self.ffn2_smooth_attrs = ffn2_smooth_attrs
        self.cache_k_scale_attrs = cache_k_scale_attrs
        self.cache_v_scale_attrs = cache_v_scale_attrs
        self.cache_k_out_scale_attrs = cache_k_out_scale_attrs
        self.cache_v_out_scale_attrs = cache_v_out_scale_attrs

        self.quant_type = quant_type
        self.weight_block_size = weight_block_size
        self.moe_quant_type = moe_quant_type
        self.weightonly_group_size = weightonly_group_size
        self.quant_round_type = quant_round_type
        self.quant_max_bound = quant_max_bound
        self.quant_min_bound = quant_min_bound
        if "fp8" in self.quant_type:
            self.quant_max_bound = 448.0
            self.quant_min_bound = -448.0

        self.cachekv_int8_type = cachekv_int8_type

        self.epsilon = epsilon
        self.residual_alpha = residual_alpha
        self.num_layers = num_layers
        self.nranks = nranks
        self.rank_id = rank_id
        self.trans_qkvw = trans_qkvw
        self.ring_id = ring_id

        self.append_attn = append_attn

        self.moe_config = moe_config
        self.avx_config = avx_config
        self.speculate_config = speculate_config
        self.mla_config = mla_config


class FusedMultiTransformerBase(Layer):
    def __init__(self, config: FusedMultiTransformerConfig):
        super().__init__()

        self.config = config
        self.moe_quant_type = config.moe_quant_type

        assert config.embed_dim > 0, "Expected embed_dim to be greater than 0, " "but received {}".format(
            config.embed_dim
        )
        assert config.num_heads > 0, "Expected nhead to be greater than 0, " "but received {}".format(config.num_heads)
        assert config.intermediate_size > 0, "Expected intermediate_size to be greater than 0, but received {}".format(
            config.intermediate_size
        )

        # self.normalize_before = normalize_before
        self._dtype = self._helper.get_default_dtype()
        if self._dtype == "bfloat16":
            self._fuse_kernel_compute_dtype = "bf16"
        elif self._dtype == "float16":
            self._fuse_kernel_compute_dtype = "fp16"
        elif self._dtype == "float32":
            self._fuse_kernel_compute_dtype = "fp32"
        else:
            raise ValueError(
                "FusedMultiTransformer just support float32, float16 and bfloat16 as default dtype, but received {}".format(
                    self._dtype
                )
            )
        self._epsilon = config.epsilon
        self._residual_alpha = config.residual_alpha
        self.nranks = config.nranks
        self.norm_type = config.norm_type
        if self.norm_type == "layernorm":
            self.norm_func = fused_layer_norm
        elif self.norm_type == "rmsnorm":
            self.norm_func = fused_rms_norm
        else:
            raise NotImplementedError("Only support norm type of [layernorm, rmsnorm]")
        self.use_neox_rotary_style = config.use_neox_rotary_style
        self._norm_weight_dtype = "float32" if self.norm_type == "layernorm" else self._dtype

        self.activation = config.activation

        self.embed_dim = config.embed_dim
        if config.mla_config.use_mla():
            self.head_dim = config.mla_config.v_head_dim
        else:
            self.head_dim = config.embed_dim // config.num_heads
            assert self.head_dim * config.num_heads == config.embed_dim, "embed_dim must be divisible by num_heads"

        # tensor model parallel
        if config.nranks > 1:
            assert config.ring_id != -1
        assert config.num_heads % config.nranks == 0
        assert config.intermediate_size % config.nranks == 0
        assert config.moe_config.shared_expert_intermediate_size % config.nranks == 0
        assert config.moe_config.moe_intermediate_size % config.nranks == 0
        self.num_heads = config.num_heads // config.nranks
        self.kv_num_heads = config.kv_num_heads // config.nranks
        self.intermediate_size = config.intermediate_size // config.nranks
        self.config.moe_config.shared_expert_intermediate_size //= config.nranks
        self.config.moe_config.moe_intermediate_size //= config.nranks

        self.num_layers = config.num_layers
        assert self.num_layers > 0
        if config.qkv_weight_attrs is not None and isinstance(config.qkv_weight_attrs, (list, tuple)):
            assert self.num_layers == len(config.qkv_weight_attrs)

        if self.config.mla_config.use_mla():
            mscale = self.config.mla_config.mscale
            self.softmax_scale = float(self.config.mla_config.qk_head_dim**-0.5) * mscale * mscale
        else:
            self.softmax_scale = float(self.head_dim**-0.5)

        self.position_ids: list[int] = []

        self.weight_dtype = self._dtype
        self.create_params_type = self.get_weight_create_dype()

        self.ln_scales, self.ln_biases = [], []
        self.qkv_biases = []
        self.linear_biases = []
        self.ffn_ln_scales, self.ffn_ln_biases = [], []
        self.ffn1_biases = []
        self.ffn2_biases = []
        self.e_score_correction_biases = []

        self.shared_expert_gate_weights = []
        self.shared_expert_ffn1_weights = []
        self.shared_expert_ffn2_weights = []

        self.cache_k_scales, self.cache_v_scales = [], []
        self.cache_k_out_scales, self.cache_v_out_scales = [], []

        self.init_weight_shape(config)

        for i in range(self.num_layers):
            ln_scale_attr = self.get_attr(config.ln_scale_attrs, i)
            ln_bias_attr = self.get_attr(config.ln_bias_attrs, i)

            qkv_bias_attr = self.get_attr(config.qkv_bias_attrs, i)
            linear_bias_attr = self.get_attr(config.linear_bias_attrs, i)

            ffn_ln_scale_attr = self.get_attr(config.ffn_ln_scale_attrs, i)
            ffn_ln_bias_attr = self.get_attr(config.ffn_ln_bias_attrs, i)
            ffn1_bias_attr = self.get_attr(config.ffn1_bias_attrs, i)
            ffn2_bias_attr = self.get_attr(config.ffn2_bias_attrs, i)
            e_score_correction_bias_attr = self.get_attr(config.e_score_correction_bias_attrs, i)

            cache_k_scale_attr = self.get_attr(config.cache_k_scale_attrs, i)
            cache_v_scale_attr = self.get_attr(config.cache_v_scale_attrs, i)
            cache_k_out_scale_attr = self.get_attr(config.cache_k_out_scale_attrs, i)
            cache_v_out_scale_attr = self.get_attr(config.cache_v_out_scale_attrs, i)

            ln_scale = self.create_parameter(
                attr=ln_scale_attr,
                shape=[config.embed_dim],
                default_initializer=Constant(value=1.0),
                dtype=self._norm_weight_dtype,
            )
            ln_bias = None
            if ln_bias_attr:
                ln_bias = self.create_parameter(
                    attr=ln_bias_attr,
                    shape=[config.embed_dim],
                    is_bias=True,
                    dtype=self._norm_weight_dtype,
                )

            qkv_bias = None
            if qkv_bias_attr:
                qkv_bias = self.create_parameter(
                    shape=[(self.num_heads + 2 * self.kv_num_heads) * self.head_dim],
                    attr=qkv_bias_attr,
                    dtype=self._dtype,
                    is_bias=True,
                )

            linear_bias = None
            if linear_bias_attr:
                linear_bias = self.create_parameter(
                    shape=[config.embed_dim],
                    attr=linear_bias_attr,
                    dtype=self._dtype,
                    is_bias=True,
                )

            ffn_ln_scale = self.create_parameter(
                shape=[config.embed_dim],
                attr=ffn_ln_scale_attr,
                is_bias=False,
                default_initializer=Constant(1.0),
                dtype=self._norm_weight_dtype,
            )

            ffn_ln_bias = None
            if ffn_ln_bias_attr:
                ffn_ln_bias = self.create_parameter(
                    shape=[config.embed_dim],
                    attr=ffn_ln_bias_attr,
                    is_bias=True,
                    dtype=self._norm_weight_dtype,
                )

            ffn1_bias = None
            if ffn1_bias_attr:
                if self.config.moe_config.use_moe(i):
                    ffn1_bias = self.create_parameter(
                        shape=[self.config.moe_config.num_experts, self.intermediate_size * 2]
                        if self.activation.endswith("glu")
                        else [self.config.moe_config.num_experts, self.intermediate_size],
                        attr=ffn1_bias_attr,
                        dtype=self._dtype,
                        is_bias=True,
                    )
                else:
                    ffn1_bias = self.create_parameter(
                        shape=[self.intermediate_size * 2]
                        if self.activation.endswith("glu")
                        else [self.intermediate_size],
                        attr=ffn1_bias_attr,
                        dtype=self._dtype,
                        is_bias=True,
                    )

            e_score_correction_bias = None
            if e_score_correction_bias_attr:
                if self.config.moe_config.use_moe(i):
                    if self.config.moe_config.topk_method == "noaux_tc":
                        e_score_correction_bias = self.create_parameter(
                            shape=[self.config.moe_config.num_experts],
                            attr=e_score_correction_bias_attr,
                            dtype="float32",
                            is_bias=True,
                        )

            ffn2_bias = None
            if ffn2_bias_attr:
                if self.config.moe_config.use_moe(i):
                    ffn2_bias = self.create_parameter(
                        shape=[self.config.moe_config.num_experts, config.embed_dim],
                        attr=ffn2_bias_attr,
                        dtype=self._dtype,
                        is_bias=True,
                    )
                else:
                    ffn2_bias = self.create_parameter(
                        shape=[config.embed_dim],
                        attr=ffn2_bias_attr,
                        dtype=self._dtype,
                        is_bias=True,
                    )

            cache_scale_dtype = "float32"
            if self.config.append_attn:
                cache_scale_dtype = self._dtype

            cache_k_scale = None
            if cache_k_scale_attr:
                cache_k_scale = self.create_parameter(
                    shape=[self.kv_num_heads],
                    attr=cache_k_scale_attr,
                    dtype=cache_scale_dtype,
                    is_bias=False,
                )

            cache_v_scale = None
            if cache_v_scale_attr:
                cache_v_scale = self.create_parameter(
                    shape=[self.kv_num_heads],
                    attr=cache_v_scale_attr,
                    dtype=cache_scale_dtype,
                    is_bias=False,
                )

            cache_k_out_scale = None
            if cache_k_out_scale_attr:
                cache_k_out_scale = self.create_parameter(
                    shape=[self.kv_num_heads],
                    attr=cache_k_out_scale_attr,
                    dtype=cache_scale_dtype,
                    is_bias=False,
                )

            cache_v_out_scale = None
            if cache_v_out_scale_attr:
                cache_v_out_scale = self.create_parameter(
                    shape=[self.kv_num_heads],
                    attr=cache_v_out_scale_attr,
                    dtype=cache_scale_dtype,
                    is_bias=False,
                )

            # tensor model parallel
            if config.nranks > 1:
                # column parallel
                _set_var_distributed(qkv_bias)
                _set_var_distributed(ffn1_bias)

            self.ln_scales.append(ln_scale)
            self.ln_biases.append(ln_bias)
            self.qkv_biases.append(qkv_bias)
            self.linear_biases.append(linear_bias)

            self.ffn_ln_scales.append(ffn_ln_scale)
            self.ffn_ln_biases.append(ffn_ln_bias)
            self.ffn1_biases.append(ffn1_bias)
            self.ffn2_biases.append(ffn2_bias)
            self.e_score_correction_biases.append(e_score_correction_bias)

            self.cache_k_scales.append(cache_k_scale)
            self.cache_v_scales.append(cache_v_scale)
            self.cache_k_out_scales.append(cache_k_out_scale)
            self.cache_v_out_scales.append(cache_v_out_scale)

            self._add_parameter(ln_scale)
            self._add_parameter(ln_bias)
            self._add_parameter(qkv_bias)
            self._add_parameter(linear_bias)

            self._add_parameter(ffn_ln_scale)
            self._add_parameter(ffn_ln_bias)
            self._add_parameter(ffn1_bias)
            self._add_parameter(ffn2_bias)
            self._add_parameter(e_score_correction_bias)

            self._add_parameter(cache_k_scale)
            self._add_parameter(cache_v_scale)
            self._add_parameter(cache_k_out_scale)
            self._add_parameter(cache_v_out_scale)

        self.dropout_rate = config.dropout_rate

    def init_weight(self):
        self.qkv_weights = []
        self.linear_weights = []
        self.gate_weights = []
        self.ffn1_weights = []
        self.ffn2_weights = []

        self.q_proj_weights = []
        self.q_a_proj_weights = []
        self.q_a_layernorm_weights = []
        self.q_b_proj_weights = []
        self.kv_a_proj_with_mqa_weights = []
        self.kv_a_layernorm_weights = []
        self.kv_b_proj_weights = []
        self.q_nope_k_b_proj_weights = []
        self.q_rope_proj_weights = []
        self.v_b_o_proj_weights = []

        for i in range(self.num_layers):
            q_proj_weight = None
            q_a_proj_weight = None
            q_a_layernorm_weight = None
            q_b_proj_weight = None
            kv_a_proj_with_mqa_weight = None
            kv_a_layernorm_weight = None
            kv_b_proj_weight = None

            q_nope_k_b_proj_weight = None
            q_rope_proj_weight = None
            v_b_o_proj_weight = None
            if self.config.mla_config.use_mla():
                q_proj_weight_attr = self.get_attr(self.config.mla_config.q_proj_weight_attrs, i)
                q_a_proj_weight_attr = self.get_attr(self.config.mla_config.q_a_proj_weight_attrs, i)
                q_a_layernorm_weight_attr = self.get_attr(self.config.mla_config.q_a_layernorm_weight_attrs, i)
                q_b_proj_weight_attr = self.get_attr(self.config.mla_config.q_b_proj_weight_attrs, i)
                if q_proj_weight_attr:
                    q_proj_weight = self.create_parameter(
                        shape=self.q_proj_weight_shape,
                        attr=q_proj_weight_attr,
                        dtype=self.create_params_type,
                        is_bias=False,
                    )
                if q_a_proj_weight_attr:
                    q_a_proj_weight = self.create_parameter(
                        shape=self.q_a_proj_weight_shape,
                        attr=q_a_proj_weight_attr,
                        dtype=self.create_params_type,
                        is_bias=False,
                    )
                if q_a_layernorm_weight_attr:
                    q_a_layernorm_weight = self.create_parameter(
                        shape=[self.config.mla_config.q_lora_rank],
                        attr=q_a_layernorm_weight_attr,
                        dtype=self._norm_weight_dtype,
                        is_bias=False,
                    )
                if q_b_proj_weight_attr:
                    q_b_proj_weight = self.create_parameter(
                        shape=self.q_b_proj_weight_shape,
                        attr=q_b_proj_weight_attr,
                        dtype=self.create_params_type,
                        is_bias=False,
                    )

                kv_a_proj_with_mqa_weight_attr = self.get_attr(
                    self.config.mla_config.kv_a_proj_with_mqa_weight_attrs, i
                )
                kv_a_layernorm_weight_attr = self.get_attr(self.config.mla_config.kv_a_layernorm_weight_attrs, i)
                kv_b_proj_weight_attr = self.get_attr(self.config.mla_config.kv_b_proj_weight_attrs, i)
                if kv_a_proj_with_mqa_weight_attr:
                    kv_a_proj_with_mqa_weight = self.create_parameter(
                        shape=self.kv_a_proj_with_mqa_weight_shape,
                        attr=kv_a_proj_with_mqa_weight_attr,
                        dtype=self.create_params_type,
                        is_bias=False,
                    )
                if kv_a_layernorm_weight_attr:
                    kv_a_layernorm_weight = self.create_parameter(
                        shape=[self.config.mla_config.kv_lora_rank],
                        attr=kv_a_layernorm_weight_attr,
                        dtype=self._norm_weight_dtype,
                        is_bias=False,
                    )
                if kv_b_proj_weight_attr:
                    kv_b_proj_weight = self.create_parameter(
                        shape=self.kv_b_proj_weight_shape,
                        attr=kv_b_proj_weight_attr,
                        dtype=self.create_params_type,
                        is_bias=False,
                    )

                q_nope_k_b_proj_weight_attr = self.get_attr(self.config.mla_config.q_nope_k_b_proj_weight_attrs, i)
                q_rope_proj_weight_attr = self.get_attr(self.config.mla_config.q_rope_proj_weight_attrs, i)
                v_b_o_proj_weight_attr = self.get_attr(self.config.mla_config.v_b_o_proj_weight_attrs, i)
                if q_nope_k_b_proj_weight_attr:
                    q_nope_k_b_proj_weight = self.create_parameter(
                        shape=self.q_nope_k_b_proj_weight_shape,
                        attr=q_nope_k_b_proj_weight_attr,
                        dtype=self.create_params_type,
                        is_bias=False,
                    )

                if q_rope_proj_weight_attr:
                    q_rope_proj_weight = self.create_parameter(
                        shape=self.q_rope_proj_weight_shape,
                        attr=q_rope_proj_weight_attr,
                        dtype=self.create_params_type,
                        is_bias=False,
                    )

                if v_b_o_proj_weight_attr:
                    v_b_o_proj_weight = self.create_parameter(
                        shape=self.v_b_o_proj_weight_shape,
                        attr=v_b_o_proj_weight_attr,
                        dtype=self.create_params_type,
                        is_bias=False,
                    )

            qkv_weight = None
            qkv_weight_attr = self.get_attr(self.config.qkv_weight_attrs, i)
            if qkv_weight_attr:
                qkv_weight = self.create_parameter(
                    shape=self.qkv_weight_shape,
                    attr=qkv_weight_attr,
                    dtype=self.create_params_type,
                    is_bias=False,
                )

            linear_weight = None
            linear_weight_attr = self.get_attr(self.config.linear_weight_attrs, i)
            if linear_weight_attr:
                linear_weight = self.create_parameter(
                    shape=self.linear_weight_shape,
                    attr=linear_weight_attr,
                    dtype=self.create_params_type,
                    is_bias=False,
                )

            gate_weight = None
            gate_weight_attr = self.get_attr(self.config.gate_weight_attrs, i)
            if self.config.moe_config.use_moe(i):
                gate_weight = self.create_parameter(
                    shape=[self.config.embed_dim, self.config.moe_config.num_experts],
                    attr=gate_weight_attr,
                    dtype="float32",
                    is_bias=False,
                    default_initializer=paddle.nn.initializer.Constant(0),
                )

            ffn1_weight = None
            ffn2_weight = None
            ffn1_weight_attr = self.get_attr(self.config.ffn1_weight_attrs, i)
            ffn2_weight_attr = self.get_attr(self.config.ffn2_weight_attrs, i)
            if self.config.moe_config.use_moe(i):
                ffn1_weight = self.create_parameter(
                    shape=self.moe_ffn1_weight_shape,
                    attr=ffn1_weight_attr,
                    dtype=self.create_params_type,
                    is_bias=False,
                )
                ffn2_weight = self.create_parameter(
                    shape=self.moe_ffn2_weight_shape,
                    attr=ffn2_weight_attr,
                    dtype=self.create_params_type,
                    is_bias=False,
                )
            else:
                ffn1_weight = self.create_parameter(
                    shape=self.ffn1_weight_shape,
                    attr=ffn1_weight_attr,
                    dtype=self.create_params_type,
                    is_bias=False,
                )
                ffn2_weight = self.create_parameter(
                    shape=self.ffn2_weight_shape,
                    attr=ffn2_weight_attr,
                    dtype=self.create_params_type,
                    is_bias=False,
                )

            shared_expert_ffn1_weight = None
            shared_expert_ffn2_weight = None
            shared_expert_gate_weight = None
            if self.config.moe_config.use_shared_expert(i):
                if self.config.moe_config.shared_expert_with_gate:
                    shared_expert_gate_weight_attr = self.get_attr(
                        self.config.moe_config.shared_expert_gate_weight_attrs, i
                    )
                shared_expert_ffn1_weight_attr = self.get_attr(
                    self.config.moe_config.shared_expert_ffn1_weight_attrs, i
                )
                shared_expert_ffn2_weight_attr = self.get_attr(
                    self.config.moe_config.shared_expert_ffn2_weight_attrs, i
                )

                shared_expert_ffn1_weight = self.create_parameter(
                    shape=self.shared_expert_ffn1_weight_shape,
                    attr=shared_expert_ffn1_weight_attr,
                    dtype=self.create_params_type,
                )
                shared_expert_ffn2_weight = self.create_parameter(
                    shape=self.shared_expert_ffn2_weight_shape,
                    attr=shared_expert_ffn2_weight_attr,
                    dtype=self.create_params_type,
                )
                if self.config.moe_config.shared_expert_with_gate:
                    shared_expert_gate_weight = self.create_parameter(
                        shape=self.shared_expert_gate_weight_shape,
                        attr=shared_expert_gate_weight_attr,
                        dtype=self._helper.get_default_dtype(),
                    )

            # tensor model parallel
            if self.config.nranks > 1:
                # column parallel
                _set_var_distributed(qkv_weight)
                _set_var_distributed(q_proj_weight)
                _set_var_distributed(q_b_proj_weight)
                _set_var_distributed(kv_b_proj_weight)
                _set_var_distributed(q_nope_k_b_proj_weight)
                _set_var_distributed(q_rope_proj_weight)
                _set_var_distributed(ffn1_weight)
                # row parallel
                _set_var_distributed(linear_weight)
                _set_var_distributed(v_b_o_proj_weight)
                _set_var_distributed(ffn2_weight)

                _set_var_distributed(shared_expert_ffn1_weight)
                _set_var_distributed(shared_expert_ffn2_weight)

            self.q_proj_weights.append(q_proj_weight)
            self.q_a_proj_weights.append(q_a_proj_weight)
            self.q_a_layernorm_weights.append(q_a_layernorm_weight)
            self.q_b_proj_weights.append(q_b_proj_weight)
            self.kv_a_proj_with_mqa_weights.append(kv_a_proj_with_mqa_weight)
            self.kv_a_layernorm_weights.append(kv_a_layernorm_weight)
            self.kv_b_proj_weights.append(kv_b_proj_weight)
            self.qkv_weights.append(qkv_weight)

            self.q_nope_k_b_proj_weights.append(q_nope_k_b_proj_weight)
            self.q_rope_proj_weights.append(q_rope_proj_weight)
            self.v_b_o_proj_weights.append(v_b_o_proj_weight)

            self.linear_weights.append(linear_weight)

            self.gate_weights.append(gate_weight)
            self.ffn1_weights.append(ffn1_weight)
            self.ffn2_weights.append(ffn2_weight)

            self.shared_expert_ffn1_weights.append(shared_expert_ffn1_weight)
            self.shared_expert_ffn2_weights.append(shared_expert_ffn2_weight)
            self.shared_expert_gate_weights.append(shared_expert_gate_weight)

            self._add_parameter(q_proj_weight)
            self._add_parameter(q_a_proj_weight)
            self._add_parameter(q_a_layernorm_weight)
            self._add_parameter(q_b_proj_weight)
            self._add_parameter(kv_a_proj_with_mqa_weight)
            self._add_parameter(kv_a_layernorm_weight)
            self._add_parameter(kv_b_proj_weight)

            self._add_parameter(q_nope_k_b_proj_weight)
            self._add_parameter(q_rope_proj_weight)
            self._add_parameter(v_b_o_proj_weight)
            self._add_parameter(qkv_weight)

            self._add_parameter(shared_expert_ffn1_weight)
            self._add_parameter(shared_expert_ffn2_weight)
            self._add_parameter(shared_expert_gate_weight)

            self._add_parameter(linear_weight)

            self._add_parameter(gate_weight)
            self._add_parameter(ffn1_weight)
            self._add_parameter(ffn2_weight)

    def get_attr(self, attrs, idx):
        if isinstance(attrs, (list, tuple)):
            assert (
                len(attrs) == self.num_layers
            ), f"length of attrs is {len(attrs)} is not equal to self.num_layers {self.num_layers}"
            return attrs[idx]
        return attrs

    def _add_parameter(self, param):
        if param is None:
            return
        assert param.name not in self._parameters
        self._parameters[param.name] = param

    def init_weight_shape(self, config):

        if self.config.mla_config.use_mla():
            if self.config.mla_config.q_lora_rank is None:
                self.q_proj_weight_shape = [
                    self.config.embed_dim,
                    self.num_heads * (self.config.mla_config.qk_head_dim),
                ]
            else:
                self.q_a_proj_weight_shape = [self.config.embed_dim, self.config.mla_config.q_lora_rank]
                self.q_b_proj_weight_shape = [
                    self.config.mla_config.q_lora_rank,
                    self.num_heads * (self.config.mla_config.qk_head_dim),
                ]

            self.kv_a_proj_with_mqa_weight_shape = [
                self.config.embed_dim,
                self.config.mla_config.kv_lora_rank + self.config.mla_config.qk_rope_head_dim,
            ]
            self.kv_b_proj_weight_shape = [
                self.config.mla_config.kv_lora_rank,
                self.num_heads * (self.config.mla_config.qk_nope_head_dim + self.config.mla_config.v_head_dim),
            ]

            self.q_nope_k_b_proj_weight_shape = [
                self.embed_dim if self.config.mla_config.q_lora_rank is None else self.config.mla_config.q_lora_rank,
                self.num_heads * self.config.mla_config.kv_lora_rank,
            ]
            self.q_rope_proj_weight_shape = [
                self.embed_dim if self.config.mla_config.q_lora_rank is None else self.config.mla_config.q_lora_rank,
                self.num_heads * self.config.mla_config.qk_rope_head_dim,
            ]
            self.v_b_o_proj_weight_shape = [
                self.num_heads * self.config.mla_config.kv_lora_rank,
                self.embed_dim,
            ]
        else:
            self.qkv_weight_shape = (
                [(self.num_heads + 2 * self.kv_num_heads) * self.head_dim, self.embed_dim]
                if config.trans_qkvw
                else [self.embed_dim, (self.num_heads + 2 * self.kv_num_heads) * self.head_dim]
            )

        self.linear_weight_shape = [self.num_heads * self.head_dim, self.embed_dim]

        self.ffn1_weight_shape = (
            [self.embed_dim, self.intermediate_size * 2]
            if self.activation.endswith("glu")
            else [self.embed_dim, self.intermediate_size]
        )
        self.ffn2_weight_shape = [self.intermediate_size, self.embed_dim]

        if self.config.moe_config.has_moe():
            self.moe_ffn1_weight_shape = (
                [self.config.moe_config.num_experts, self.embed_dim, self.config.moe_config.moe_intermediate_size * 2]
                if self.activation.endswith("glu")
                else [self.config.moe_config.num_experts, self.embed_dim, self.config.moe_config.moe_intermediate_size]
            )
            self.moe_ffn2_weight_shape = [
                self.config.moe_config.num_experts,
                self.config.moe_config.moe_intermediate_size,
                self.embed_dim,
            ]

        if self.config.moe_config.has_shared_expert():
            self.shared_expert_ffn1_weight_shape = [
                self.embed_dim,
                self.config.moe_config.shared_expert_intermediate_size * 2,
            ]
            self.shared_expert_ffn2_weight_shape = [
                self.config.moe_config.shared_expert_intermediate_size,
                self.embed_dim,
            ]
            if self.config.moe_config.shared_expert_with_gate:
                self.shared_expert_gate_weight_shape = [
                    self.embed_dim,
                    1,
                ]

    def skip_quant(self, layer_name, layer_idx):
        return False

    def get_weight_create_dype(self):
        return self._dtype

    def compute_layernorm_before_qkv(self, src, i):
        if i == 0 and not self.config.speculate_config.speculate_method == "eagle":
            ln_out = self.norm_func(src, self.ln_scales[i], self.ln_biases[i], self._epsilon, begin_norm_axis=1)[0]
        else:
            ln_out = src

        return ln_out

    def compute_qkv_linear(self, ln_out, i, latent_cache=None, **kwargs):
        if self.config.mla_config.use_mla():
            if self.config.mla_config.q_lora_rank is not None:
                query = paddle.matmul(ln_out, self.q_a_proj_weights[i])
                query = self.norm_func(
                    x=query,
                    norm_weight=self.q_a_layernorm_weights[i],
                    norm_bias=None,
                    epsilon=self._epsilon,
                    begin_norm_axis=1,
                )[0]
                query = paddle.matmul(query, self.q_b_proj_weights[i])
            else:
                query = paddle.matmul(ln_out, self.q_proj_weights[i])

            query = query.reshape([-1, self.num_heads, self.config.mla_config.qk_head_dim])
            query_nope, query_pe = query.split(
                [self.config.mla_config.qk_nope_head_dim, self.config.mla_config.qk_rope_head_dim], axis=-1
            )

            compressed_kv = paddle.matmul(ln_out, self.kv_a_proj_with_mqa_weights[i])
            compressed_kv, key_pe = compressed_kv.split(
                [self.config.mla_config.kv_lora_rank, self.config.mla_config.qk_rope_head_dim], axis=-1
            )
            key_pe = key_pe.reshape([-1, 1, self.config.mla_config.qk_rope_head_dim])
            compressed_kv = self.norm_func(
                x=compressed_kv,
                norm_weight=self.kv_a_layernorm_weights[i],
                norm_bias=None,
                epsilon=self._epsilon,
                begin_norm_axis=1,
            )[0]
            query_pe, key_pe = self.config.rotary_emb(self.position_ids, query_pe, key_pe)

            if self.config.mla_config.use_absorb():
                from paddlenlp_ops import prefill_mla_write_cache

                prefill_mla_write_cache(
                    compressed_kv,
                    key_pe,
                    latent_cache,
                    kwargs.get("seq_lens_encoder", None),
                    kwargs.get("seq_lens_decoder", None),
                    kwargs.get("padding_offsets", None),
                    kwargs.get("cum_offsets", None),
                    kwargs.get("block_tables", None),
                    "none",
                    kwargs.get("max_input_length", -1),
                )

            key_value = paddle.matmul(compressed_kv, self.kv_b_proj_weights[i])
            key_value = key_value.reshape(
                [-1, self.num_heads, self.config.mla_config.qk_nope_head_dim + self.config.mla_config.v_head_dim]
            )
            key_nope, value = key_value.split(
                [self.config.mla_config.qk_nope_head_dim, self.config.mla_config.v_head_dim], axis=-1
            )

            query[..., self.config.mla_config.qk_nope_head_dim :] = query_pe
            key = paddle.empty_like(query)
            key[..., : self.config.mla_config.qk_nope_head_dim] = key_nope
            key[..., self.config.mla_config.qk_nope_head_dim :] = key_pe

            if self.config.mla_config.use_absorb():
                value = paddle.nn.functional.pad(
                    value, [0, self.config.mla_config.qk_head_dim - self.config.mla_config.v_head_dim], value=0
                )
                return query, key, value
            else:
                qkv_out = paddle.concat(
                    [
                        query.reshape([-1, self.num_heads * self.config.mla_config.qk_head_dim]),
                        key.reshape([-1, self.num_heads * self.config.mla_config.qk_head_dim]),
                        value.reshape([-1, self.num_heads * self.config.mla_config.v_head_dim]),
                    ],
                    axis=-1,
                )
                return qkv_out
        else:
            qkv_out = paddle.matmul(ln_out, self.qkv_weights[i], False, True)
            if self.qkv_biases[i] is not None:
                qkv_out = paddle.add(qkv_out, self.qkv_biases[i])
            return qkv_out

    def compute_qkv(self, src, residual_input, i):
        ln_out = self.compute_layernorm_before_qkv(src, i)

        if self.config.mla_config.use_absorb():
            qkv_out = ln_out
        else:
            qkv_out = self.compute_qkv_linear(ln_out, i)

        return qkv_out, residual_input

    def compute_max_len(self, seq_lens_encoder, seq_lens_decoder, cum_offsets):
        if seq_lens_encoder is None or seq_lens_decoder is None or cum_offsets is None:
            return None, None
        return paddle.incubate.nn.functional.blha_get_max_len(
            seq_lens_encoder, seq_lens_decoder, cum_offsets  # cum_offsets.shape[0] used as bsz
        )

    def compute_fmha(
        self,
        qkv_out,
        padding_offset,
        seq_lens,
        input_ids,
        rotary_embs,
        rotary_emb_dims,
        caches,
        pre_caches,
        pre_caches_length,
        attn_mask,
        i,
    ):
        bsz = input_ids.shape[0]
        """
        qkv: bsz, seq_len, 3, numhead, headsize ->
        q_out: bsz, numhead, seq_len, headsize
        kv_out: 2, bsz, numhead, seq_len, headsize
        """
        q_out, k_out, v_out = qkv_transpose_split(
            qkv_out, padding_offset, seq_lens, input_ids, self.num_heads, self.head_dim
        )

        # rotary emb (inplace)
        if rotary_embs is not None:
            encode_rotary_qk(
                q_out,
                k_out,
                rotary_embs,
                seq_lens,
                rotary_emb_dims=rotary_emb_dims,
                use_neox=self.use_neox_rotary_style,
            )

        if pre_caches is not None:
            k_out = paddle.concat([pre_caches[i][0, :bsz], k_out], axis=2)
            v_out = paddle.concat([pre_caches[i][1, :bsz], v_out], axis=2)

        # write cache kv (inplace)
        write_cache_kv(k_out, v_out, caches[i], seq_lens + pre_caches_length)

        # cutlass fmha
        qktv_out = variable_length_memory_efficient_attention(
            q_out,
            k_out,
            v_out,
            seq_lens,
            seq_lens + pre_caches_length,
            mask=attn_mask,
            scale=self.softmax_scale,
        )

        return transpose_remove_padding(qktv_out, seq_lens, padding_offset)

    def compute_mmha(self, qkv_out, caches, attn_mask, seq_lens, rotary_embs, rotary_emb_dims, i):
        return masked_multihead_attention(
            x=qkv_out,
            cache_kv=caches[i],
            src_mask=attn_mask,
            sequence_lengths=seq_lens,
            rotary_tensor=rotary_embs,
            rotary_emb_dims=rotary_emb_dims,
            use_neox_rotary_style=self.use_neox_rotary_style,
        )[0]

    def compute_out_linear(self, fmha_out, i):
        return paddle.matmul(fmha_out, self.linear_weights[i])

    def compute_attn(
        self,
        time_step,
        qkv_out,
        padding_offset,
        seq_lens,
        input_ids,
        rotary_embs,
        rotary_emb_dims,
        caches,
        pre_caches,
        pre_caches_length,
        attn_mask,
        i,
        **kwargs,
    ):
        # fmha compute
        if time_step is None:  # context
            fmha_out = self.compute_fmha(
                qkv_out,
                padding_offset,
                seq_lens,
                input_ids,
                rotary_embs,
                rotary_emb_dims,
                caches,
                pre_caches,
                pre_caches_length,
                attn_mask,
                i,
            )

        else:
            fmha_out = self.compute_mmha(qkv_out, caches, attn_mask, seq_lens, rotary_embs, rotary_emb_dims, i)

        return fmha_out

    def compute_ffn_layernorm(self, out_linear_out, residual_input, i):
        norm_out = self.norm_func(
            out_linear_out,
            norm_weight=self.ffn_ln_scales[i],
            norm_bias=self.ffn_ln_biases[i],
            epsilon=self._epsilon,
            begin_norm_axis=1,
            bias=self.linear_biases[i],
            residual=residual_input,
        )
        tmp_out, residual_input = norm_out[0], norm_out[1]

        return tmp_out, residual_input

    def compute_fused_moe(self, tmp_out, i):
        e_score_correction_bias = self.e_score_correction_biases[i]

        def get_moe_scores(
            gating_output: paddle.Tensor,
            config: MoeConfig,
        ) -> paddle.Tensor:
            # Compute softmax or sigmoid scores based on the topk_method
            if config.topk_method == "greedy":
                scores = paddle.nn.functional.softmax(gating_output, axis=-1)
                return scores
            elif config.topk_method == "group_limited_greedy":
                scores = paddle.nn.functional.softmax(gating_output, axis=-1)
                scores_with_bias = scores
            elif config.topk_method == "noaux_tc":
                if e_score_correction_bias is None:
                    raise ValueError("e_score_correction_bias must be provided for 'noaux_tc' method.")
                scores = paddle.nn.functional.sigmoid(gating_output)
                scores_with_bias = scores + e_score_correction_bias.unsqueeze(0)
            else:
                raise ValueError(
                    f"Unsupported topk_method: {config.topk_method}. Please choose 'group_limited_greedy' or 'noaux_tc'."
                )
            from paddlenlp_ops import noaux_tc

            scores = noaux_tc(
                scores,
                scores_with_bias,
                config.num_expert_group,
                config.topk_group,
                config.top_k,
                config.routed_scaling_factor,
            )
            return scores

        if self.config.moe_config.topk_method is not None:
            from paddle.incubate.nn.functional import moe_dispatch, moe_ffn, moe_reduce

            gate_out = paddle.matmul(tmp_out.cast("float32"), self.gate_weights[i])
            # 应用各种策略后重塑的 scores
            scores = get_moe_scores(gate_out, self.config.moe_config)

            # topk 在 moe_dispatch 中
            (
                permute_input,
                token_nums_per_expert,
                permute_indices_per_token,
                top_k_weights,
                top_k_indices,
            ) = moe_dispatch(tmp_out, scores, self.config.moe_config.top_k, False, topk_only_mode=True)

            ffn_out = moe_ffn(
                permute_input,
                token_nums_per_expert,
                self.ffn1_weights[i],
                self.ffn2_weights[i],
                self.ffn1_biases[i],
                self.ffn1_weights_scale[i] if hasattr(self, "ffn1_weights_scale") else None,
                self.ffn2_weights_scale[i] if hasattr(self, "ffn2_weights_scale") else None,
                self.quant_type if hasattr(self, "quant_type") else "None",
            )

            fused_moe_out = moe_reduce(
                ffn_out,
                top_k_weights,
                permute_indices_per_token,
                top_k_indices,
                self.ffn2_biases[i],
                norm_topk_prob=False,  # 在noaux_tc中做了
                routed_scaling_factor=1.0,  # 在noaux_tc中做了
            )
        else:
            fused_moe_out = fused_moe(
                tmp_out,
                self.gate_weights[i],
                self.ffn1_weights[i],
                self.ffn2_weights[i],
                self.ffn1_biases[i],
                self.ffn1_weights_scale[i] if hasattr(self, "ffn1_weights_scale") else None,
                self.ffn2_biases[i],
                self.ffn2_weights_scale[i] if hasattr(self, "ffn2_weights_scale") else None,
                self.quant_type if hasattr(self, "quant_type") else "None",
                self.config.moe_config.top_k,
                self.config.moe_config.norm_topk_prob,
            )
        return fused_moe_out

    def compute_activation(self, ffn1_out, i):
        return fused_bias_act(ffn1_out, self.ffn1_biases[i], act_method=self.activation)

    def compute_ffn1(self, tmp_out, i):
        return paddle.matmul(tmp_out, self.ffn1_weights[i])

    def compute_ffn2(self, ffn1_out, i):
        return paddle.matmul(ffn1_out, self.ffn2_weights[i])

    def compute_bias_residual_layernorm(self, ffn2_out, residual_input, i, num_layers):
        if i != num_layers - 1:
            norm_out = self.norm_func(
                ffn2_out,
                norm_weight=self.ln_scales[i + 1],
                norm_bias=self.ln_biases[i + 1],
                epsilon=self._epsilon,
                begin_norm_axis=1,
                bias=self.ffn2_biases[i],
                residual=residual_input,
            )
            tmp_out, residual_input = norm_out[0], norm_out[1]
        else:
            tmp_out = fused_layer_norm(
                ffn2_out,
                norm_weight=None,
                norm_bias=None,
                epsilon=self._epsilon,
                begin_norm_axis=1,
                bias=self.ffn2_biases[i],
                residual=residual_input,
            )[0]
        return tmp_out, residual_input

    def compute_shared_expert(self, tmp_out, i):
        ffn1_out = paddle.matmul(tmp_out, self.shared_expert_ffn1_weights[i])
        ffn1_out = fused_bias_act(ffn1_out, None, act_method=self.activation)
        ffn2_out = paddle.matmul(ffn1_out, self.shared_expert_ffn2_weights[i])
        if self.config.moe_config.shared_expert_with_gate:
            gate_out = paddle.matmul(tmp_out, self.shared_expert_gate_weights[i])
            gate_out = paddle.nn.functional.sigmoid(gate_out)
            return gate_out * ffn2_out
        return ffn2_out

    def pre_process(self, **kwargs):
        if self.config.mla_config.use_mla():
            seq_lens_encoder = kwargs.get("seq_lens_encoder", None)
            seq_lens_decoder = kwargs.get("seq_lens_decoder", None)
            seq_lens_this_time = kwargs.get("seq_lens_this_time", None)
            position_ids_shape = paddle.sum(seq_lens_this_time)
            self.position_ids = paddle.empty(shape=position_ids_shape, dtype=seq_lens_encoder.dtype)
            self.mask_encoder_batch = paddle.empty(shape=position_ids_shape, dtype=seq_lens_encoder.dtype).unsqueeze(1)

            from paddlenlp_ops import get_position_ids_and_mask_encoder_batch

            # In-place operations that compute the position_ids.
            os.environ["stride_in_no_check_dy2st_diff"] = "1"
            get_position_ids_and_mask_encoder_batch(
                seq_lens_encoder, seq_lens_decoder, seq_lens_this_time, self.position_ids, self.mask_encoder_batch
            )

    def post_process(self, **kwargs):
        time_step = kwargs.get("time_step", None)
        multi_block_output = kwargs.get("multi_block_output", None)
        cum_offsets = kwargs.get("cum_offsets", None)
        seq_lens = kwargs.get("seq_lens", None)
        input_ids = kwargs.get("input_ids", None)

        if time_step is None:
            out = rebuild_padding(multi_block_output, cum_offsets, seq_lens, input_ids)
        else:
            out = multi_block_output

        return out

    def forward(
        self,
        input_ids,
        src,
        cum_offsets=None,
        padding_offset=None,
        attn_mask=None,
        caches=None,
        pre_caches=None,
        pre_caches_length=0,
        rotary_embs=None,
        rotary_emb_dims=0,
        seq_lens=None,
        time_step=None,
        **kwargs,
    ):
        r"""
        Applies multi transformer layers on the input.

        Parameters:
            src (Tensor): The input of Transformer layers. It is
                a tensor with shape `[batch_size, sequence_length, d_model]`.
                The data type should be float16 or float32.
            attn_mask (Tensor, optional): A tensor used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions. It is a tensor with shape
                `[batch_size, 1, sequence_length, sequence_length]`. It can be
                None when nothing wanted or needed to be prevented attention to.
                Default None.
            caches (list(Tensor)|tuple(Tensor), optional): The cache structure
                tensors for the inference generation model. It is only used for
                inference and should be None for training. The shape is
                `[2, batch_size, num_head, max_seq_len, head_dim]`. Default None.
            pre_caches (list(Tensor)|tuple(Tensor), optional): The prefix caches
                for the generation model. The shape is `[2, bsz, num\_head, cache\_len, head\_dim]`. Default None.
            rotary_embs (Tensor optional): The RoPE embs for the rotary computation. The shape is `[2, bsz, 1, seq\_len, head\_dim]`. Default None.
            rotary_emb_dims (int, optional): The rotary_emb_dims of rotary computation, and it is 0 when rotary_embs is None,
                1 when rotary_embs is not None and pos_extra_ids is None, 2 when rotary_embs and pos_extra_ids are both not None. Default 0.
            seq_lens (Tensor optional): The sequence lengths of this batch. The shape is `[bsz]`. Default None.
            time_step (Tensor, optional): The time step tensor for the generation
                model. Which used in decode stage, to represent the time step,
                that is, the real seq_len of CacheKV. The shape is `[1]`, must be
                in CPUPlace. Default None.

        Returns:
            Tensor|tuple: If `caches` is None, return a tensor that has
            the same shape and data type with `src`, representing the output
            of Transformer layers. If `caches` is not None, return the
            tuple (output, caches), which output is the output of
            Transformer layers, caches is inplace with input `caches`.
        """
        self.pre_process(**kwargs)
        kwargs["cum_offsets"] = cum_offsets

        if caches is not None:
            assert len(caches) == len(self.linear_weights) or len(caches) == 2 * len(self.linear_weights)

        assert self.num_layers == len(self.linear_weights)

        max_enc_len_this_time, max_dec_len_this_time = self.compute_max_len(
            kwargs.get("seq_lens_encoder", None), kwargs.get("seq_lens_decoder", None), cum_offsets
        )
        kwargs["max_enc_len_this_time"] = max_enc_len_this_time
        kwargs["max_dec_len_this_time"] = max_dec_len_this_time

        if self.config.append_attn:

            from paddlenlp_ops import get_block_shape_and_split_kv_block

            (
                kwargs["encoder_batch_ids"],
                kwargs["encoder_tile_ids_per_batch"],
                kwargs["encoder_num_blocks"],
                kwargs["kv_batch_ids"],
                kwargs["kv_tile_ids_per_batch"],
                kwargs["kv_num_blocks"],
                kwargs["decoder_batch_ids"],
                kwargs["decoder_tile_ids_per_batch"],
                kwargs["decoder_num_blocks"],
                kwargs["decoder_num_blocks_cpu"],
                kwargs["max_len_kv"],
            ) = get_block_shape_and_split_kv_block(
                kwargs.get("seq_lens_encoder", None),
                kwargs.get("seq_lens_decoder", None),
                max_enc_len_this_time,
                max_dec_len_this_time,
                kwargs.get("seq_lens_this_time", None),
                kwargs.get("cum_offsets", None),
                self.num_heads // self.kv_num_heads,
                kwargs.get("block_size", 64),
                self.config.speculate_config.speculate_max_draft_token_num,
            )

        residual_input = src
        for i in range(self.num_layers):
            qkv_out, residual_input = self.compute_qkv(src, residual_input, i)
            fmha_out = self.compute_attn(
                time_step,
                qkv_out,
                padding_offset,
                seq_lens,
                input_ids,
                rotary_embs,
                rotary_emb_dims,
                caches,
                pre_caches,
                pre_caches_length,
                attn_mask,
                i,
                **kwargs,
            )
            if self.config.mla_config.use_absorb():
                out_linear_out = fmha_out
            else:
                out_linear_out = self.compute_out_linear(fmha_out, i)

            # print(f"{i}: out_linear_out: {out_linear_out}")

            # all_reduce
            if self.nranks > 1:
                dist.all_reduce(out_linear_out)

            # ffn layernorm
            tmp_out, residual_input = self.compute_ffn_layernorm(out_linear_out, residual_input, i)

            if self.config.moe_config.use_moe(i):
                # fused moe
                ffn2_out = self.compute_fused_moe(tmp_out, i)

                # shared_expert
                if self.config.moe_config.use_shared_expert(i):
                    shared_expert_out = self.compute_shared_expert(tmp_out, i)
                    ffn2_out = ffn2_out + shared_expert_out
            else:
                # ffn1 matmul
                ffn1_out = self.compute_ffn1(tmp_out, i)
                ffn1_out = self.compute_activation(ffn1_out, i)

                # ffn2 matmul
                ffn2_out = self.compute_ffn2(ffn1_out, i)

            # all_reduce
            if self.nranks > 1:
                dist.all_reduce(ffn2_out)

            # norm + residual_add_bias
            tmp_out, residual_input = self.compute_bias_residual_layernorm(
                ffn2_out, residual_input, i, self.num_layers
            )
            src = tmp_out

        kwargs["time_step"] = time_step
        kwargs["multi_block_output"] = tmp_out
        kwargs["seq_lens"] = seq_lens
        kwargs["input_ids"] = input_ids

        out = self.post_process(**kwargs)
        return out, caches


class FusedMultiTransformerPostLayernorm(FusedMultiTransformerBase):
    def __init__(self, config: FusedMultiTransformerConfig):
        super().__init__(config)

    def compute_qkv(self, src, residual_input, i):
        qkv_out = self.compute_qkv_linear(src, i)
        return qkv_out, src

    def compute_ffn_layernorm(self, out_linear_out, residual_input, i):
        tmp_out = self.norm_func(
            out_linear_out,
            norm_weight=self.ln_scales[i],
            norm_bias=self.ln_biases[i],
            epsilon=self._epsilon,
            residual_alpha=self._residual_alpha,
            begin_norm_axis=1,
            bias=self.linear_biases[i],
            residual=residual_input,
        )[0]

        return tmp_out, tmp_out

    def compute_bias_residual_layernorm(self, ffn2_out, residual_input, i, num_layers):
        tmp_out = self.norm_func(
            ffn2_out,
            norm_weight=self.ffn_ln_scales[i],
            norm_bias=self.ffn_ln_biases[i],
            epsilon=self._epsilon,
            residual_alpha=self._residual_alpha,
            begin_norm_axis=1,
            bias=self.ffn2_biases[i],
            residual=residual_input,
        )[0]
        return tmp_out, tmp_out


class FusedMultiTransformerWeightOnly(FusedMultiTransformerBase):
    def __init__(self, config: FusedMultiTransformerConfig):
        super().__init__(config)
        self.quant_type = config.quant_type
        self.weightonly_group_size = config.weightonly_group_size
        if self.quant_type == "weight_only_int8":
            self.weight_dtype = "int8"
        elif self.quant_type == "weight_only_int4":
            self.weight_dtype = "int4"
        else:
            assert (
                self.quant_type == "weight_only_int8" or self.quant_type == "weight_only_int4"
            ), "Expected quant_type equal to 'weight_only_int8' or 'weight_only_int4', but received {}".format(
                self.quant_type
            )

        self.weight_scale_dtype = self._dtype
        self.qkv_weights_scale = []
        self.linear_weights_scale = []
        self.ffn1_weights_scale = []
        self.ffn2_weights_scale = []

        self.q_proj_weights_scale = []
        self.q_a_proj_weights_scale = []
        self.q_b_proj_weights_scale = []
        self.kv_a_proj_with_mqa_weights_scale = []
        self.kv_b_proj_weights_scale = []
        self.q_nope_k_b_proj_weights_scale = []
        self.q_rope_proj_weights_scale = []
        self.v_b_o_proj_weights_scale = []

        self.shared_expert_ffn1_weights_scale = []
        self.shared_expert_ffn2_weights_scale = []

        for i in range(self.num_layers):

            q_proj_weight_scale = None
            q_a_proj_weight_scale = None
            q_b_proj_weight_scale = None
            kv_a_proj_with_mqa_weight_scale = None
            kv_b_proj_weight_scale = None
            if self.config.mla_config.use_mla():
                q_proj_weight_scale_attr = self.get_attr(self.config.mla_config.q_proj_weight_scale_attrs, i)
                if q_proj_weight_scale_attr:
                    q_proj_weight_scale = self.create_parameter(
                        shape=[self.num_heads * (self.config.mla_config.qk_head_dim)]
                        if self.weightonly_group_size < 0
                        else [
                            (self.q_proj_weight_shape[1] + self.weightonly_group_size - 1)
                            // self.weightonly_group_size,
                            self.num_heads * (self.config.mla_config.qk_head_dim),
                        ],
                        attr=q_proj_weight_scale_attr,
                        dtype=self.weight_scale_dtype,
                        is_bias=False,
                    )

                q_a_proj_weight_scale_attr = self.get_attr(self.config.mla_config.q_a_proj_weight_scale_attrs, i)
                q_b_proj_weight_scale_attr = self.get_attr(self.config.mla_config.q_b_proj_weight_scale_attrs, i)
                if q_a_proj_weight_scale_attr:
                    q_a_proj_weight_scale = self.create_parameter(
                        shape=[self.config.mla_config.q_lora_rank]
                        if self.weightonly_group_size < 0
                        else [
                            (self.q_a_proj_weight_shape[1] + self.weightonly_group_size - 1)
                            // self.weightonly_group_size,
                            self.config.mla_config.q_lora_rank,
                        ],
                        attr=q_a_proj_weight_scale_attr,
                        dtype=self.weight_scale_dtype,
                        is_bias=False,
                    )
                if q_b_proj_weight_scale_attr:
                    q_b_proj_weight_scale = self.create_parameter(
                        shape=[self.num_heads * (self.config.mla_config.qk_head_dim)]
                        if self.weightonly_group_size < 0
                        else [
                            (self.q_b_proj_weight_shape[1] + self.weightonly_group_size - 1)
                            // self.weightonly_group_size,
                            self.num_heads * (self.config.mla_config.qk_head_dim),
                        ],
                        attr=q_b_proj_weight_scale_attr,
                        dtype=self.weight_scale_dtype,
                        is_bias=False,
                    )

                kv_a_proj_with_mqa_weight_scale_attr = self.get_attr(
                    self.config.mla_config.kv_a_proj_with_mqa_weight_scale_attrs, i
                )
                kv_b_proj_weight_scale_attr = self.get_attr(self.config.mla_config.kv_b_proj_weight_scale_attrs, i)
                if kv_a_proj_with_mqa_weight_scale_attr:
                    kv_a_proj_with_mqa_weight_scale = self.create_parameter(
                        shape=[self.config.mla_config.kv_lora_rank + self.config.mla_config.qk_rope_head_dim]
                        if self.weightonly_group_size < 0
                        else [
                            (self.kv_a_proj_with_mqa_weight_shape[1] + self.weightonly_group_size - 1)
                            // self.weightonly_group_size,
                            self.config.mla_config.kv_lora_rank + self.config.mla_config.qk_rope_head_dim,
                        ],
                        attr=kv_a_proj_with_mqa_weight_scale_attr,
                        dtype=self.weight_scale_dtype,
                        is_bias=False,
                    )
                if kv_b_proj_weight_scale_attr:
                    kv_b_proj_weight_scale = self.create_parameter(
                        shape=[
                            self.num_heads
                            * (self.config.mla_config.qk_nope_head_dim + self.config.mla_config.v_head_dim)
                        ]
                        if self.weightonly_group_size < 0
                        else [
                            (self.kv_b_proj_weight_shape[1] + self.weightonly_group_size - 1)
                            // self.weightonly_group_size,
                            self.num_heads
                            * (self.config.mla_config.qk_nope_head_dim + self.config.mla_config.v_head_dim),
                        ],
                        attr=kv_b_proj_weight_scale_attr,
                        dtype=self.weight_scale_dtype,
                        is_bias=False,
                    )

            qkv_weight_scale = None
            qkv_weight_scale_attr = self.get_attr(config.qkv_weight_scale_attrs, i)
            if qkv_weight_scale_attr:
                qkv_weight_scale = self.create_parameter(
                    shape=[(self.num_heads + 2 * self.kv_num_heads) * self.head_dim]
                    if self.weightonly_group_size < 0
                    else [
                        (self.qkv_weight_shape[1] + self.weightonly_group_size - 1) // self.weightonly_group_size,
                        (self.num_heads + 2 * self.kv_num_heads) * self.head_dim,
                    ],
                    attr=qkv_weight_scale_attr,
                    dtype=self.weight_scale_dtype,
                    is_bias=False,
                )

            linear_weight_scale = None
            linear_weight_scale_attr = self.get_attr(config.linear_weight_scale_attrs, i)
            if linear_weight_scale_attr:
                linear_weight_scale = self.create_parameter(
                    shape=[self.embed_dim]
                    if self.weightonly_group_size < 0
                    else [
                        (self.linear_weight_shape[1] + self.weightonly_group_size - 1) // self.weightonly_group_size,
                        self.embed_dim,
                    ],
                    attr=linear_weight_scale_attr,
                    dtype=self.weight_scale_dtype,
                    is_bias=False,
                )

            q_nope_k_b_proj_weight_scale = None
            q_rope_proj_weight_scale = None
            v_b_o_proj_weight_scale = None
            if self.config.mla_config.use_absorb():
                q_nope_k_b_proj_weight_scale_attr = self.get_attr(
                    self.config.mla_config.q_nope_k_b_proj_weight_scale_attrs, i
                )
                q_rope_proj_weight_scale_attr = self.get_attr(self.config.mla_config.q_rope_proj_weight_scale_attrs, i)
                v_b_o_proj_weight_scale_attr = self.get_attr(self.config.mla_config.v_b_o_proj_weight_scale_attrs, i)
                if q_nope_k_b_proj_weight_scale_attr:
                    q_nope_k_b_proj_weight_scale = self.create_parameter(
                        shape=[self.num_heads * self.config.mla_config.kv_lora_rank]
                        if self.weightonly_group_size < 0
                        else [
                            (self.q_nope_k_b_proj_weight_shape[1] + self.weightonly_group_size - 1)
                            // self.weightonly_group_size,
                            self.num_heads * self.config.mla_config.kv_lora_rank,
                        ],
                        attr=q_nope_k_b_proj_weight_scale_attr,
                        dtype=self.weight_scale_dtype,
                        is_bias=False,
                    )
                if q_rope_proj_weight_scale_attr:
                    q_rope_proj_weight_scale = self.create_parameter(
                        shape=[self.num_heads * self.config.mla_config.qk_rope_head_dim]
                        if self.weightonly_group_size < 0
                        else [
                            (self.q_rope_proj_weight_shape[1] + self.weightonly_group_size - 1)
                            // self.weightonly_group_size,
                            self.num_heads * self.config.mla_config.qk_rope_head_dim,
                        ],
                        attr=q_rope_proj_weight_scale_attr,
                        dtype=self.weight_scale_dtype,
                        is_bias=False,
                    )
                if v_b_o_proj_weight_scale_attr:
                    v_b_o_proj_weight_scale = self.create_parameter(
                        shape=[self.embed_dim]
                        if self.weightonly_group_size < 0
                        else [
                            (self.v_b_o_proj_weight_shape[1] + self.weightonly_group_size - 1)
                            // self.weightonly_group_size,
                            self.embed_dim,
                        ],
                        attr=v_b_o_proj_weight_scale_attr,
                        dtype=self.weight_scale_dtype,
                        is_bias=False,
                    )

            ffn1_weight_scale = None
            ffn2_weight_scale = None
            ffn1_weight_scale_attr = self.get_attr(config.ffn1_weight_scale_attrs, i)
            ffn2_weight_scale_attr = self.get_attr(config.ffn2_weight_scale_attrs, i)
            if self.config.moe_config.use_moe(i):
                ffn1_weight_scale = self.create_parameter(
                    shape=[self.config.moe_config.num_experts, self.config.moe_config.moe_intermediate_size * 2]
                    if config.activation.endswith("glu")
                    else [self.config.moe_config.num_experts, self.config.moe_config.moe_intermediate_size],
                    attr=ffn1_weight_scale_attr,
                    dtype=self.weight_scale_dtype,
                    is_bias=False,
                )
            else:
                base_shape = (
                    [self.intermediate_size * 2] if config.activation.endswith("glu") else [self.intermediate_size]
                )
                ffn1_weight_scale = self.create_parameter(
                    shape=base_shape
                    if self.weightonly_group_size < 0
                    else [
                        (self.ffn1_weight_shape[1] + self.weightonly_group_size - 1) // self.weightonly_group_size,
                        base_shape[0],
                    ],
                    attr=ffn1_weight_scale_attr,
                    dtype=self.weight_scale_dtype,
                    is_bias=False,
                )

            if self.config.moe_config.use_moe(i):
                ffn2_weight_scale = self.create_parameter(
                    shape=[self.config.moe_config.num_experts, self.embed_dim],
                    attr=ffn2_weight_scale_attr,
                    dtype=self.weight_scale_dtype,
                    is_bias=False,
                )
            else:
                ffn2_weight_scale = self.create_parameter(
                    shape=[self.embed_dim]
                    if self.weightonly_group_size < 0
                    else [
                        (self.ffn2_weight_shape[1] + self.weightonly_group_size - 1) // self.weightonly_group_size,
                        self.embed_dim,
                    ],
                    attr=ffn2_weight_scale_attr,
                    dtype=self.weight_scale_dtype,
                    is_bias=False,
                )

            shared_expert_ffn1_weight_scale = None
            shared_expert_ffn2_weight_scale = None
            shared_expert_ffn1_weight_scale_attr = self.get_attr(
                config.moe_config.shared_expert_ffn1_weight_scale_attrs, i
            )
            shared_expert_ffn2_weight_scale_attr = self.get_attr(
                config.moe_config.shared_expert_ffn2_weight_scale_attrs, i
            )
            if self.config.moe_config.use_shared_expert(i):
                shared_expert_ffn1_weight_scale = self.create_parameter(
                    shape=[self.config.moe_config.shared_expert_intermediate_size * 2]
                    if self.weightonly_group_size < 0
                    else [
                        (self.shared_expert_ffn1_weight_shape[1] + self.weightonly_group_size - 1)
                        // self.weightonly_group_size,
                        self.config.moe_config.shared_expert_intermediate_size * 2,
                    ],
                    attr=shared_expert_ffn1_weight_scale_attr,
                    dtype=self.weight_scale_dtype,
                    is_bias=False,
                )
                shared_expert_ffn2_weight_scale = self.create_parameter(
                    shape=[self.embed_dim]
                    if self.weightonly_group_size < 0
                    else [
                        (self.shared_expert_ffn2_weight_shape[1] + self.weightonly_group_size - 1)
                        // self.weightonly_group_size,
                        self.embed_dim,
                    ],
                    attr=shared_expert_ffn2_weight_scale_attr,
                    dtype=self.weight_scale_dtype,
                    is_bias=False,
                )

            self.q_proj_weights_scale.append(q_proj_weight_scale)
            self.q_a_proj_weights_scale.append(q_a_proj_weight_scale)
            self.q_b_proj_weights_scale.append(q_b_proj_weight_scale)
            self.kv_a_proj_with_mqa_weights_scale.append(kv_a_proj_with_mqa_weight_scale)
            self.kv_b_proj_weights_scale.append(kv_b_proj_weight_scale)
            self.qkv_weights_scale.append(qkv_weight_scale)

            self.q_nope_k_b_proj_weights_scale.append(q_nope_k_b_proj_weight_scale)
            self.q_rope_proj_weights_scale.append(q_rope_proj_weight_scale)
            self.v_b_o_proj_weights_scale.append(v_b_o_proj_weight_scale)

            self.linear_weights_scale.append(linear_weight_scale)
            self.ffn1_weights_scale.append(ffn1_weight_scale)
            self.ffn2_weights_scale.append(ffn2_weight_scale)

            self.shared_expert_ffn1_weights_scale.append(shared_expert_ffn1_weight_scale)
            self.shared_expert_ffn2_weights_scale.append(shared_expert_ffn2_weight_scale)

            self._add_parameter(q_proj_weight_scale)
            self._add_parameter(q_a_proj_weight_scale)
            self._add_parameter(q_b_proj_weight_scale)
            self._add_parameter(kv_a_proj_with_mqa_weight_scale)
            self._add_parameter(kv_b_proj_weight_scale)
            self._add_parameter(qkv_weight_scale)

            self._add_parameter(q_nope_k_b_proj_weight_scale)
            self._add_parameter(q_rope_proj_weight_scale)
            self._add_parameter(v_b_o_proj_weight_scale)

            self._add_parameter(linear_weight_scale)
            self._add_parameter(ffn1_weight_scale)
            self._add_parameter(ffn2_weight_scale)

            self._add_parameter(shared_expert_ffn1_weight_scale)
            self._add_parameter(shared_expert_ffn2_weight_scale)

    def get_weight_create_dype(self):
        return "int8"  # If use weightonly int4, params dtype is int8, and one of the dimension will be half.

    def init_weight_shape(self, config):
        super().init_weight_shape(config)

        if self.config.mla_config.use_mla():
            if self.config.mla_config.q_lora_rank is None:
                self.q_proj_weight_shape = [
                    self.num_heads * (self.config.mla_config.qk_head_dim),
                    self.config.embed_dim,
                ]
            else:
                self.q_a_proj_weight_shape = [self.config.mla_config.q_lora_rank, self.config.embed_dim]
                self.q_b_proj_weight_shape = [
                    self.num_heads * (self.config.mla_config.qk_head_dim),
                    self.config.mla_config.q_lora_rank,
                ]

            self.kv_a_proj_with_mqa_weight_shape = [
                self.config.mla_config.kv_lora_rank + self.config.mla_config.qk_rope_head_dim,
                self.config.embed_dim,
            ]
            self.kv_b_proj_weight_shape = [
                self.num_heads * (self.config.mla_config.qk_nope_head_dim + self.config.mla_config.v_head_dim),
                self.config.mla_config.kv_lora_rank,
            ]

            self.q_nope_k_b_proj_weight_shape = [
                self.num_heads * self.config.mla_config.kv_lora_rank,
                self.embed_dim if self.config.mla_config.q_lora_rank is None else self.config.mla_config.q_lora_rank,
            ]
            self.q_rope_proj_weight_shape = [
                self.num_heads * self.config.mla_config.qk_rope_head_dim,
                self.embed_dim if self.config.mla_config.q_lora_rank is None else self.config.mla_config.q_lora_rank,
            ]
            self.v_b_o_proj_weight_shape = [
                self.embed_dim,
                self.num_heads * self.config.mla_config.kv_lora_rank,
            ]
        else:
            self.qkv_weight_shape = (
                [(self.num_heads + 2 * self.kv_num_heads) * self.head_dim, self.embed_dim]
                if config.trans_qkvw
                else [self.embed_dim, (self.num_heads + 2 * self.kv_num_heads) * self.head_dim]
            )

        self.linear_weight_shape = [self.embed_dim, self.num_heads * self.head_dim]
        self.ffn1_weight_shape = (
            [self.intermediate_size * 2, self.embed_dim]
            if self.activation.endswith("glu")
            else [self.intermediate_size, self.embed_dim]
        )
        self.ffn2_weight_shape = [self.embed_dim, self.intermediate_size]

        if config.quant_type == "weight_only_int4":
            if self.config.mla_config.use_mla():
                if self.config.mla_config.q_lora_rank is None:
                    self.q_proj_weight_shape[0] //= 2
                else:
                    self.q_a_proj_weight_shape[0] //= 2
                    self.q_b_proj_weight_shape[0] //= 2
                self.kv_a_proj_with_mqa_weight_shape[0] //= 2
                self.kv_b_proj_weight_shape[0] //= 2

                self.q_nope_k_b_proj_weight_shape[0] //= 2
                self.q_rope_proj_weight_shape[0] //= 2
                self.v_b_o_proj_weight_shape[0] //= 2
            else:
                self.qkv_weight_shape[0] //= 2
            self.linear_weight_shape[0] //= 2
            self.ffn1_weight_shape[0] //= 2
            self.ffn2_weight_shape[0] //= 2

        if self.config.moe_config.has_moe():
            self.moe_ffn1_weight_shape = (
                [self.config.moe_config.num_experts, self.embed_dim, self.config.moe_config.moe_intermediate_size * 2]
                if self.activation.endswith("glu")
                else [self.config.moe_config.num_experts, self.embed_dim, self.config.moe_config.moe_intermediate_size]
            )
            self.moe_ffn2_weight_shape = [
                self.config.moe_config.num_experts,
                self.config.moe_config.moe_intermediate_size,
                self.embed_dim,
            ]

            if config.quant_type == "weight_only_int4":
                if config.moe_config.has_shared_expert():
                    self.moe_ffn1_weight_shape[2] //= 2
                    self.moe_ffn2_weight_shape[1] //= 2
                else:
                    self.moe_ffn1_weight_shape[2] //= 2
                    self.moe_ffn2_weight_shape[2] //= 2

        if self.config.moe_config.has_shared_expert():
            self.shared_expert_ffn1_weight_shape = [
                self.config.moe_config.shared_expert_intermediate_size * 2,
                self.embed_dim,
            ]
            self.shared_expert_ffn2_weight_shape = [
                self.embed_dim,
                self.config.moe_config.shared_expert_intermediate_size,
            ]
            if self.config.moe_config.shared_expert_with_gate:
                self.shared_expert_gate_weight_shape = [
                    self.embed_dim,
                    1,
                ]
            if config.quant_type == "weight_only_int4":
                self.shared_expert_ffn1_weight_shape[0] //= 2
                self.shared_expert_ffn2_weight_shape[0] //= 2

    def compute_qkv_linear(self, ln_out, i, latent_cache=None, **kwargs):
        if self.config.mla_config.use_mla():
            if self.config.mla_config.q_lora_rank is not None:
                query = weight_only_linear(
                    ln_out,
                    weight=self.q_a_proj_weights[i],
                    weight_scale=self.q_a_proj_weights_scale[i],
                    weight_dtype=self.weight_dtype,
                    group_size=self.weightonly_group_size,
                )
                query = self.norm_func(
                    x=query,
                    norm_weight=self.q_a_layernorm_weights[i],
                    norm_bias=None,
                    epsilon=self._epsilon,
                    begin_norm_axis=1,
                )[0]
                query = weight_only_linear(
                    query,
                    weight=self.q_b_proj_weights[i],
                    weight_scale=self.q_b_proj_weights_scale[i],
                    weight_dtype=self.weight_dtype,
                    group_size=self.weightonly_group_size,
                )
            else:
                query = weight_only_linear(
                    ln_out,
                    weight=self.q_proj_weights[i],
                    weight_scale=self.q_proj_weights_scale[i],
                    weight_dtype=self.weight_dtype,
                    group_size=self.weightonly_group_size,
                )

            query = query.reshape([-1, self.num_heads, self.config.mla_config.qk_head_dim])
            query_nope, query_pe = query.split(
                [self.config.mla_config.qk_nope_head_dim, self.config.mla_config.qk_rope_head_dim], axis=-1
            )

            compressed_kv = weight_only_linear(
                ln_out,
                weight=self.kv_a_proj_with_mqa_weights[i],
                weight_scale=self.kv_a_proj_with_mqa_weights_scale[i],
                weight_dtype=self.weight_dtype,
                group_size=self.weightonly_group_size,
            )
            compressed_kv, key_pe = compressed_kv.split(
                [self.config.mla_config.kv_lora_rank, self.config.mla_config.qk_rope_head_dim], axis=-1
            )
            key_pe = key_pe.reshape([-1, 1, self.config.mla_config.qk_rope_head_dim])
            compressed_kv = self.norm_func(
                x=compressed_kv,
                norm_weight=self.kv_a_layernorm_weights[i],
                norm_bias=None,
                epsilon=self._epsilon,
                begin_norm_axis=1,
            )[0]
            query_pe, key_pe = self.config.rotary_emb(self.position_ids, query_pe, key_pe)

            if self.config.mla_config.use_absorb():
                from paddlenlp_ops import prefill_mla_write_cache

                prefill_mla_write_cache(
                    compressed_kv,
                    key_pe,
                    latent_cache,
                    kwargs.get("seq_lens_encoder", None),
                    kwargs.get("seq_lens_decoder", None),
                    kwargs.get("padding_offsets", None),
                    kwargs.get("cum_offsets", None),
                    kwargs.get("block_tables", None),
                    "none",
                    kwargs.get("max_input_length", -1),
                )

            key_value = weight_only_linear(
                compressed_kv,
                weight=self.kv_b_proj_weights[i],
                weight_scale=self.kv_b_proj_weights_scale[i],
                weight_dtype=self.weight_dtype,
                group_size=self.weightonly_group_size,
            )
            key_value = key_value.reshape(
                [-1, self.num_heads, self.config.mla_config.qk_nope_head_dim + self.config.mla_config.v_head_dim]
            )
            key_nope, value = key_value.split(
                [self.config.mla_config.qk_nope_head_dim, self.config.mla_config.v_head_dim], axis=-1
            )

            query[..., self.config.mla_config.qk_nope_head_dim :] = query_pe
            key = paddle.empty_like(query)
            key[..., : self.config.mla_config.qk_nope_head_dim] = key_nope
            key[..., self.config.mla_config.qk_nope_head_dim :] = key_pe

            if self.config.mla_config.use_absorb():
                value = paddle.nn.functional.pad(
                    value, [0, self.config.mla_config.qk_head_dim - self.config.mla_config.v_head_dim], value=0
                )
                return query, key, value
            else:
                qkv_out = paddle.concat(
                    [
                        query.reshape([-1, self.num_heads * self.config.mla_config.qk_head_dim]),
                        key.reshape([-1, self.num_heads * self.config.mla_config.qk_head_dim]),
                        value.reshape([-1, self.num_heads * self.config.mla_config.v_head_dim]),
                    ],
                    axis=-1,
                )
                return qkv_out
        else:
            qkv_out = weight_only_linear(
                ln_out,
                weight=self.qkv_weights[i],
                bias=self.qkv_biases[i],
                weight_scale=self.qkv_weights_scale[i],
                weight_dtype=self.weight_dtype,
                group_size=self.weightonly_group_size,
            )
            return qkv_out

    def compute_out_linear(self, fmha_out, i):
        return weight_only_linear(
            fmha_out,
            weight=self.linear_weights[i],
            weight_scale=self.linear_weights_scale[i],
            weight_dtype=self.weight_dtype,
            group_size=self.weightonly_group_size,
        )

    def compute_ffn1(self, tmp_out, i):
        return weight_only_linear(
            tmp_out,
            weight=self.ffn1_weights[i],
            weight_scale=self.ffn1_weights_scale[i],
            weight_dtype=self.weight_dtype,
            group_size=self.weightonly_group_size,
        )

    def compute_ffn2(self, ffn1_out, i):
        return weight_only_linear(
            ffn1_out,
            weight=self.ffn2_weights[i],
            weight_scale=self.ffn2_weights_scale[i],
            weight_dtype=self.weight_dtype,
            group_size=self.weightonly_group_size,
        )

    def compute_shared_expert(self, tmp_out, i):
        ffn1_out = weight_only_linear(
            tmp_out,
            weight=self.shared_expert_ffn1_weights[i],
            weight_scale=self.shared_expert_ffn1_weights_scale[i],
            weight_dtype=self.weight_dtype,
            group_size=self.weightonly_group_size,
        )
        ffn1_out = fused_bias_act(ffn1_out, None, act_method=self.activation)
        ffn2_out = weight_only_linear(
            ffn1_out,
            weight=self.shared_expert_ffn2_weights[i],
            weight_scale=self.shared_expert_ffn2_weights_scale[i],
            weight_dtype=self.weight_dtype,
            group_size=self.weightonly_group_size,
        )
        if self.config.moe_config.shared_expert_with_gate:
            gate_out = paddle.matmul(tmp_out, self.shared_expert_gate_weights[i])
            gate_out = paddle.nn.functional.sigmoid(gate_out)
            return gate_out * ffn2_out
        return ffn2_out


class FusedMultiTransformerWeightOnlyPostLayernorm(
    FusedMultiTransformerWeightOnly, FusedMultiTransformerPostLayernorm
):
    def __init__(self, config: FusedMultiTransformerConfig):
        super().__init__(config)


class FusedMultiTransformerAvx(Layer):
    def __init__(self, config: FusedMultiTransformerConfig):
        super().__init__()
        self.config = config
        assert config.embed_dim > 0, "Expected embed_dim to be greater than 0, " "but received {}".format(
            config.embed_dim
        )
        assert config.num_heads > 0, "Expected nhead to be greater than 0, " "but received {}".format(config.num_heads)
        assert config.intermediate_size > 0, "Expected intermediate_size to be greater than 0, but received {}".format(
            config.intermediate_size
        )
        self._dtype = "float32"
        self._epsilon = config.epsilon
        self._residual_alpha = config.residual_alpha
        self.norm_type = config.norm_type
        if self.norm_type != "layernorm" and self.norm_type != "rmsnorm":
            raise NotImplementedError("Only support norm type of [layernorm, rmsnorm]")

        self._norm_weight_dtype = "float32" if self.norm_type == "layernorm" else self._dtype

        self.embed_dim = config.embed_dim
        self.head_dim = config.embed_dim // config.num_heads
        assert self.head_dim * config.num_heads == config.embed_dim, "embed_dim must be divisible by num_heads"

        assert config.num_heads % config.nranks == 0
        assert config.intermediate_size % config.nranks == 0

        intermediate_size = config.intermediate_size
        self.num_heads = config.num_heads
        self.cache_dtype = self.config.avx_config.cache_dtype
        self.kv_num_heads = config.kv_num_heads
        self.num_layers = config.num_layers
        assert self.num_layers > 0
        if isinstance(config.qkv_weight_attrs, (list, tuple)):
            assert self.num_layers == len(config.qkv_weight_attrs)

        self.weight_dtype = self._dtype
        self.create_params_type = self._dtype
        self.activation = config.activation
        self.intermediate_size = intermediate_size
        self.max_positions = self.config.avx_config.max_position_embeddings
        self.max_pos_embed = self.config.avx_config.max_position_embeddings
        self.hiddensize = self.num_heads * self.head_dim
        self._compute_type = self.config.avx_config.compute_type

        self.ln_scales, self.ln_biases = [], []
        self.qkv_weights, self.qkv_biases = [], []
        self.linear_weights, self.linear_biases = [], []
        self.ffn_ln_scales, self.ffn_ln_biases = [], []

        self.ffn2_weights, self.ffn2_biases = [], []
        self.gate_weights, self.gate_biases = [], []
        self.up_weights, self.up_biases = [], []

        for i in range(self.num_layers):
            ln_scale_attr = self.get_attr(config.ln_scale_attrs, i)
            ln_bias_attr = self.get_attr(config.ln_bias_attrs, i)
            qkv_weight_attr = self.get_attr(config.qkv_weight_attrs, i)

            qkv_bias_attr = self.get_attr(config.qkv_bias_attrs, i)
            linear_weight_attr = self.get_attr(config.linear_weight_attrs, i)
            linear_bias_attr = self.get_attr(config.linear_bias_attrs, i)

            ffn_ln_scale_attr = self.get_attr(config.ffn_ln_scale_attrs, i)
            ffn_ln_bias_attr = self.get_attr(config.ffn_ln_bias_attrs, i)
            gate_weight_attr = self.get_attr(config.gate_weight_attrs, i)
            gate_bias_attr = self.get_attr(config.gate_bias_attrs, i)
            up_weight_attr = self.get_attr(config.up_weight_attrs, i)
            up_bias_attr = self.get_attr(config.up_bias_attrs, i)
            ffn2_weight_attr = self.get_attr(config.ffn2_weight_attrs, i)
            ffn2_bias_attr = self.get_attr(config.ffn2_bias_attrs, i)

            ln_scale = self.create_parameter(
                attr=ln_scale_attr,
                shape=[config.embed_dim],
                default_initializer=Constant(value=1.0),
                dtype=self._norm_weight_dtype,
            )
            ln_bias = None
            if ln_bias_attr:
                ln_bias = self.create_parameter(
                    attr=ln_bias_attr,
                    shape=[config.embed_dim],
                    is_bias=True,
                    dtype=self._norm_weight_dtype,
                )
            self.init_weight_shape(config)
            qkv_weight = self.create_parameter(
                shape=self.qkv_weight_shape,
                attr=qkv_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )

            qkv_bias = None
            if qkv_bias_attr:
                qkv_bias = self.create_parameter(
                    shape=[(self.num_heads + 2 * self.kv_num_heads) * self.head_dim],
                    attr=qkv_bias_attr,
                    dtype=self._dtype,
                    is_bias=True,
                )
            linear_weight = self.create_parameter(
                shape=self.linear_weight_shape,
                attr=linear_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )
            linear_bias = None
            if linear_bias_attr:
                linear_bias = self.create_parameter(
                    shape=[config.embed_dim],
                    attr=linear_bias_attr,
                    dtype=self._dtype,
                    is_bias=True,
                )
            ffn_ln_scale = self.create_parameter(
                shape=[config.embed_dim],
                attr=ffn_ln_scale_attr,
                is_bias=False,
                default_initializer=Constant(1.0),
                dtype=self._norm_weight_dtype,
            )

            ffn_ln_bias = None
            if ffn_ln_bias_attr:
                ffn_ln_bias = self.create_parameter(
                    shape=[config.embed_dim],
                    attr=ffn_ln_bias_attr,
                    is_bias=True,
                    dtype=self._norm_weight_dtype,
                )
            gate_weight = self.create_parameter(
                shape=self.gate_weight_shape,
                attr=gate_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )
            gate_bias = None
            if gate_bias_attr:
                gate_bias = self.create_parameter(
                    shape=[config.intermediate_size],
                    attr=gate_bias_attr,
                    dtype=self._dtype,
                    is_bias=True,
                )
            up_weight = self.create_parameter(
                shape=self.up_weight_shape,
                attr=up_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )
            up_bias = None
            if up_bias_attr:
                up_bias = self.create_parameter(
                    shape=[config.intermediate_size],
                    attr=up_bias_attr,
                    dtype=self._dtype,
                    is_bias=True,
                )
            ffn2_weight = self.create_parameter(
                shape=self.ffn2_weight_shape,
                attr=ffn2_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )
            ffn2_bias = None
            if ffn2_bias_attr:
                ffn2_bias = self.create_parameter(
                    shape=[config.embed_dim],
                    attr=ffn2_bias_attr,
                    dtype=self._dtype,
                    is_bias=True,
                )
            self.ln_scales.append(ln_scale)
            self.ln_biases.append(ln_bias)
            self.qkv_weights.append(qkv_weight)
            self.qkv_biases.append(qkv_bias)
            self.linear_weights.append(linear_weight)
            self.linear_biases.append(linear_bias)

            self.ffn_ln_scales.append(ffn_ln_scale)
            self.ffn_ln_biases.append(ffn_ln_bias)
            self.gate_weights.append(gate_weight)
            self.gate_biases.append(gate_bias)
            self.up_weights.append(up_weight)
            self.up_biases.append(up_bias)
            self.ffn2_weights.append(ffn2_weight)
            self.ffn2_biases.append(ffn2_bias)

            self._add_parameter(ln_scale)
            self._add_parameter(ln_bias)
            self._add_parameter(qkv_weight)
            self._add_parameter(qkv_bias)
            self._add_parameter(linear_weight)
            self._add_parameter(linear_bias)

            self._add_parameter(ffn_ln_scale)
            self._add_parameter(ffn_ln_bias)
            self._add_parameter(gate_weight)
            self._add_parameter(gate_bias)
            self._add_parameter(up_weight)
            self._add_parameter(up_bias)
            self._add_parameter(ffn2_weight)
            self._add_parameter(ffn2_bias)

    def get_attr(self, attrs, idx):
        """
        For fake parameter
        """
        if isinstance(attrs, (list, tuple)):
            assert (
                len(attrs) == self.num_layers
            ), f"length of attrs is {len(attrs)} is not equal to self.num_layers {self.num_layers}"
            return attrs[idx]
        return attrs

    def _add_parameter(self, param):
        """
        For fake parameter
        """
        if param is None:
            return
        assert param.name not in self._parameters
        self._parameters[param.name] = param

    def init_weight_shape(self, config):
        self.gate_weight_shape = [self.embed_dim, self.intermediate_size]
        self.up_weight_shape = [self.embed_dim, self.intermediate_size]
        self.down_weight_shape = [self.intermediate_size, self.embed_dim]
        self.qkv_weight_shape = [self.embed_dim, (self.num_heads + 2 * self.kv_num_heads) * self.head_dim]
        self.linear_weight_shape = [self.num_heads * self.head_dim, self.embed_dim]
        self.ffn2_weight_shape = [self.intermediate_size, self.embed_dim]

    def forward(
        self,
        input_ids,
        src,
        past_seq_len=None,
        cur_seq_len=None,
        step_idx=None,
        **kwargs,
    ):
        from paddlenlp_ops import xft_transformer

        xft_out = xft_transformer(
            paddle.cast(src, "float32"),  # input
            self.ln_scales,  # ln1Gamma
            self.qkv_weights,  # qkvWeight
            self.linear_weights,  # attnOutWeight
            self.ffn_ln_scales,  # ln2Gamma
            self.gate_weights,  # gateWeight
            self.up_weights,  # upWeight
            self.ffn2_weights,  # downWeight
            past_seq_len,  # pastSeqLen
            cur_seq_len,  # currentSeqLen
            step_idx,  # step
            self.hiddensize,  # hiddensize
            self.num_layers,  # totalLayer
            self._compute_type,  # computeType
            self.cache_dtype,  # cacheDtype
            self.activation,  # activation
            self.norm_type,  # normType
            self.head_dim,  # attHeadDim
            self.num_heads,  # attHeadNum
            self.kv_num_heads,  # kvHeadNum
            self.max_positions,  # maxPositions
            self.max_pos_embed,  # maxPosEmbed
            self.intermediate_size,  # intermediateSize
        )
        return xft_out[:, -1, :]


class FusedMultiTransformerA8W8(FusedMultiTransformerBase):
    def __init__(self, config: FusedMultiTransformerConfig):
        super().__init__(config)
        self.quant_round_type = config.quant_round_type
        self.quant_max_bound = config.quant_max_bound
        self.quant_min_bound = config.quant_min_bound
        self.use_gemm_dequant = False

        self.qkv_out_scales = []
        self.linear_out_scales = []
        self.ffn1_out_scales = []
        self.ffn2_out_scales = []

        self.linear_shifts, self.linear_smooths, self.ffn2_shifts, self.ffn2_smooths = [], [], [], []

        for i in range(self.num_layers):
            qkv_out_scale_attr = self.get_attr(config.qkv_out_scale_attrs, i)
            linear_out_scale_attr = self.get_attr(config.linear_out_scale_attrs, i)
            ffn1_out_scale_attr = self.get_attr(config.ffn1_out_scale_attrs, i)
            ffn2_out_scale_attr = self.get_attr(config.ffn2_out_scale_attrs, i)

            linear_shift_attr = self.get_attr(config.linear_shift_attrs, i)
            linear_smooth_attr = self.get_attr(config.linear_smooth_attrs, i)
            ffn2_shift_attr = self.get_attr(config.ffn2_shift_attrs, i)
            ffn2_smooth_attr = self.get_attr(config.ffn2_smooth_attrs, i)

            qkv_out_scale = self.create_parameter(
                shape=[self.head_dim * (2 * self.kv_num_heads + self.num_heads)],
                attr=qkv_out_scale_attr,
                dtype="float32",
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(-1),
            )
            linear_out_scale = self.create_parameter(
                shape=[self.embed_dim],
                attr=linear_out_scale_attr,
                dtype="float32",
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(-1),
            )
            ffn1_out_scale = self.create_parameter(
                shape=[self.intermediate_size * 2] if self.activation.endswith("glu") else [self.intermediate_size],
                attr=ffn1_out_scale_attr,
                dtype="float32",
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(-1),
            )
            ffn2_out_scale = self.create_parameter(
                shape=[self.embed_dim],
                attr=ffn2_out_scale_attr,
                dtype="float32",
                is_bias=False,
                default_initializer=paddle.nn.initializer.Constant(-1),
            )

            linear_shift = None
            if linear_shift_attr:
                linear_shift = self.create_parameter(
                    shape=[self.num_heads * self.head_dim], attr=linear_shift_attr, dtype=self._dtype, is_bias=False
                )

            linear_smooth = None
            if linear_smooth_attr:
                linear_smooth = self.create_parameter(
                    shape=[self.num_heads * self.head_dim], attr=linear_smooth_attr, dtype=self._dtype, is_bias=False
                )

            ffn2_shift = None
            if ffn2_shift_attr:
                ffn2_shift = self.create_parameter(
                    shape=[self.intermediate_size], attr=ffn2_shift_attr, dtype=self._dtype, is_bias=False
                )

            ffn2_smooth = None
            if ffn2_smooth_attr:
                ffn2_smooth = self.create_parameter(
                    shape=[self.intermediate_size], attr=ffn2_smooth_attr, dtype=self._dtype, is_bias=False
                )

            self.qkv_out_scales.append(qkv_out_scale)
            self.linear_out_scales.append(linear_out_scale)
            self.ffn1_out_scales.append(ffn1_out_scale)
            self.ffn2_out_scales.append(ffn2_out_scale)

            if linear_shift is not None:
                self.linear_shifts.append(linear_shift)
                self.linear_smooths.append(linear_smooth)
                self.ffn2_shifts.append(ffn2_shift)
                self.ffn2_smooths.append(ffn2_smooth)

            self._add_parameter(qkv_out_scale)
            self._add_parameter(linear_out_scale)
            self._add_parameter(ffn1_out_scale)
            self._add_parameter(ffn2_out_scale)

            self._add_parameter(linear_shift)
            self._add_parameter(linear_smooth)
            self._add_parameter(ffn2_shift)
            self._add_parameter(ffn2_smooth)

    def init_weight(self):
        self.qkv_weights = []
        self.linear_weights = []
        self.gate_weights = []
        self.ffn1_weights = []
        self.ffn2_weights = []

        for i in range(self.num_layers):
            qkv_weight_attr = self.get_attr(self.config.qkv_weight_attrs, i)
            linear_weight_attr = self.get_attr(self.config.linear_weight_attrs, i)
            gate_weight_attr = self.get_attr(self.config.gate_weight_attrs, i)
            ffn1_weight_attr = self.get_attr(self.config.ffn1_weight_attrs, i)
            ffn2_weight_attr = self.get_attr(self.config.ffn2_weight_attrs, i)

            qkv_weight = self.create_parameter(
                shape=self.qkv_weight_shape,
                attr=qkv_weight_attr,
                dtype=self.get_weight_create_dype("qkv_weight_scale", i),
                is_bias=False,
            )
            linear_weight = self.create_parameter(
                shape=self.linear_weight_shape,
                attr=linear_weight_attr,
                dtype=self.get_weight_create_dype("out_linear_weight_scale", i),
                is_bias=False,
            )

            gate_weight = None

            if self.config.moe_config.use_moe(i):
                gate_weight = self.create_parameter(
                    shape=[self.config.embed_dim, self.config.moe_config.num_experts],
                    attr=gate_weight_attr,
                    dtype="float32",
                    is_bias=False,
                    default_initializer=paddle.nn.initializer.Constant(0),
                )

            if self.config.moe_config.use_moe(i):
                ffn1_weight = self.create_parameter(
                    shape=self.moe_ffn1_weight_shape,
                    attr=ffn1_weight_attr,
                    dtype=self.create_params_type,
                    is_bias=False,
                )
            else:
                ffn1_weight = self.create_parameter(
                    shape=self.ffn1_weight_shape,
                    attr=ffn1_weight_attr,
                    dtype=self.get_weight_create_dype("ffn1_weight_scale", i),
                    is_bias=False,
                )
            if self.config.moe_config.use_moe(i):
                ffn2_weight = self.create_parameter(
                    shape=self.moe_ffn2_weight_shape,
                    attr=ffn2_weight_attr,
                    dtype=self.create_params_type,
                    is_bias=False,
                )
            else:
                ffn2_weight = self.create_parameter(
                    shape=self.ffn2_weight_shape,
                    attr=ffn2_weight_attr,
                    dtype=self.get_weight_create_dype("ffn2_weight_scale", i),
                    is_bias=False,
                )

            # tensor model parallel
            if self.config.nranks > 1:
                # column parallel
                _set_var_distributed(qkv_weight)
                _set_var_distributed(ffn1_weight)
                # row parallel
                _set_var_distributed(linear_weight)
                _set_var_distributed(ffn2_weight)

            self.qkv_weights.append(qkv_weight)
            self.linear_weights.append(linear_weight)

            self.gate_weights.append(gate_weight)
            self.ffn1_weights.append(ffn1_weight)
            self.ffn2_weights.append(ffn2_weight)

            self._add_parameter(qkv_weight)
            self._add_parameter(linear_weight)
            if gate_weight is not None:
                self._add_parameter(gate_weight)
            self._add_parameter(ffn1_weight)
            self._add_parameter(ffn2_weight)

    def skip_quant(self, layer_name, layer_idx):
        """
        Determine whether to skip quantization for a given layer based on weight scales.

        Parameters:
        - layer_name (str): The name of the layer to check.
        - layer_idx (int): The index of the specific layer to check.

        Returns:
        - bool: True if quantization should be skipped, False otherwise.
        """
        return hasattr(self, "weight_scales") and np.all(self.weight_scales[layer_name][layer_idx] == -1)

    def get_weight_create_dype(self, layer_name=None, layer_idx=None):
        if layer_name is not None and layer_idx is not None:
            if self.skip_quant(layer_name, layer_idx):
                return self._dtype
        return "int8"

    def init_weight_shape(self, config):
        super().init_weight_shape(config)

        if not paddle.is_compiled_with_rocm():
            self.linear_weight_shape = [self.embed_dim, self.num_heads * self.head_dim]
            self.ffn1_weight_shape = (
                [self.intermediate_size * 2, self.embed_dim]
                if self.activation.endswith("glu")
                else [self.intermediate_size, self.embed_dim]
            )
            self.ffn2_weight_shape = [self.embed_dim, self.intermediate_size]

    def compute_layernorm_before_qkv(self, src, i):
        if i == 0:
            ln_out = self.norm_func(
                src,
                self.ln_scales[i],
                self.ln_biases[i],
                self._epsilon,
                begin_norm_axis=1,
                quant_scale=self.act_scales["qkv_in_scale"][i],  # quant_in_scale
                quant_round_type=self.quant_round_type,
                quant_max_bound=self.quant_max_bound,
                quant_min_bound=self.quant_min_bound,
            )[0]
        else:
            ln_out = src

        return ln_out

    def compute_qkv_linear(self, ln_out, i):
        if self.config.mla_config.use_mla():
            raise NotImplementedError("Not support MLA yet.")
        else:
            if paddle.is_compiled_with_rocm():
                qkv_out = paddle.matmul(ln_out, self.qkv_weights[i])
            else:
                qkv_out = paddle.matmul(ln_out, self.qkv_weights[i], False, True)
            return qkv_out

    def compute_fmha(
        self,
        qkv_out,
        padding_offset,
        seq_lens,
        input_ids,
        rotary_embs,
        rotary_emb_dims,
        caches,
        pre_caches,
        pre_caches_length,
        attn_mask,
        i,
    ):
        if not self.skip_quant("qkv_weight_scale", i):
            qkv_out = dequant_int8(qkv_out, self.qkv_out_scales[i], self._dtype)
        if self.qkv_biases[i] is not None:
            qkv_out = paddle.add(qkv_out, self.qkv_biases[i])

        bsz = input_ids.shape[0]
        """
        qkv: bsz, seq_len, 3, numhead, headsize ->
        q_out: bsz, numhead, seq_len, headsize
        kv_out: 2, bsz, numhead, seq_len, headsize
        """
        q_out, k_out, v_out = qkv_transpose_split(
            qkv_out, padding_offset, seq_lens, input_ids, self.num_heads, self.head_dim
        )

        # rotary emb (inplace)
        if rotary_embs is not None:
            encode_rotary_qk(
                q_out,
                k_out,
                rotary_embs,
                seq_lens,
                rotary_emb_dims=rotary_emb_dims,
                use_neox=self.use_neox_rotary_style,
            )

        if pre_caches is not None:
            k_out = paddle.concat([pre_caches[i][0, :bsz], k_out], axis=2)
            v_out = paddle.concat([pre_caches[i][1, :bsz], v_out], axis=2)

        # write cache kv (inplace)
        write_cache_kv(k_out, v_out, caches[i], seq_lens + pre_caches_length)

        # cutlass fmha
        qktv_out = variable_length_memory_efficient_attention(
            q_out,
            k_out,
            v_out,
            seq_lens,
            seq_lens + pre_caches_length,
            mask=attn_mask,
            scale=self.softmax_scale,
        )

        fmha_out = transpose_remove_padding(qktv_out, seq_lens, padding_offset)
        fmha_out = quant_int8(
            fmha_out,
            self.linear_shifts[i] if len(self.linear_shifts) > 0 else None,
            self.linear_smooths[i] if len(self.linear_smooths) > 0 else None,
            self.act_scales["out_linear_in_scale"][i],
            self.quant_round_type,
            self.quant_max_bound,
            self.quant_min_bound,
        )
        return fmha_out

    def compute_mmha(self, qkv_out, caches, attn_mask, seq_lens, rotary_embs, rotary_emb_dims, i):
        return masked_multihead_attention(
            x=qkv_out,
            bias=self.qkv_biases[i],
            cache_kv=caches[i],
            src_mask=attn_mask,
            sequence_lengths=seq_lens,
            rotary_tensor=rotary_embs,
            rotary_emb_dims=rotary_emb_dims,
            use_neox_rotary_style=self.use_neox_rotary_style,
            qkv_out_scale=self.qkv_out_scales[i],
            out_shift=self.linear_shifts[i] if len(self.linear_shifts) > 0 else None,
            out_smooth=self.linear_smooths[i] if len(self.linear_smooths) > 0 else None,
            out_scale=self.act_scales["out_linear_in_scale"][i],
            quant_round_type=self.quant_round_type,
            quant_max_bound=self.quant_max_bound,
            quant_min_bound=self.quant_min_bound,
            compute_dtype=self._fuse_kernel_compute_dtype,
        )[0]

    def compute_out_linear(self, fmha_out, i):
        if self.skip_quant("out_linear_weight_scale", i):
            if paddle.is_compiled_with_rocm():
                out_linear_out = paddle.matmul(fmha_out, self.linear_weights[i])
            else:
                out_linear_out = paddle.matmul(fmha_out, self.linear_weights[i], False, True)
        else:
            if paddle.is_compiled_with_rocm():
                out_linear_out = paddle.matmul(fmha_out, self.linear_weights[i])
                out_linear_out = dequant_int8(out_linear_out, self.linear_out_scales[i], self._dtype)
            else:
                if self.use_gemm_dequant:
                    from paddlenlp_ops import gemm_dequant

                    out_linear_out = gemm_dequant(
                        fmha_out, self.linear_weights[i], self.linear_out_scales[i], self._dtype
                    )
                else:
                    out_linear_out = paddle.matmul(fmha_out, self.linear_weights[i], False, True)
                    out_linear_out = dequant_int8(out_linear_out, self.linear_out_scales[i], self._dtype)
        return out_linear_out

    def compute_ffn_layernorm(self, out_linear_out, residual_input, i):
        norm_out = self.norm_func(
            out_linear_out,
            self.ffn_ln_scales[i],
            self.ffn_ln_biases[i],
            self._epsilon,
            bias=self.linear_biases[i],
            residual=residual_input,
            begin_norm_axis=1,
            quant_scale=self.act_scales["ffn1_in_scale"][i],  # quant_in_scale
            quant_round_type=self.quant_round_type,
            quant_max_bound=self.quant_max_bound,
            quant_min_bound=self.quant_min_bound,
        )
        tmp_out, residual_input = norm_out[0], norm_out[1]

        return tmp_out, residual_input

    def compute_activation(self, ffn1_out, i):
        return fused_bias_act(
            ffn1_out,
            self.ffn1_biases[i],
            act_method=self.activation,
            compute_dtype=self._fuse_kernel_compute_dtype,
            dequant_scales=self.ffn1_out_scales[i],
            shift=self.ffn2_shifts[i] if len(self.ffn2_shifts) > 0 else None,
            smooth=self.ffn2_smooths[i] if len(self.ffn2_smooths) > 0 else None,
            quant_scale=self.act_scales["ffn2_in_scale"][i],
            quant_round_type=self.quant_round_type,
            quant_max_bound=self.quant_max_bound,
            quant_min_bound=self.quant_min_bound,
        )

    def compute_ffn1(self, tmp_out, i):
        if paddle.device.is_compiled_with_rocm():
            return paddle.matmul(tmp_out, self.ffn1_weights[i])
        else:
            return paddle.matmul(tmp_out, self.ffn1_weights[i], False, True)

    def compute_ffn2(self, ffn1_out, i):
        if self.skip_quant("ffn2_weight_scale", i):
            if paddle.device.is_compiled_with_rocm():
                ffn2_out = paddle.matmul(ffn1_out, self.ffn2_weights[i])
            else:
                ffn2_out = paddle.matmul(ffn1_out, self.ffn2_weights[i], False, True)
        else:
            if paddle.device.is_compiled_with_rocm():
                ffn2_out = paddle.matmul(ffn1_out, self.ffn2_weights[i])
                ffn2_out = dequant_int8(ffn2_out, self.ffn2_out_scales[i], self._dtype)
            else:
                if self.use_gemm_dequant:
                    from paddlenlp_ops import gemm_dequant

                    ffn2_out = gemm_dequant(ffn1_out, self.ffn2_weights[i], self.ffn2_out_scales[i], self._dtype)
                else:
                    ffn2_out = paddle.matmul(ffn1_out, self.ffn2_weights[i], False, True)
                    ffn2_out = dequant_int8(ffn2_out, self.ffn2_out_scales[i], self._dtype)
        return ffn2_out

    def compute_bias_residual_layernorm(self, ffn2_out, residual_input, i, num_layers):
        if i != num_layers - 1:
            norm_out = self.norm_func(
                ffn2_out,
                self.ln_scales[i + 1],
                self.ln_biases[i + 1],
                self._epsilon,
                residual=residual_input,
                begin_norm_axis=1,
                quant_scale=self.act_scales["qkv_in_scale"][i + 1],
                quant_round_type=self.quant_round_type,
                quant_max_bound=self.quant_max_bound,
                quant_min_bound=self.quant_min_bound,
            )
            tmp_out, residual_input = norm_out[0], norm_out[1]
        else:
            tmp_out = fused_layer_norm(
                ffn2_out,
                norm_weight=None,
                norm_bias=None,
                epsilon=self._epsilon,
                begin_norm_axis=1,
                bias=self.ffn2_biases[i],
                residual=residual_input,
            )[0]
        return tmp_out, residual_input


class FusedBlockMultiTransformer(FusedMultiTransformerBase):
    def __init__(self, config: FusedMultiTransformerConfig):
        super().__init__(config)
        if paddle.is_compiled_with_xpu():
            self.cache_k_per_batch_maxs = paddle.full(shape=[10, 6], fill_value=0, dtype="float32")
            self.cache_v_per_batch_maxs = paddle.full(shape=[10, 6], fill_value=0, dtype="float32")

    def compute_mla_absorb(
        self,
        qkv_out,
        caches,
        i,
        **kwargs,
    ):
        from paddlenlp_ops import decode_mla_write_cache, multi_head_latent_attention

        ln_out = qkv_out
        latent_cache = caches[i]

        out_linear_out = paddle.zeros(shape=[ln_out.shape[0], self.embed_dim], dtype=ln_out.dtype)

        if kwargs["max_enc_len_this_time"]:  # prefill phase
            query, key, value = self.compute_qkv_linear(ln_out, i, latent_cache=latent_cache, **kwargs)

            from paddlenlp.utils.env import PREFILL_USE_SAGE_ATTN

            if PREFILL_USE_SAGE_ATTN:
                from paddlenlp_ops import sage_attention_dsk

                query_256 = paddle.nn.functional.pad(paddle.unsqueeze(query, axis=0), (0, 256 - 192))
                key_256 = paddle.nn.functional.pad(paddle.unsqueeze(key, axis=0), (0, 256 - 192))

                value_128, _ = paddle.split(value, [128, 64], axis=-1)
                value_128 = paddle.unsqueeze(value_128, axis=0)

                km = paddle.mean(key_256, axis=1, keepdim=True)
                km = km.squeeze(1)

                fmha_out_prefill = sage_attention_dsk(
                    query_256,
                    key_256,
                    value_128,
                    km,
                    kwargs.get("cu_seqlens_q", None),
                    None,  # vm
                    self.softmax_scale,
                    "per_warp",  # qk_quant_gran
                    "fp16",  # pv_accum_dtype
                    0,  # tensor layout, 0 -> NHD
                    True,  # is_causal
                    True,  # smooth k
                    False,  # smooth v
                    False,  # return lse
                )
                fmha_out_prefill = paddle.nn.functional.pad(fmha_out_prefill, (0, 192 - 128))
                fmha_out_prefill = paddle.squeeze(fmha_out_prefill, axis=0)
            else:
                fmha_out_prefill = paddle.nn.functional.flash_attention.flash_attn_unpadded(
                    query,
                    key,
                    value,
                    kwargs.get("cu_seqlens_q", None),
                    kwargs.get("cu_seqlens_k", None),
                    kwargs.get("max_enc_len_this_time", -1),
                    kwargs.get("max_enc_len_this_time", -1),
                    self.softmax_scale,
                    causal=True,
                    training=False,
                )[0]

            fmha_out_prefill = fmha_out_prefill.reshape([-1, self.num_heads, self.config.mla_config.qk_head_dim])
            fmha_out_prefill = fmha_out_prefill[:, :, : self.config.mla_config.v_head_dim]
            fmha_out_prefill = fmha_out_prefill.reshape([-1, self.num_heads * self.config.mla_config.v_head_dim])

            fmha_out_prefill = fmha_out_prefill * self.mask_encoder_batch.cast(fmha_out_prefill.dtype)

            out_linear_out_prefill = self.compute_out_linear(fmha_out_prefill, i)
            out_linear_out = out_linear_out + out_linear_out_prefill

            # print(f"prefill {i}: out_linear_out: {out_linear_out}")

        if kwargs["max_dec_len_this_time"]:  # decode phase
            if self.config.mla_config.q_lora_rank is not None:
                query = paddle.matmul(ln_out, self.q_a_proj_weights[i])
                query = self.norm_func(
                    x=query,
                    norm_weight=self.q_a_layernorm_weights[i],
                    norm_bias=None,
                    epsilon=self._epsilon,
                    begin_norm_axis=1,
                )[0]
                ln_out_or_q_c = query
            else:
                ln_out_or_q_c = ln_out

            compressed_kv = paddle.matmul(ln_out, self.kv_a_proj_with_mqa_weights[i])
            compressed_kv, key_pe = compressed_kv.split(
                [self.config.mla_config.kv_lora_rank, self.config.mla_config.qk_rope_head_dim], axis=-1
            )
            key_pe = key_pe.reshape([-1, 1, self.config.mla_config.qk_rope_head_dim])
            compressed_kv = self.norm_func(
                x=compressed_kv,
                norm_weight=self.kv_a_layernorm_weights[i],
                norm_bias=None,
                epsilon=self._epsilon,
                begin_norm_axis=1,
            )[0]

            query_nope = paddle.matmul(ln_out_or_q_c, self.q_nope_k_b_proj_weights[i])
            query_nope = query_nope.reshape(shape=[-1, self.num_heads, self.config.mla_config.kv_lora_rank])
            query_pe = paddle.matmul(ln_out_or_q_c, self.q_rope_proj_weights[i])
            query_pe = query_pe.reshape(shape=[-1, self.num_heads, self.config.mla_config.qk_rope_head_dim])
            query_pe, key_pe = self.config.rotary_emb(self.position_ids, query_pe, key_pe)

            decode_mla_write_cache(
                compressed_kv,
                key_pe,
                latent_cache,
                kwargs.get("seq_lens_decoder", None),
                kwargs.get("seq_lens_encoder", None),
                kwargs.get("padding_offsets", None),
                kwargs.get("cum_offsets", None),
                kwargs.get("block_tables", None),
                "none",
                kwargs.get("max_input_length", -1),
                self.config.speculate_config.speculate_method is not None,  # speculate_decoder
            )

            q_input = paddle.concat([query_nope, query_pe], axis=-1)
            q_input = q_input.reshape(
                [
                    -1,
                    self.num_heads * (self.config.mla_config.kv_lora_rank + self.config.mla_config.qk_rope_head_dim),
                ]
            )

            fmha_out_decode = multi_head_latent_attention(
                q_input,
                latent_cache,
                latent_cache,
                kwargs.get("seq_lens_encoder", None),
                kwargs.get("seq_lens_decoder", None),
                kwargs.get("seq_lens_this_time", None),
                kwargs.get("cu_seqlens_q", None),
                kwargs.get("padding_offsets", None),
                kwargs.get("cum_offsets", None),
                kwargs.get("block_tables", None),
                kwargs.get("encoder_batch_ids", None),
                kwargs.get("encoder_tile_ids_per_batch", None),
                kwargs.get("encoder_num_blocks", None),
                kwargs.get("kv_batch_ids", None),
                kwargs.get("kv_tile_ids_per_batch", None),
                kwargs.get("kv_num_blocks", None),
                kwargs.get("decoder_batch_ids", None),
                kwargs.get("decoder_tile_ids_per_batch", None),
                kwargs.get("decoder_num_blocks", None),
                kwargs.get("decoder_num_blocks_cpu", None),
                kwargs.get("max_enc_len_this_time", None),
                kwargs.get("max_dec_len_this_time", None),
                kwargs.get("max_len_kv", None),
                None,  # attn_mask
                None,  # qkv_bias
                None,  # qkv_out_scales
                None,  # cache_k_quant_scales
                None,  # cache_v_quant_scales
                None,  # cache_k_dequant_scales
                None,  # cache_v_dequant_scales
                None,  # cache_k_zp
                None,  # cache_v_zp
                None,  # out_shifts
                None,  # out_smooths
                self._fuse_kernel_compute_dtype,
                "none",  # cache_quant_type
                self.config.mla_config.kv_lora_rank,
                kwargs.get("max_input_length", -1),
                self.softmax_scale,  # softmax_scale
                0.0,  # quant_max_bound
                0.0,  # quant_min_bound
                0.0,  # out_linear_in_scale
                self.config.speculate_config.speculate_max_draft_token_num,
                True,  # causal
                self.config.speculate_config.speculate_method is not None,  # speculate_decoder
            )

            out_linear_out_decode = paddle.matmul(fmha_out_decode, self.v_b_o_proj_weights[i])
            out_linear_out = out_linear_out + out_linear_out_decode

            # print(f"decode {i}: out_linear_out: {out_linear_out}")

        return out_linear_out

    def compute_attn(
        self,
        time_step,
        qkv_out,
        padding_offset,
        seq_lens,
        input_ids,
        rotary_embs,
        rotary_emb_dims,
        caches,
        pre_caches,
        pre_caches_length,
        attn_mask,
        i,
        **kwargs,
    ):
        if self.config.mla_config.use_absorb():
            return self.compute_mla_absorb(qkv_out, caches, i, **kwargs)

        if self.config.append_attn:
            from paddlenlp_ops import append_attention

            fmha_out = append_attention(
                qkv_out,
                caches[2 * i],
                caches[2 * i + 1],
                kwargs.get("seq_lens_encoder", None),
                kwargs.get("seq_lens_decoder", None),
                kwargs.get("seq_lens_this_time", None),
                kwargs.get("padding_offsets", None),
                kwargs.get("cum_offsets", None),
                kwargs.get("block_tables", None),
                kwargs.get("encoder_batch_ids", None),
                kwargs.get("encoder_tile_ids_per_batch", None),
                kwargs.get("encoder_num_blocks", None),
                kwargs.get("kv_batch_ids", None),
                kwargs.get("kv_tile_ids_per_batch", None),
                kwargs.get("kv_num_blocks", None),
                kwargs.get("decoder_batch_ids", None),
                kwargs.get("decoder_tile_ids_per_batch", None),
                kwargs.get("decoder_num_blocks_cpu", None),
                kwargs.get("max_enc_len_this_time", None),
                kwargs.get("max_dec_len_this_time", None),
                kwargs.get("max_len_kv", None),
                rotary_embs,
                None,  # attn_mask
                None,  # qkv_bias
                None,  # qkv_out_scales
                None,  # cache_k_quant_scales
                None,  # cache_v_quant_scales
                None,  # cache_k_dequant_scales
                None,  # cache_v_dequant_scales
                None,  # cache_k_zp
                None,  # cache_v_zp
                None,  # out_shifts
                None,  # out_smooths
                self._fuse_kernel_compute_dtype,
                "none",  # cache_quant_type
                self.use_neox_rotary_style,
                kwargs.get("max_input_length", -1),
                self.softmax_scale,  # softmax_scale
                0.0,  # quant_max_bound
                0.0,  # quant_min_bound
                0.0,  # out_linear_in_scale
                self.config.speculate_config.speculate_max_draft_token_num,
                True,  # causal
                self.config.speculate_config.speculate_method is not None,  # speculate_decoder
            )[0]
        else:
            if paddle.is_compiled_with_xpu():
                fmha_out = paddle.incubate.nn.functional.block_multihead_attention_xpu(
                    qkv_out,
                    caches[2 * i],
                    caches[2 * i + 1],
                    kwargs.get("seq_lens_encoder", None),
                    kwargs.get("seq_lens_decoder", None),
                    kwargs.get("seq_lens_this_time", None),
                    kwargs.get("padding_offsets", None),
                    kwargs.get("cum_offsets", None),
                    kwargs.get("cu_seqlens_q", None),
                    kwargs.get("cu_seqlens_k", None),
                    kwargs.get("block_tables", None),
                    self.cache_k_per_batch_maxs,
                    self.cache_v_per_batch_maxs,
                    pre_caches[2 * i] if pre_caches is not None else None,  # pre_key_cache
                    pre_caches[2 * i + 1] if pre_caches is not None else None,  # pre_value_cache
                    None,  # k_quant_scale
                    None,  # v_quant_scale
                    None,  # k_dequant_scale
                    None,  # v_dequant_scale
                    None,  # qkv_out_scales
                    None,  # qkv_bias
                    None,  # out_shifts
                    None,  # out_smooths
                    kwargs.get("max_enc_len_this_time", None),
                    kwargs.get("max_dec_len_this_time", None),
                    rotary_embs,
                    attn_mask,
                    kwargs.get("tgt_mask", None),
                    kwargs.get("max_input_length", -1),
                    kwargs.get("block_size", 64),
                    self.use_neox_rotary_style,
                    self.config.cachekv_int8_type == "dynamic",
                    quant_round_type=self.config.quant_round_type,
                    quant_max_bound=self.config.quant_max_bound,
                    quant_min_bound=self.config.quant_min_bound,
                    rope_theta=self.config.rope_theta,
                )[0]
            else:
                k_quant_scales = kwargs.get("k_quant_scales", None)
                v_quant_scales = kwargs.get("v_quant_scales", None)
                k_dequant_scales = kwargs.get("k_dequant_scales", None)
                v_dequant_scales = kwargs.get("v_dequant_scales", None)

                fmha_out = paddle.incubate.nn.functional.block_multihead_attention(
                    qkv_out,
                    caches[2 * i],
                    caches[2 * i + 1],
                    kwargs.get("seq_lens_encoder", None),
                    kwargs.get("seq_lens_decoder", None),
                    kwargs.get("seq_lens_this_time", None),
                    kwargs.get("padding_offsets", None),
                    kwargs.get("cum_offsets", None),
                    kwargs.get("cu_seqlens_q", None),
                    kwargs.get("cu_seqlens_k", None),
                    kwargs.get("block_tables", None),
                    pre_caches[2 * i] if pre_caches is not None else None,  # pre_key_cache
                    pre_caches[2 * i + 1] if pre_caches is not None else None,  # pre_value_cache
                    k_quant_scales[i] if k_quant_scales is not None else None,
                    v_quant_scales[i] if v_quant_scales is not None else None,
                    k_dequant_scales[i] if k_dequant_scales is not None else None,
                    v_dequant_scales[i] if v_dequant_scales is not None else None,
                    None,  # qkv_out_scales
                    None,  # qkv_bias
                    None,  # out_shifts
                    None,  # out_smooths
                    kwargs.get("max_enc_len_this_time", None),
                    kwargs.get("max_dec_len_this_time", None),
                    rotary_embs,
                    attn_mask,
                    kwargs.get("tgt_mask", None),
                    kwargs.get("max_input_length", -1),
                    kwargs.get("block_size", 64),
                    self.use_neox_rotary_style,
                    self.config.cachekv_int8_type == "dynamic",
                    quant_round_type=self.config.quant_round_type,
                    quant_max_bound=self.config.quant_max_bound,
                    quant_min_bound=self.config.quant_min_bound,
                    rope_theta=self.config.rope_theta,
                )[0]

        if self.config.mla_config.use_mla():
            fmha_out = fmha_out.reshape([-1, self.num_heads * self.config.mla_config.v_head_dim])

        return fmha_out

    def post_process(self, **kwargs):
        multi_block_output = kwargs.get("multi_block_output", None)
        cum_offsets = kwargs.get("cum_offsets", None)
        seq_lens_encoder = kwargs.get("seq_lens_encoder", None)
        seq_lens_decoder = kwargs.get("seq_lens_decoder", None)
        max_input_length = kwargs.get("max_input_length", -1)
        output_padding_offset = kwargs.get("output_padding_offset", None)  # only used in speculative decoding

        if self.config.speculate_config.return_full_hidden_states:
            return multi_block_output
        else:
            out = rebuild_padding_v2(
                multi_block_output,
                cum_offsets,
                seq_lens_decoder,
                seq_lens_encoder,
                output_padding_offset,
                max_input_length,
            )
            return out


class FusedBlockMultiTransformerWeightOnly(FusedBlockMultiTransformer, FusedMultiTransformerWeightOnly):
    def __init__(self, config: FusedMultiTransformerConfig):
        super().__init__(config)

    def compute_mla_absorb(
        self,
        qkv_out,
        caches,
        i,
        **kwargs,
    ):
        from paddlenlp_ops import decode_mla_write_cache, multi_head_latent_attention

        ln_out = qkv_out
        latent_cache = caches[i]

        out_linear_out = paddle.zeros(shape=[ln_out.shape[0], self.embed_dim], dtype=ln_out.dtype)

        if kwargs["max_enc_len_this_time"]:  # prefill phase
            query, key, value = self.compute_qkv_linear(ln_out, i, latent_cache=latent_cache, **kwargs)

            from paddlenlp.utils.env import PREFILL_USE_SAGE_ATTN

            if PREFILL_USE_SAGE_ATTN:
                from paddlenlp_ops import sage_attention_dsk

                query_256 = paddle.nn.functional.pad(paddle.unsqueeze(query, axis=0), (0, 256 - 192))
                key_256 = paddle.nn.functional.pad(paddle.unsqueeze(key, axis=0), (0, 256 - 192))

                value_128, _ = paddle.split(value, [128, 64], axis=-1)
                value_128 = paddle.unsqueeze(value_128, axis=0)

                km = paddle.mean(key_256, axis=1, keepdim=True)
                km = km.squeeze(1)

                fmha_out_prefill = sage_attention_dsk(
                    query_256,
                    key_256,
                    value_128,
                    km,
                    kwargs.get("cu_seqlens_q", None),
                    None,  # vm
                    self.softmax_scale,
                    "per_warp",  # qk_quant_gran
                    "fp16",  # pv_accum_dtype
                    0,  # tensor layout, 0 -> NHD
                    True,  # is_causal
                    True,  # smooth k
                    False,  # smooth v
                    False,  # return lse
                )
                fmha_out_prefill = paddle.nn.functional.pad(fmha_out_prefill, (0, 192 - 128))
                fmha_out_prefill = paddle.squeeze(fmha_out_prefill, axis=0)
            else:
                fmha_out_prefill = paddle.nn.functional.flash_attention.flash_attn_unpadded(
                    query,
                    key,
                    value,
                    kwargs.get("cu_seqlens_q", None),
                    kwargs.get("cu_seqlens_k", None),
                    kwargs.get("max_enc_len_this_time", -1),
                    kwargs.get("max_enc_len_this_time", -1),
                    self.softmax_scale,
                    causal=True,
                    training=False,
                )[0]

            fmha_out_prefill = fmha_out_prefill.reshape([-1, self.num_heads, self.config.mla_config.qk_head_dim])
            fmha_out_prefill = fmha_out_prefill[:, :, : self.config.mla_config.v_head_dim]
            fmha_out_prefill = fmha_out_prefill.reshape([-1, self.num_heads * self.config.mla_config.v_head_dim])

            fmha_out_prefill = fmha_out_prefill * self.mask_encoder_batch.cast(fmha_out_prefill.dtype)

            out_linear_out_prefill = self.compute_out_linear(fmha_out_prefill, i)
            out_linear_out = out_linear_out + out_linear_out_prefill

            # print(f"prefill {i}: out_linear_out: {out_linear_out}")

        if kwargs["max_dec_len_this_time"]:  # decode phase
            if self.config.mla_config.q_lora_rank is not None:
                query = weight_only_linear(
                    ln_out,
                    weight=self.q_a_proj_weights[i],
                    weight_scale=self.q_a_proj_weights_scale[i],
                    weight_dtype=self.weight_dtype,
                    group_size=self.weightonly_group_size,
                )
                query = self.norm_func(
                    x=query,
                    norm_weight=self.q_a_layernorm_weights[i],
                    norm_bias=None,
                    epsilon=self._epsilon,
                    begin_norm_axis=1,
                )[0]
                ln_out_or_q_c = query
            else:
                ln_out_or_q_c = ln_out

            compressed_kv = weight_only_linear(
                ln_out,
                weight=self.kv_a_proj_with_mqa_weights[i],
                weight_scale=self.kv_a_proj_with_mqa_weights_scale[i],
                weight_dtype=self.weight_dtype,
                group_size=self.weightonly_group_size,
            )
            compressed_kv, key_pe = compressed_kv.split(
                [self.config.mla_config.kv_lora_rank, self.config.mla_config.qk_rope_head_dim], axis=-1
            )
            key_pe = key_pe.reshape([-1, 1, self.config.mla_config.qk_rope_head_dim])
            compressed_kv = self.norm_func(
                x=compressed_kv,
                norm_weight=self.kv_a_layernorm_weights[i],
                norm_bias=None,
                epsilon=self._epsilon,
                begin_norm_axis=1,
            )[0]

            query_nope = weight_only_linear(
                ln_out_or_q_c,
                weight=self.q_nope_k_b_proj_weights[i],
                weight_scale=self.q_nope_k_b_proj_weights_scale[i],
                weight_dtype=self.weight_dtype,
                group_size=self.weightonly_group_size,
            )
            query_nope = query_nope.reshape(shape=[-1, self.num_heads, self.config.mla_config.kv_lora_rank])
            query_pe = weight_only_linear(
                ln_out_or_q_c,
                weight=self.q_rope_proj_weights[i],
                weight_scale=self.q_rope_proj_weights_scale[i],
                weight_dtype=self.weight_dtype,
                group_size=self.weightonly_group_size,
            )
            query_pe = query_pe.reshape(shape=[-1, self.num_heads, self.config.mla_config.qk_rope_head_dim])
            query_pe, key_pe = self.config.rotary_emb(self.position_ids, query_pe, key_pe)

            decode_mla_write_cache(
                compressed_kv,
                key_pe,
                latent_cache,
                kwargs.get("seq_lens_decoder", None),
                kwargs.get("seq_lens_encoder", None),
                kwargs.get("padding_offsets", None),
                kwargs.get("cum_offsets", None),
                kwargs.get("block_tables", None),
                "none",
                kwargs.get("max_input_length", -1),
                self.config.speculate_config.speculate_method is not None,  # speculate_decoder
            )

            q_input = paddle.concat([query_nope, query_pe], axis=-1)
            q_input = q_input.reshape(
                [
                    -1,
                    self.num_heads * (self.config.mla_config.kv_lora_rank + self.config.mla_config.qk_rope_head_dim),
                ]
            )

            fmha_out_decode = multi_head_latent_attention(
                q_input,
                latent_cache,
                latent_cache,
                kwargs.get("seq_lens_encoder", None),
                kwargs.get("seq_lens_decoder", None),
                kwargs.get("seq_lens_this_time", None),
                kwargs.get("cu_seqlens_q", None),
                kwargs.get("padding_offsets", None),
                kwargs.get("cum_offsets", None),
                kwargs.get("block_tables", None),
                kwargs.get("encoder_batch_ids", None),
                kwargs.get("encoder_tile_ids_per_batch", None),
                kwargs.get("encoder_num_blocks", None),
                kwargs.get("kv_batch_ids", None),
                kwargs.get("kv_tile_ids_per_batch", None),
                kwargs.get("kv_num_blocks", None),
                kwargs.get("decoder_batch_ids", None),
                kwargs.get("decoder_tile_ids_per_batch", None),
                kwargs.get("decoder_num_blocks", None),
                kwargs.get("decoder_num_blocks_cpu", None),
                kwargs.get("max_enc_len_this_time", None),
                kwargs.get("max_dec_len_this_time", None),
                kwargs.get("max_len_kv", None),
                None,  # attn_mask
                None,  # qkv_bias
                None,  # qkv_out_scales
                None,  # cache_k_quant_scales
                None,  # cache_v_quant_scales
                None,  # cache_k_dequant_scales
                None,  # cache_v_dequant_scales
                None,  # cache_k_zp
                None,  # cache_v_zp
                None,  # out_shifts
                None,  # out_smooths
                self._fuse_kernel_compute_dtype,
                "none",  # cache_quant_type
                self.config.mla_config.kv_lora_rank,
                kwargs.get("max_input_length", -1),
                self.softmax_scale,  # softmax_scale
                0.0,  # quant_max_bound
                0.0,  # quant_min_bound
                0.0,  # out_linear_in_scale
                self.config.speculate_config.speculate_max_draft_token_num,
                True,  # causal
                self.config.speculate_config.speculate_method is not None,  # speculate_decoder
            )

            out_linear_out_decode = weight_only_linear(
                fmha_out_decode,
                weight=self.v_b_o_proj_weights[i],
                weight_scale=self.v_b_o_proj_weights_scale[i],
                weight_dtype=self.weight_dtype,
                group_size=self.weightonly_group_size,
            )
            out_linear_out = out_linear_out + out_linear_out_decode

            # print(f"decode {i}: out_linear_out: {out_linear_out}")

        return out_linear_out


class FusedBlockMultiTransformerA8W8(FusedBlockMultiTransformer, FusedMultiTransformerA8W8):
    def __init__(self, config: FusedMultiTransformerConfig):
        super().__init__(config)

    def compute_attn(
        self,
        time_step,
        qkv_out,
        padding_offset,
        seq_lens,
        input_ids,
        rotary_embs,
        rotary_emb_dims,
        caches,
        pre_caches,
        pre_caches_length,
        attn_mask,
        i,
        **kwargs,
    ):
        k_quant_scales = kwargs.get("k_quant_scales", None)
        v_quant_scales = kwargs.get("v_quant_scales", None)
        k_dequant_scales = kwargs.get("k_dequant_scales", None)
        v_dequant_scales = kwargs.get("v_dequant_scales", None)
        cache_k_zps = kwargs.get("cache_k_zp", None)
        cache_v_zps = kwargs.get("cache_v_zp", None)

        cache_quant_type_str = "none"
        if self.config.cachekv_int8_type == "static":
            k_quant_scales = self.cache_k_scales
            v_quant_scales = self.cache_v_scales
            k_dequant_scales = self.cache_k_out_scales
            v_dequant_scales = self.cache_v_out_scales
            cache_quant_type_str = "cache_int8"

        if self.config.append_attn:
            from paddlenlp_ops import append_attention

            fmha_out = append_attention(
                qkv_out,
                caches[2 * i],
                caches[2 * i + 1],
                kwargs.get("seq_lens_encoder", None),
                kwargs.get("seq_lens_decoder", None),
                kwargs.get("seq_lens_this_time", None),
                kwargs.get("padding_offsets", None),
                kwargs.get("cum_offsets", None),
                kwargs.get("block_tables", None),
                kwargs.get("encoder_batch_ids", None),
                kwargs.get("encoder_tile_ids_per_batch", None),
                kwargs.get("encoder_num_blocks", None),
                kwargs.get("kv_batch_ids", None),
                kwargs.get("kv_tile_ids_per_batch", None),
                kwargs.get("kv_num_blocks", None),
                kwargs.get("decoder_batch_ids", None),
                kwargs.get("decoder_tile_ids_per_batch", None),
                kwargs.get("decoder_num_blocks_cpu", None),
                kwargs.get("max_enc_len_this_time", None),
                kwargs.get("max_dec_len_this_time", None),
                kwargs.get("max_len_kv", None),
                rotary_embs,
                None,  # attn_mask
                self.qkv_biases[i] if len(self.qkv_biases) > 0 else None,
                self.qkv_out_scales[i] if not self.skip_quant("qkv_weight_scale", i) else None,
                k_quant_scales[i] if k_quant_scales is not None else None,
                v_quant_scales[i] if v_quant_scales is not None else None,
                k_dequant_scales[i] if k_dequant_scales is not None else None,
                v_dequant_scales[i] if v_dequant_scales is not None else None,
                cache_k_zps[i] if cache_k_zps is not None else None,
                cache_v_zps[i] if cache_v_zps is not None else None,
                self.linear_shifts[i] if len(self.linear_shifts) > 0 else None,
                self.linear_smooths[i] if len(self.linear_smooths) > 0 else None,
                self._fuse_kernel_compute_dtype,
                cache_quant_type_str,
                self.use_neox_rotary_style,
                kwargs.get("max_input_length", -1),
                self.softmax_scale,
                self.quant_max_bound,
                self.quant_min_bound,
                self.act_scales["out_linear_in_scale"][i],
                self.config.speculate_config.speculate_max_draft_token_num,
                True,  # causal
                self.config.speculate_config.speculate_method is not None,  # speculate_decoder
            )[0]
        else:
            fmha_out = paddle.incubate.nn.functional.block_multihead_attention(
                qkv_out,
                caches[2 * i],
                caches[2 * i + 1],
                kwargs.get("seq_lens_encoder", None),
                kwargs.get("seq_lens_decoder", None),
                kwargs.get("seq_lens_this_time", None),
                kwargs.get("padding_offsets", None),
                kwargs.get("cum_offsets", None),
                kwargs.get("cu_seqlens_q", None),
                kwargs.get("cu_seqlens_k", None),
                kwargs.get("block_tables", None),
                pre_caches[2 * i] if pre_caches is not None else None,  # pre_key_cache
                pre_caches[2 * i + 1] if pre_caches is not None else None,  # pre_value_cache
                k_quant_scales[i] if k_quant_scales is not None else None,
                v_quant_scales[i] if v_quant_scales is not None else None,
                k_dequant_scales[i] if k_dequant_scales is not None else None,
                v_dequant_scales[i] if v_dequant_scales is not None else None,
                self.qkv_out_scales[i] if not self.skip_quant("qkv_weight_scale", i) else None,
                self.qkv_biases[i] if len(self.qkv_biases) > 0 else None,
                self.linear_shifts[i] if len(self.linear_shifts) > 0 else None,
                self.linear_smooths[i] if len(self.linear_smooths) > 0 else None,
                kwargs.get("max_enc_len_this_time", None),
                kwargs.get("max_dec_len_this_time", None),
                rotary_embs,
                attn_mask,
                kwargs.get("tgt_mask", None),
                kwargs.get("max_input_length", -1),
                kwargs.get("block_size", 64),
                self.use_neox_rotary_style,
                self.config.cachekv_int8_type == "dynamic",
                quant_round_type=self.quant_round_type,
                quant_max_bound=self.quant_max_bound,
                quant_min_bound=self.quant_min_bound,
                out_scale=self.act_scales["out_linear_in_scale"][i],
                compute_dtype=self._fuse_kernel_compute_dtype,
                rope_theta=self.config.rope_theta,
            )[0]

        return fmha_out


class FusedBlockMultiTransformerFP8(FusedBlockMultiTransformer):
    def __init__(self, config: FusedMultiTransformerConfig):
        super().__init__(config)
        self.act_scales = None
        self.weight_scales = None

        self.quant_round_type = config.quant_round_type
        self.quant_max_bound = config.quant_max_bound
        self.quant_min_bound = config.quant_min_bound

        self.ffn1_0_biases = []
        self.ffn1_1_biases = []

        self.qkv_out_scales = []
        self.linear_out_scales = []
        self.ffn1_0_out_scales = []
        self.ffn1_1_out_scales = []
        self.ffn2_out_scales = []

        self.init_weight_shape(config)

        for i in range(self.num_layers):
            self.qkv_out_scales.append(-1.0)
            self.linear_out_scales.append(-1.0)
            self.ffn1_0_out_scales.append(-1.0)
            self.ffn1_1_out_scales.append(-1.0)
            self.ffn2_out_scales.append(-1.0)

            ffn1_0_bias_attr = self.get_attr(config.ffn1_0_bias_attrs, i)
            ffn1_1_bias_attr = self.get_attr(config.ffn1_1_bias_attrs, i)

            ffn1_0_bias = None
            if ffn1_0_bias_attr:
                ffn1_0_bias = self.create_parameter(
                    shape=[self.intermediate_size],
                    attr=ffn1_0_bias_attr,
                    dtype=self._dtype,
                    is_bias=True,
                )

            ffn1_1_bias = None
            if ffn1_1_bias_attr:
                ffn1_1_bias = self.create_parameter(
                    shape=[self.intermediate_size],
                    attr=ffn1_1_bias_attr,
                    dtype=self._dtype,
                    is_bias=True,
                )

            # tensor model parallel
            if config.nranks > 1:
                # column parallel
                _set_var_distributed(ffn1_0_bias)
                _set_var_distributed(ffn1_1_bias)

            self.ffn1_0_biases.append(ffn1_0_bias)
            self.ffn1_1_biases.append(ffn1_1_bias)

            self._add_parameter(ffn1_0_bias)
            self._add_parameter(ffn1_1_bias)

    def init_weight(self):
        self.qkv_weights = []
        self.linear_weights = []
        self.ffn1_0_weights = []
        self.ffn1_1_weights = []
        self.ffn2_weights = []

        for i in range(self.num_layers):
            qkv_weight_attr = self.get_attr(self.config.qkv_weight_attrs, i)
            linear_weight_attr = self.get_attr(self.config.linear_weight_attrs, i)
            ffn1_0_weight_attr = self.get_attr(self.config.ffn1_0_weight_attrs, i)
            ffn1_1_weight_attr = self.get_attr(self.config.ffn1_1_weight_attrs, i)
            ffn2_weight_attr = self.get_attr(self.config.ffn2_weight_attrs, i)

            qkv_weight = self.create_parameter(
                shape=self.qkv_weight_shape,
                attr=qkv_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )
            linear_weight = self.create_parameter(
                shape=self.linear_weight_shape,
                attr=linear_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )
            ffn1_0_weight = self.create_parameter(
                shape=self.ffn1_0_weight_shape,
                attr=ffn1_0_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )
            ffn1_1_weight = self.create_parameter(
                shape=self.ffn1_1_weight_shape,
                attr=ffn1_1_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )
            ffn2_weight = self.create_parameter(
                shape=self.ffn2_weight_shape,
                attr=ffn2_weight_attr,
                dtype=self.create_params_type,
                is_bias=False,
            )

            # tensor model parallel
            if self.config.nranks > 1:
                # column parallel
                _set_var_distributed(qkv_weight)
                _set_var_distributed(ffn1_0_weight)
                _set_var_distributed(ffn1_1_weight)
                # row parallel
                _set_var_distributed(linear_weight)
                _set_var_distributed(ffn2_weight)

            self.qkv_weights.append(qkv_weight)
            self.linear_weights.append(linear_weight)

            self.ffn1_0_weights.append(ffn1_0_weight)
            self.ffn1_1_weights.append(ffn1_1_weight)

            self.ffn2_weights.append(ffn2_weight)

            self._add_parameter(qkv_weight)
            self._add_parameter(linear_weight)

            self._add_parameter(ffn1_0_weight)
            self._add_parameter(ffn1_1_weight)
            self._add_parameter(ffn2_weight)

    def init_weight_shape(self, config):
        """
        For fake parameter
        """
        self.qkv_weight_shape = (
            [(self.num_heads + 2 * self.kv_num_heads) * self.head_dim, self.embed_dim]
            if config.trans_qkvw
            else [self.embed_dim, (self.num_heads + 2 * self.kv_num_heads) * self.head_dim]
        )
        self.linear_weight_shape = [self.num_heads * self.head_dim, self.embed_dim]
        self.ffn1_0_weight_shape = [self.intermediate_size, self.embed_dim]
        self.ffn1_1_weight_shape = [self.intermediate_size, self.embed_dim]
        self.ffn2_weight_shape = [self.embed_dim, self.intermediate_size]

    def get_weight_create_dype(self, layer_name=None, layer_idx=None):
        """
        For fake parameter
        """
        if layer_name is not None and layer_idx is not None:
            if self.weight_scales[layer_name][layer_idx] == -1:
                return self._dtype
        return "float8_e4m3fn"

    def compute_layernorm_before_qkv(self, src, i):
        """
        For fake parameter
        """
        if i == 0:
            ln_out = self.norm_func(
                src,
                self.ln_scales[i],
                self.ln_biases[i],
                self._epsilon,
                begin_norm_axis=1,
                quant_scale=self.act_scales["qkv_in_scale"][i],  # quant_in_scale
                quant_round_type=1,
                quant_max_bound=self.config.quant_max_bound,
                quant_min_bound=self.config.quant_min_bound,
            )[0]
        else:
            ln_out = src

        return ln_out

    def compute_qkv_linear(self, ln_out, i):
        if self.config.mla_config.use_mla():
            raise NotImplementedError("Not support MLA yet.")
        else:
            if paddle.is_compiled_with_rocm() or float(paddle.version.cuda()) < 11.6:
                qkv_out = paddle.matmul(ln_out, self.qkv_weights[i], False, True)
                if self.qkv_biases[i] is not None:
                    qkv_out = paddle.add(qkv_out, self.qkv_biases[i])
                return qkv_out
            else:
                qkv_out = fp8_gemm_fused(
                    ln_out,
                    self.qkv_weights[i],
                    transpose_x=False,
                    transpose_y=True,
                    bias=self.qkv_biases[i],
                    scale=self.weight_scales["qkv_weight_scale"][i] / (self.act_scales["qkv_in_scale"][i] * 448 * 448),
                    output_dtype=self._dtype,
                    act="identity",
                )

            return qkv_out

    def compute_out_linear(self, fmha_out, i):
        """
        For fake parameter
        """
        return fp8_gemm_fused(
            fmha_out,
            self.linear_weights[i],
            bias=None,
            transpose_x=False,
            transpose_y=True,
            scale=self.weight_scales["out_linear_weight_scale"][i]
            / (self.act_scales["out_linear_in_scale"][i] * 448 * 448),
            output_dtype=self._dtype,
            act="identity",
        )

    def compute_attn(
        self,
        time_step,
        qkv_out,
        padding_offset,
        seq_lens,
        input_ids,
        rotary_embs,
        rotary_emb_dims,
        caches,
        pre_caches,
        pre_caches_length,
        attn_mask,
        i,
        **kwargs,
    ):
        """
            Compute the attention for a single time step.
        Args:
            time_step (int): The current time step.
            qkv_out (Tensor): The output of the linear layer.
            padding_offset (Tensor): The padding offset tensor.
            seq_lens (Tensor): The sequence length tensor.
            input_ids (Tensor): The input ids tensor.
            rotary_embs (Tensor, optional): The rotary embeddings tensor. Defaults to None.
            rotary_emb_dims (int, optional): The rotary embedding dimension. Defaults to None.
            caches (List[Tensor], optional): The cache list. Defaults to None.
            pre_caches (List[Tensor], optional): The pre-cache list. Defaults to None.
            pre_caches_length (int, optional): The pre-cache length. Defaults to None.
            attn_mask (Tensor, optional): The attention mask tensor. Defaults to None.
            i (int, optional): The index of the block. Defaults to None.
            **kwargs (dict, optional): Additional keyword arguments. Defaults to {}.
        Returns:
            Tensor: The output linear layer output.
        Raises:
            None.
        """
        k_quant_scales = kwargs.get("k_quant_scales", None)
        v_quant_scales = kwargs.get("v_quant_scales", None)
        k_dequant_scales = kwargs.get("k_dequant_scales", None)
        v_dequant_scales = kwargs.get("v_dequant_scales", None)
        cache_k_zps = kwargs.get("cache_k_zp", None)
        cache_v_zps = kwargs.get("cache_v_zp", None)

        cache_quant_type_str = "none"
        if self.config.cachekv_int8_type == "static":
            k_quant_scales = self.cache_k_scales
            v_quant_scales = self.cache_v_scales
            k_dequant_scales = self.cache_k_out_scales
            v_dequant_scales = self.cache_v_out_scales
            cache_quant_type_str = "cache_int8"

        if self.config.append_attn:
            from paddlenlp_ops import append_attention

            fmha_out = append_attention(
                qkv_out,
                caches[2 * i],
                caches[2 * i + 1],
                kwargs.get("seq_lens_encoder", None),
                kwargs.get("seq_lens_decoder", None),
                kwargs.get("seq_lens_this_time", None),
                kwargs.get("padding_offsets", None),
                kwargs.get("cum_offsets", None),
                kwargs.get("block_tables", None),
                kwargs.get("encoder_batch_ids", None),
                kwargs.get("encoder_tile_ids_per_batch", None),
                kwargs.get("encoder_num_blocks", None),
                kwargs.get("kv_batch_ids", None),
                kwargs.get("kv_tile_ids_per_batch", None),
                kwargs.get("kv_num_blocks", None),
                kwargs.get("decoder_batch_ids", None),
                kwargs.get("decoder_tile_ids_per_batch", None),
                kwargs.get("decoder_num_blocks_cpu", None),
                kwargs.get("max_enc_len_this_time", None),
                kwargs.get("max_dec_len_this_time", None),
                kwargs.get("max_len_kv", None),
                rotary_embs,
                None,  # attn_mask
                None,  # qkv_bias
                None,  # qkv_out_scales
                k_quant_scales[i] if k_quant_scales is not None else None,
                v_quant_scales[i] if v_quant_scales is not None else None,
                k_dequant_scales[i] if k_dequant_scales is not None else None,
                v_dequant_scales[i] if v_dequant_scales is not None else None,
                cache_k_zps[i] if cache_k_zps is not None else None,
                cache_v_zps[i] if cache_v_zps is not None else None,
                None,  # linear_shifts
                None,  # linear_smooths
                self._fuse_kernel_compute_dtype,
                cache_quant_type_str,
                self.use_neox_rotary_style,
                kwargs.get("max_input_length", -1),
                self.softmax_scale,
                self.quant_max_bound,
                self.quant_min_bound,
                self.act_scales["out_linear_in_scale"][i],
                self.config.speculate_config.speculate_max_draft_token_num,
                True,  # causal
                False,  # speculate_decoder
            )[0]
        else:
            fmha_out = paddle.incubate.nn.functional.block_multihead_attention(
                qkv_out,
                caches[2 * i],
                caches[2 * i + 1],
                kwargs.get("seq_lens_encoder", None),
                kwargs.get("seq_lens_decoder", None),
                kwargs.get("seq_lens_this_time", None),
                kwargs.get("padding_offsets", None),
                kwargs.get("cum_offsets", None),
                kwargs.get("cu_seqlens_q", None),
                kwargs.get("cu_seqlens_k", None),
                kwargs.get("block_tables", None),
                pre_caches[2 * i] if pre_caches is not None else None,  # pre_key_cache
                pre_caches[2 * i + 1] if pre_caches is not None else None,  # pre_value_cache
                k_quant_scales[i] if k_quant_scales is not None else None,
                v_quant_scales[i] if v_quant_scales is not None else None,
                k_dequant_scales[i] if k_dequant_scales is not None else None,
                v_dequant_scales[i] if v_dequant_scales is not None else None,
                None,  # qkv_out_scales
                None,  # qkv_bias
                None,  # out_shifts
                None,  # out_smooths
                kwargs.get("max_enc_len_this_time", None),
                kwargs.get("max_dec_len_this_time", None),
                rotary_embs,
                attn_mask,
                kwargs.get("tgt_mask", None),  # tgt_mask
                kwargs.get("max_input_length", -1),
                kwargs.get("block_size", 64),
                self.use_neox_rotary_style,
                self.config.use_dynamic_cachekv_quant,
                quant_round_type=self.config.quant_round_type,
                quant_max_bound=self.config.quant_max_bound,
                quant_min_bound=self.config.quant_min_bound,
                out_scale=self.act_scales.scale["out_linear_in_scale"][i],
                rope_theta=self.config.rope_theta,
            )[0]

        return fmha_out

    def compute_ffn_layernorm(self, out_linear_out, residual_input, i):
        """
        For fake parameter
        """
        norm_out = self.norm_func(
            out_linear_out,
            norm_weight=self.ffn_ln_scales[i],
            norm_bias=self.ffn_ln_biases[i],
            epsilon=self._epsilon,
            begin_norm_axis=1,
            bias=self.linear_biases[i],
            residual=residual_input,
            quant_scale=self.act_scales["ffn1_in_scale"][i],  # quant_in_scale
            quant_round_type=1,
            quant_max_bound=self.config.quant_max_bound,
            quant_min_bound=self.config.quant_min_bound,
        )
        tmp_out, residual_input = norm_out[0], norm_out[1]

        return tmp_out, residual_input

    def compute_ffn1(self, tmp_out, i):
        """
        For fake parameter
        """
        if use_cutlass_fp8_gemm():
            res = fp8_dual_gemm_fused(
                tmp_out,
                self.ffn1_0_weights[i],
                self.ffn1_1_weights[i],
                transpose_x=False,
                transpose_y=True,
                bias0=self.ffn1_0_biases[i],
                bias1=self.ffn1_1_biases[i],
                scale0=self.weight_scales["ffn1_0_weight_scale"][i]
                / (self.act_scales["ffn1_in_scale"][i] * 448 * 448),
                scale1=self.weight_scales["ffn1_1_weight_scale"][i]
                / (self.act_scales["ffn1_in_scale"][i] * 448 * 448),
                scale_out=self.act_scales["ffn2_in_scale"][i] * 448,
                act="swiglu",
            )
            return res
        else:
            tem_0 = fp8_gemm_fused(
                tmp_out,
                self.ffn1_0_weights[i],
                transpose_x=False,
                transpose_y=True,
                scale=self.weight_scales["ffn1_0_weight_scale"][i] / (self.act_scales["ffn1_in_scale"][i] * 448 * 448),
                bias=self.ffn1_0_biases[i],
                output_dtype=self._dtype,
                act="identity",
            )

            tem_1 = fp8_gemm_fused(
                tmp_out,
                self.ffn1_1_weights[i],
                transpose_x=False,
                transpose_y=True,
                scale=self.weight_scales["ffn1_1_weight_scale"][i] / (self.act_scales["ffn1_in_scale"][i] * 448 * 448),
                bias=self.ffn1_1_biases[i],
                output_dtype=self._dtype,
                act="identity",
            )

            from paddle.incubate.nn.functional import swiglu

            tem = swiglu(paddle.cast(tem_0, "float32"), paddle.cast(tem_1, "float32"))
            res = paddle.cast(tem * self.act_scales["ffn2_in_scale"][i] * 448, "float8_e4m3fn")
        return res

    def compute_activation(self, ffn1_out, i):
        return ffn1_out

    def compute_ffn2(self, ffn1_out, i):
        """
        For fake parameter
        """
        return fp8_gemm_fused(
            ffn1_out,
            self.ffn2_weights[i],
            bias=None,
            transpose_x=False,
            transpose_y=True,
            scale=self.weight_scales["ffn2_weight_scale"][i] / (self.act_scales["ffn2_in_scale"][i] * 448 * 448),
            output_dtype=self._dtype,
            act="identity",
        )

    def compute_bias_residual_layernorm(self, ffn2_out, residual_input, i, num_layers):
        """
        For fake parameter
        """
        if i != num_layers - 1:
            norm_out = self.norm_func(
                ffn2_out,
                norm_weight=self.ln_scales[i + 1],
                norm_bias=self.ln_biases[i + 1],
                epsilon=self._epsilon,
                begin_norm_axis=1,
                bias=self.ffn2_biases[i],
                residual=residual_input,
                quant_scale=self.act_scales["qkv_in_scale"][i + 1],  # quant_in_scale
                quant_round_type=1,
                quant_max_bound=self.config.quant_max_bound,
                quant_min_bound=self.config.quant_min_bound,
            )
            tmp_out, residual_input = norm_out[0], norm_out[1]
        else:
            tmp_out = fused_layer_norm(
                ffn2_out,
                norm_weight=None,
                norm_bias=None,
                epsilon=self._epsilon,
                begin_norm_axis=1,
                bias=self.ffn2_biases[i],
                residual=residual_input,
            )[0]
        return tmp_out, residual_input


class FusedBlockMultiTransformerFP8DynamicQuant(FusedBlockMultiTransformer):
    def __init__(self, config: FusedMultiTransformerConfig):
        super().__init__(config)
        self.quant_type = config.quant_type
        self.fp8_type = "float8_e4m3fn"
        self.weight_scale_dtype = "float32"
        self.weight_block_size = self.config.weight_block_size

        self.qkv_weights_scale = []
        self.linear_weights_scale = []
        self.ffn1_weights_scale = []
        self.ffn2_weights_scale = []

        self.q_proj_weights_scale = []
        self.q_a_proj_weights_scale = []
        self.q_b_proj_weights_scale = []
        self.kv_a_proj_with_mqa_weights_scale = []
        self.kv_b_proj_weights_scale = []
        self.q_nope_k_b_proj_weights_scale = []
        self.q_rope_proj_weights_scale = []
        self.v_b_o_proj_weights_scale = []

        self.shared_expert_ffn1_weights_scale = []
        self.shared_expert_ffn2_weights_scale = []

        for i in range(self.num_layers):

            linear_weight_scale_attr = self.get_attr(self.config.linear_weight_scale_attrs, i)
            ffn1_weight_scale_attr = self.get_attr(self.config.ffn1_weight_scale_attrs, i)
            ffn2_weight_scale_attr = self.get_attr(self.config.ffn2_weight_scale_attrs, i)

            if self.config.moe_config.use_shared_expert(i):
                shared_expert_ffn1_weight_scale_attr = self.get_attr(
                    self.config.moe_config.shared_expert_ffn1_weight_scale_attrs, i
                )
                shared_expert_ffn2_weight_scale_attr = self.get_attr(
                    self.config.moe_config.shared_expert_ffn2_weight_scale_attrs, i
                )

            q_a_proj_weight_scale = None
            q_b_proj_weight_scale = None
            kv_a_proj_with_mqa_weight_scale = None
            kv_b_proj_weight_scale = None
            if self.config.mla_config.use_mla():
                q_proj_weight_scale = None
                q_proj_weight_scale_attr = self.get_attr(self.config.mla_config.q_proj_weight_scale_attrs, i)
                if q_proj_weight_scale_attr:
                    q_proj_weight_scale = self.create_parameter(
                        shape=self.get_scale_shape(self.q_proj_weight_shape),
                        attr=q_proj_weight_scale_attr,
                        dtype="float32",
                        is_bias=False,
                    )

                q_a_proj_weight_scale_attr = self.get_attr(self.config.mla_config.q_a_proj_weight_scale_attrs, i)
                q_b_proj_weight_scale_attr = self.get_attr(self.config.mla_config.q_b_proj_weight_scale_attrs, i)
                if q_a_proj_weight_scale_attr:
                    q_a_proj_weight_scale = self.create_parameter(
                        shape=self.get_scale_shape(self.q_a_proj_weight_shape),
                        attr=q_a_proj_weight_scale_attr,
                        dtype="float32",
                        is_bias=False,
                    )
                if q_b_proj_weight_scale_attr:
                    q_b_proj_weight_scale = self.create_parameter(
                        shape=self.get_scale_shape(self.q_b_proj_weight_shape),
                        attr=q_b_proj_weight_scale_attr,
                        dtype="float32",
                        is_bias=False,
                    )

                kv_a_proj_with_mqa_weight_scale_attr = self.get_attr(
                    self.config.mla_config.kv_a_proj_with_mqa_weight_scale_attrs, i
                )
                kv_b_proj_weight_scale_attr = self.get_attr(self.config.mla_config.kv_b_proj_weight_scale_attrs, i)
                if kv_a_proj_with_mqa_weight_scale_attr:
                    kv_a_proj_with_mqa_weight_scale = self.create_parameter(
                        shape=self.get_scale_shape(self.kv_a_proj_with_mqa_weight_shape),
                        attr=kv_a_proj_with_mqa_weight_scale_attr,
                        dtype="float32",
                        is_bias=False,
                    )
                if kv_b_proj_weight_scale_attr:
                    kv_b_proj_weight_scale = self.create_parameter(
                        shape=self.get_scale_shape(self.kv_b_proj_weight_shape),
                        attr=kv_b_proj_weight_scale_attr,
                        dtype="float32",
                        is_bias=False,
                    )

            qkv_weight_scale = None
            qkv_weight_scale_attr = self.get_attr(self.config.qkv_weight_scale_attrs, i)
            if qkv_weight_scale_attr:
                qkv_weight_scale = self.create_parameter(
                    shape=self.get_scale_shape(self.qkv_weight_shape),
                    attr=qkv_weight_scale_attr,
                    dtype="float32",
                    is_bias=False,
                )

            linear_weight_scale = None
            linear_weight_scale_attr = self.get_attr(config.linear_weight_scale_attrs, i)
            if linear_weight_scale_attr:
                linear_weight_scale = self.create_parameter(
                    shape=self.get_scale_shape(self.linear_weight_shape),
                    attr=linear_weight_scale_attr,
                    dtype="float32",
                    is_bias=False,
                )

            q_nope_k_b_proj_weight_scale = None
            q_rope_proj_weight_scale = None
            v_b_o_proj_weight_scale = None
            if self.config.mla_config.use_absorb():
                q_nope_k_b_proj_weight_scale_attr = self.get_attr(
                    self.config.mla_config.q_nope_k_b_proj_weight_scale_attrs, i
                )
                q_rope_proj_weight_scale_attr = self.get_attr(self.config.mla_config.q_rope_proj_weight_scale_attrs, i)
                v_b_o_proj_weight_scale_attr = self.get_attr(self.config.mla_config.v_b_o_proj_weight_scale_attrs, i)
                if q_nope_k_b_proj_weight_scale_attr:
                    q_nope_k_b_proj_weight_scale = self.create_parameter(
                        shape=self.get_scale_shape(self.q_nope_k_b_proj_weight_shape),
                        attr=q_nope_k_b_proj_weight_scale_attr,
                        dtype="float32",
                        is_bias=False,
                    )
                if q_rope_proj_weight_scale_attr:
                    q_rope_proj_weight_scale = self.create_parameter(
                        shape=self.get_scale_shape(self.q_rope_proj_weight_shape),
                        attr=q_rope_proj_weight_scale_attr,
                        dtype="float32",
                        is_bias=False,
                    )
                if v_b_o_proj_weight_scale_attr:
                    v_b_o_proj_weight_scale = self.create_parameter(
                        shape=self.get_scale_shape(self.v_b_o_proj_weight_shape),
                        attr=v_b_o_proj_weight_scale_attr,
                        dtype="float32",
                        is_bias=False,
                    )

            ffn1_weight_scale = None
            ffn2_weight_scale = None
            ffn1_weight_scale_attr = self.get_attr(config.ffn1_weight_scale_attrs, i)
            ffn2_weight_scale_attr = self.get_attr(config.ffn2_weight_scale_attrs, i)
            if self.config.moe_config.use_moe(i):
                if self.moe_quant_type in ["weight_only_int4", "weight_only_int8"]:
                    ffn1_weight_scale = self.create_parameter(
                        shape=[self.config.moe_config.num_experts, self.config.moe_config.moe_intermediate_size * 2]
                        if config.activation.endswith("glu")
                        else [self.config.moe_config.num_experts, self.config.moe_config.moe_intermediate_size],
                        attr=ffn1_weight_scale_attr,
                        dtype=self._dtype,
                        is_bias=False,
                    )
                else:
                    ffn1_weight_scale = self.create_parameter(
                        shape=self.get_scale_shape(self.moe_ffn1_weight_shape, ffn1=True),
                        attr=ffn1_weight_scale_attr,
                        dtype="float32",
                        is_bias=False,
                    )
            else:
                ffn1_weight_scale = self.create_parameter(
                    shape=self.get_scale_shape(self.ffn1_weight_shape, ffn1=True),
                    attr=ffn1_weight_scale_attr,
                    dtype="float32",
                    is_bias=False,
                )

            if self.config.moe_config.use_moe(i):
                if self.moe_quant_type in ["weight_only_int4", "weight_only_int8"]:
                    ffn2_weight_scale = self.create_parameter(
                        shape=[self.config.moe_config.num_experts, self.embed_dim],
                        attr=ffn2_weight_scale_attr,
                        dtype=self._dtype,
                        is_bias=False,
                    )
                else:
                    ffn2_weight_scale = self.create_parameter(
                        shape=self.get_scale_shape(self.moe_ffn2_weight_shape, ffn1=True),
                        attr=ffn2_weight_scale_attr,
                        dtype="float32",
                        is_bias=False,
                    )
            else:
                ffn2_weight_scale = self.create_parameter(
                    shape=self.get_scale_shape(self.ffn2_weight_shape),
                    attr=ffn2_weight_scale_attr,
                    dtype="float32",
                    is_bias=False,
                )

            shared_expert_ffn1_weight_scale = None
            shared_expert_ffn2_weight_scale = None
            shared_expert_ffn1_weight_scale_attr = self.get_attr(
                config.moe_config.shared_expert_ffn1_weight_scale_attrs, i
            )
            shared_expert_ffn2_weight_scale_attr = self.get_attr(
                config.moe_config.shared_expert_ffn2_weight_scale_attrs, i
            )
            if self.config.moe_config.use_shared_expert(i):
                shared_expert_ffn1_weight_scale = self.create_parameter(
                    shape=self.get_scale_shape(self.shared_expert_ffn1_weight_shape, ffn1=True),
                    attr=shared_expert_ffn1_weight_scale_attr,
                    dtype="float32",
                    is_bias=False,
                )

                shared_expert_ffn2_weight_scale = self.create_parameter(
                    shape=self.get_scale_shape(self.shared_expert_ffn2_weight_shape),
                    attr=shared_expert_ffn2_weight_scale_attr,
                    dtype="float32",
                    is_bias=False,
                )

            self.q_proj_weights_scale.append(q_proj_weight_scale)
            self.q_a_proj_weights_scale.append(q_a_proj_weight_scale)
            self.q_b_proj_weights_scale.append(q_b_proj_weight_scale)
            self.kv_a_proj_with_mqa_weights_scale.append(kv_a_proj_with_mqa_weight_scale)
            self.kv_b_proj_weights_scale.append(kv_b_proj_weight_scale)
            self.qkv_weights_scale.append(qkv_weight_scale)

            self.q_nope_k_b_proj_weights_scale.append(q_nope_k_b_proj_weight_scale)
            self.q_rope_proj_weights_scale.append(q_rope_proj_weight_scale)
            self.v_b_o_proj_weights_scale.append(v_b_o_proj_weight_scale)

            self.linear_weights_scale.append(linear_weight_scale)
            self.ffn1_weights_scale.append(ffn1_weight_scale)
            self.ffn2_weights_scale.append(ffn2_weight_scale)

            self.shared_expert_ffn1_weights_scale.append(shared_expert_ffn1_weight_scale)
            self.shared_expert_ffn2_weights_scale.append(shared_expert_ffn2_weight_scale)

            self._add_parameter(q_proj_weight_scale)
            self._add_parameter(q_a_proj_weight_scale)
            self._add_parameter(q_b_proj_weight_scale)
            self._add_parameter(kv_a_proj_with_mqa_weight_scale)
            self._add_parameter(kv_b_proj_weight_scale)
            self._add_parameter(qkv_weight_scale)

            self._add_parameter(q_nope_k_b_proj_weight_scale)
            self._add_parameter(q_rope_proj_weight_scale)
            self._add_parameter(v_b_o_proj_weight_scale)

            self._add_parameter(linear_weight_scale)
            self._add_parameter(ffn1_weight_scale)
            self._add_parameter(ffn2_weight_scale)

            self._add_parameter(shared_expert_ffn1_weight_scale)
            self._add_parameter(shared_expert_ffn2_weight_scale)

    def get_scale_shape(self, weight_shape: list, ffn1=False):
        n, k = weight_shape[-2:]
        block_k, block_n = self.weight_block_size
        scale_shape = [i for i in weight_shape]
        scale_shape[-2] = (n + block_n - 1) // block_n if block_n != 0 else 1
        if ffn1 and (block_k + block_n) == 0:
            scale_shape[-2] *= 2
        scale_shape[-1] = (k + block_k - 1) // block_k if block_k != 0 else 1
        return scale_shape

    def init_weight_shape(self, config):
        super().init_weight_shape(config)

        if self.config.mla_config.use_mla():
            if self.config.mla_config.q_lora_rank is None:
                self.q_proj_weight_shape = [
                    self.num_heads * (self.config.mla_config.qk_head_dim),
                    self.config.embed_dim,
                ]
            else:
                self.q_a_proj_weight_shape = [self.config.mla_config.q_lora_rank, self.config.embed_dim]
                self.q_b_proj_weight_shape = [
                    self.num_heads * (self.config.mla_config.qk_head_dim),
                    self.config.mla_config.q_lora_rank,
                ]

            self.kv_a_proj_with_mqa_weight_shape = [
                self.config.mla_config.kv_lora_rank + self.config.mla_config.qk_rope_head_dim,
                self.config.embed_dim,
            ]
            self.kv_b_proj_weight_shape = [
                self.num_heads * (self.config.mla_config.qk_nope_head_dim + self.config.mla_config.v_head_dim),
                self.config.mla_config.kv_lora_rank,
            ]

            self.q_nope_k_b_proj_weight_shape = [
                self.num_heads * self.config.mla_config.kv_lora_rank,
                self.embed_dim if self.config.mla_config.q_lora_rank is None else self.config.mla_config.q_lora_rank,
            ]
            self.q_rope_proj_weight_shape = [
                self.num_heads * self.config.mla_config.qk_rope_head_dim,
                self.embed_dim if self.config.mla_config.q_lora_rank is None else self.config.mla_config.q_lora_rank,
            ]
            self.v_b_o_proj_weight_shape = [
                self.embed_dim,
                self.num_heads * self.config.mla_config.kv_lora_rank,
            ]
        else:
            self.qkv_weight_shape = (
                [(self.num_heads + 2 * self.kv_num_heads) * self.head_dim, self.embed_dim]
                if config.trans_qkvw
                else [self.embed_dim, (self.num_heads + 2 * self.kv_num_heads) * self.head_dim]
            )

        self.linear_weight_shape = [self.embed_dim, self.num_heads * self.head_dim]
        self.ffn1_weight_shape = (
            [self.intermediate_size * 2, self.embed_dim]
            if self.activation.endswith("glu")
            else [self.intermediate_size, self.embed_dim]
        )

        self.ffn2_weight_shape = [self.embed_dim, self.intermediate_size]

        if self.config.moe_config.has_moe():
            self.moe_ffn1_weight_shape = (
                [
                    self.config.moe_config.num_experts,
                    self.config.moe_config.moe_intermediate_size * 2,
                    self.embed_dim,
                ]
                if self.activation.endswith("glu")
                else [
                    self.config.moe_config.num_experts,
                    self.config.moe_config.moe_intermediate_size,
                    self.embed_dim,
                ]
            )
            self.moe_ffn2_weight_shape = [
                self.config.moe_config.num_experts,
                self.embed_dim,
                self.config.moe_config.moe_intermediate_size,
            ]
            if self.moe_quant_type in ["weight_only_int4", "weight_only_int8"]:
                self.moe_ffn1_weight_shape = (
                    [
                        self.config.moe_config.num_experts,
                        self.embed_dim,
                        self.config.moe_config.moe_intermediate_size * 2,
                    ]
                    if self.activation.endswith("glu")
                    else [
                        self.config.moe_config.num_experts,
                        self.embed_dim,
                        self.config.moe_config.moe_intermediate_size,
                    ]
                )
                self.moe_ffn2_weight_shape = [
                    self.config.moe_config.num_experts,
                    self.config.moe_config.moe_intermediate_size,
                    self.embed_dim,
                ]
                if config.moe_quant_type == "weight_only_int4":
                    if config.moe_config.has_shared_expert():
                        self.moe_ffn1_weight_shape[2] //= 2
                        self.moe_ffn2_weight_shape[1] //= 2
                    else:
                        self.moe_ffn1_weight_shape[2] //= 2
                        self.moe_ffn2_weight_shape[2] //= 2

        if self.config.moe_config.has_shared_expert():
            self.shared_expert_ffn1_weight_shape = [
                self.config.moe_config.shared_expert_intermediate_size * 2,
                self.embed_dim,
            ]
            self.shared_expert_ffn2_weight_shape = [
                self.embed_dim,
                self.config.moe_config.shared_expert_intermediate_size,
            ]
            if self.config.moe_config.shared_expert_with_gate:
                self.shared_expert_gate_weight_shape = [
                    self.embed_dim,
                    1,
                ]

    def init_weight(self):
        self.qkv_weights = []
        self.linear_weights = []
        self.gate_weights = []
        self.ffn1_weights = []
        self.ffn2_weights = []

        self.q_proj_weights = []
        self.q_a_proj_weights = []
        self.q_a_layernorm_weights = []
        self.q_b_proj_weights = []
        self.kv_a_proj_with_mqa_weights = []
        self.kv_a_layernorm_weights = []
        self.kv_b_proj_weights = []
        self.q_nope_k_b_proj_weights = []
        self.q_rope_proj_weights = []
        self.v_b_o_proj_weights = []

        for i in range(self.num_layers):
            q_proj_weight = None
            q_a_proj_weight = None
            q_a_layernorm_weight = None
            q_b_proj_weight = None
            kv_a_proj_with_mqa_weight = None
            kv_a_layernorm_weight = None
            kv_b_proj_weight = None
            q_nope_k_b_proj_weight = None
            q_rope_proj_weight = None
            v_b_o_proj_weight = None
            if self.config.mla_config.use_mla():
                q_proj_weight_attr = self.get_attr(self.config.mla_config.q_proj_weight_attrs, i)
                q_a_proj_weight_attr = self.get_attr(self.config.mla_config.q_a_proj_weight_attrs, i)
                q_a_layernorm_weight_attr = self.get_attr(self.config.mla_config.q_a_layernorm_weight_attrs, i)
                q_b_proj_weight_attr = self.get_attr(self.config.mla_config.q_b_proj_weight_attrs, i)
                if q_proj_weight_attr:
                    q_proj_weight = self.create_parameter(
                        shape=self.q_proj_weight_shape,
                        attr=q_proj_weight_attr,
                        dtype=self.fp8_type,
                        is_bias=False,
                    )
                if q_a_proj_weight_attr:
                    q_a_proj_weight = self.create_parameter(
                        shape=self.q_a_proj_weight_shape,
                        attr=q_a_proj_weight_attr,
                        dtype=self.fp8_type,
                        is_bias=False,
                    )
                if q_a_layernorm_weight_attr:
                    q_a_layernorm_weight = self.create_parameter(
                        shape=[self.config.mla_config.q_lora_rank],
                        attr=q_a_layernorm_weight_attr,
                        dtype=self._norm_weight_dtype,
                        is_bias=False,
                    )
                if q_b_proj_weight_attr:
                    q_b_proj_weight = self.create_parameter(
                        shape=self.q_b_proj_weight_shape,
                        attr=q_b_proj_weight_attr,
                        dtype=self.fp8_type,
                        is_bias=False,
                    )

                kv_a_proj_with_mqa_weight_attr = self.get_attr(
                    self.config.mla_config.kv_a_proj_with_mqa_weight_attrs, i
                )
                kv_a_layernorm_weight_attr = self.get_attr(self.config.mla_config.kv_a_layernorm_weight_attrs, i)
                kv_b_proj_weight_attr = self.get_attr(self.config.mla_config.kv_b_proj_weight_attrs, i)
                q_nope_k_b_proj_weight_attr = self.get_attr(self.config.mla_config.q_nope_k_b_proj_weight_attrs, i)
                q_rope_proj_weight_attr = self.get_attr(self.config.mla_config.q_rope_proj_weight_attrs, i)
                v_b_o_proj_weight_attr = self.get_attr(self.config.mla_config.v_b_o_proj_weight_attrs, i)

                if kv_a_proj_with_mqa_weight_attr:
                    kv_a_proj_with_mqa_weight = self.create_parameter(
                        shape=self.kv_a_proj_with_mqa_weight_shape,
                        attr=kv_a_proj_with_mqa_weight_attr,
                        dtype=self.fp8_type,
                        is_bias=False,
                    )
                if kv_a_layernorm_weight_attr:
                    kv_a_layernorm_weight = self.create_parameter(
                        shape=[self.config.mla_config.kv_lora_rank],
                        attr=kv_a_layernorm_weight_attr,
                        dtype=self._norm_weight_dtype,
                        is_bias=False,
                    )
                if kv_b_proj_weight_attr:
                    kv_b_proj_weight = self.create_parameter(
                        shape=self.kv_b_proj_weight_shape,
                        attr=kv_b_proj_weight_attr,
                        dtype=self.fp8_type,
                        is_bias=False,
                    )
                if q_nope_k_b_proj_weight_attr:
                    q_nope_k_b_proj_weight = self.create_parameter(
                        shape=self.q_nope_k_b_proj_weight_shape,
                        attr=q_nope_k_b_proj_weight_attr,
                        dtype=self.fp8_type,
                        is_bias=False,
                    )
                if q_rope_proj_weight_attr:
                    q_rope_proj_weight = self.create_parameter(
                        shape=self.q_rope_proj_weight_shape,
                        attr=q_rope_proj_weight_attr,
                        dtype=self.fp8_type,
                        is_bias=False,
                    )
                if v_b_o_proj_weight_attr:
                    v_b_o_proj_weight = self.create_parameter(
                        shape=self.v_b_o_proj_weight_shape,
                        attr=v_b_o_proj_weight_attr,
                        dtype=self.fp8_type,
                        is_bias=False,
                    )

            qkv_weight = None
            qkv_weight_attr = self.get_attr(self.config.qkv_weight_attrs, i)
            if qkv_weight_attr:
                qkv_weight = self.create_parameter(
                    shape=self.qkv_weight_shape,
                    attr=qkv_weight_attr,
                    dtype=self.fp8_type,
                    is_bias=False,
                )

            linear_weight = None
            linear_weight_attr = self.get_attr(self.config.linear_weight_attrs, i)
            if linear_weight_attr:
                linear_weight = self.create_parameter(
                    shape=self.linear_weight_shape,
                    attr=linear_weight_attr,
                    dtype=self.fp8_type,
                    is_bias=False,
                )

            gate_weight = None
            gate_weight_attr = self.get_attr(self.config.gate_weight_attrs, i)
            if self.config.moe_config.use_moe(i):
                gate_weight = self.create_parameter(
                    shape=[self.config.embed_dim, self.config.moe_config.num_experts],
                    attr=gate_weight_attr,
                    dtype="float32",
                    is_bias=False,
                    default_initializer=paddle.nn.initializer.Constant(0),
                )

            ffn1_weight = None
            ffn2_weight = None
            ffn1_weight_attr = self.get_attr(self.config.ffn1_weight_attrs, i)
            ffn2_weight_attr = self.get_attr(self.config.ffn2_weight_attrs, i)
            if self.config.moe_config.use_moe(i):
                if self.moe_quant_type in ["weight_only_int4", "weight_only_int8"]:
                    ffn1_weight = self.create_parameter(
                        shape=self.moe_ffn1_weight_shape,
                        attr=ffn1_weight_attr,
                        dtype="int8",
                        is_bias=False,
                    )
                    ffn2_weight = self.create_parameter(
                        shape=self.moe_ffn2_weight_shape,
                        attr=ffn2_weight_attr,
                        dtype="int8",
                        is_bias=False,
                    )
                else:
                    ffn1_weight = self.create_parameter(
                        shape=self.moe_ffn1_weight_shape,
                        attr=ffn1_weight_attr,
                        dtype=self.fp8_type,
                        is_bias=False,
                    )
                    ffn2_weight = self.create_parameter(
                        shape=self.moe_ffn2_weight_shape,
                        attr=ffn2_weight_attr,
                        dtype=self.fp8_type,
                        is_bias=False,
                    )
            else:
                ffn1_weight = self.create_parameter(
                    shape=self.ffn1_weight_shape,
                    attr=ffn1_weight_attr,
                    dtype=self.fp8_type,
                    is_bias=False,
                )
                ffn2_weight = self.create_parameter(
                    shape=self.ffn2_weight_shape,
                    attr=ffn2_weight_attr,
                    dtype=self.fp8_type,
                    is_bias=False,
                )

            shared_expert_ffn1_weight = None
            shared_expert_ffn2_weight = None
            shared_expert_gate_weight = None
            if self.config.moe_config.use_shared_expert(i):
                if self.config.moe_config.shared_expert_with_gate:
                    shared_expert_gate_weight_attr = self.get_attr(
                        self.config.moe_config.shared_expert_gate_weight_attrs, i
                    )
                shared_expert_ffn1_weight_attr = self.get_attr(
                    self.config.moe_config.shared_expert_ffn1_weight_attrs, i
                )
                shared_expert_ffn2_weight_attr = self.get_attr(
                    self.config.moe_config.shared_expert_ffn2_weight_attrs, i
                )

                shared_expert_ffn1_weight = self.create_parameter(
                    shape=self.shared_expert_ffn1_weight_shape,
                    attr=shared_expert_ffn1_weight_attr,
                    dtype=self.fp8_type,
                )
                shared_expert_ffn2_weight = self.create_parameter(
                    shape=self.shared_expert_ffn2_weight_shape,
                    attr=shared_expert_ffn2_weight_attr,
                    dtype=self.fp8_type,
                )
                if self.config.moe_config.shared_expert_with_gate:
                    shared_expert_gate_weight = self.create_parameter(
                        shape=self.shared_expert_gate_weight_shape,
                        attr=shared_expert_gate_weight_attr,
                        dtype=self._helper.get_default_dtype(),
                    )

            # tensor model parallel
            if self.config.nranks > 1:
                # column parallel
                _set_var_distributed(qkv_weight)
                _set_var_distributed(q_proj_weight)
                _set_var_distributed(q_b_proj_weight)
                _set_var_distributed(kv_b_proj_weight)
                _set_var_distributed(q_nope_k_b_proj_weight)
                _set_var_distributed(q_rope_proj_weight)
                _set_var_distributed(ffn1_weight)
                # row parallel
                _set_var_distributed(linear_weight)
                _set_var_distributed(v_b_o_proj_weight)
                _set_var_distributed(ffn2_weight)

                _set_var_distributed(shared_expert_ffn1_weight)
                _set_var_distributed(shared_expert_ffn2_weight)

            self.q_proj_weights.append(q_proj_weight)
            self.q_a_proj_weights.append(q_a_proj_weight)
            self.q_a_layernorm_weights.append(q_a_layernorm_weight)
            self.q_b_proj_weights.append(q_b_proj_weight)
            self.kv_a_proj_with_mqa_weights.append(kv_a_proj_with_mqa_weight)
            self.kv_a_layernorm_weights.append(kv_a_layernorm_weight)
            self.kv_b_proj_weights.append(kv_b_proj_weight)
            self.qkv_weights.append(qkv_weight)

            self.q_nope_k_b_proj_weights.append(q_nope_k_b_proj_weight)
            self.q_rope_proj_weights.append(q_rope_proj_weight)
            self.v_b_o_proj_weights.append(v_b_o_proj_weight)

            self.linear_weights.append(linear_weight)

            self.gate_weights.append(gate_weight)
            self.ffn1_weights.append(ffn1_weight)
            self.ffn2_weights.append(ffn2_weight)

            self.shared_expert_ffn1_weights.append(shared_expert_ffn1_weight)
            self.shared_expert_ffn2_weights.append(shared_expert_ffn2_weight)
            self.shared_expert_gate_weights.append(shared_expert_gate_weight)

            self._add_parameter(q_proj_weight)
            self._add_parameter(q_a_proj_weight)
            self._add_parameter(q_a_layernorm_weight)
            self._add_parameter(q_b_proj_weight)
            self._add_parameter(kv_a_proj_with_mqa_weight)
            self._add_parameter(kv_a_layernorm_weight)
            self._add_parameter(kv_b_proj_weight)

            self._add_parameter(q_nope_k_b_proj_weight)
            self._add_parameter(q_rope_proj_weight)
            self._add_parameter(v_b_o_proj_weight)
            self._add_parameter(qkv_weight)

            self._add_parameter(shared_expert_ffn1_weight)
            self._add_parameter(shared_expert_ffn2_weight)
            self._add_parameter(shared_expert_gate_weight)

            self._add_parameter(linear_weight)

            self._add_parameter(gate_weight)
            self._add_parameter(ffn1_weight)
            self._add_parameter(ffn2_weight)

    def get_weight_create_dype(self):
        return "float8_e4m3fn"

    def per_tensor_quant_fp8(self, x):
        x_fp32 = x.cast("float32")
        x_s = x_fp32.abs().max().clip(min=0.000001) / 448.0
        x_q = x_fp32 / x_s
        x_q = x_q.clip(min=-448.0, max=448.0)
        return x_q.cast("float8_e4m3fn"), x_s

    def dynamic_quant(self, x):
        if self.weight_block_size[0] == 0 and self.weight_block_size[1] == 0:
            x_q, x_s = self.per_tensor_quant_fp8(x)
        else:
            from paddlenlp.ops.triton_ops.fused_moe import per_token_group_quant_fp8_api

            x_q, x_s = per_token_group_quant_fp8_api(x, 128, True)
            # x_q, x_s = group_quant(
            #     x, group_size=128, transpose_scale=True, quant_max_bound=448.0, quant_min_bound=-448.0
            # )
        return x_q, x_s

    def cutlass_fp8_gemm(
        self,
        x,
        y,
        x_s=None,
        y_s=None,
        bias=None,
        output_dtype="bfloat16",
        act="identity",
        ffn1=False,
    ):
        if self.weight_block_size[0] == 0 and self.weight_block_size[1] == 0:
            if x_s is None:
                x_q, x_s = self.dynamic_quant(x)
            else:
                x_q = x
            try:
                from paddlenlp_ops import (
                    cutlass_fp8_fp8_half_gemm_ptr_scale_fused as fp8_gemm_fused_ptr_scale,
                )
            except:
                assert False, "fp8_gemm_fused_ptr_scale only supported on sm90"
            if ffn1:
                n, k = y.shape
                y_0 = y[: n // 2, :]
                y_1 = y[n // 2 :, :]
                y_s_0 = y_s[0, 0]
                y_s_1 = y_s[1, 0]
                out_0 = fp8_gemm_fused_ptr_scale(
                    x=x_q,
                    y=y_0,
                    x_scale=x_s,
                    y_scale=y_s_0,
                    bias=bias,
                    transpose_x=False,
                    transpose_y=True,
                    output_dtype=output_dtype,
                )
                out_1 = fp8_gemm_fused_ptr_scale(
                    x=x_q,
                    y=y_1,
                    x_scale=x_s,
                    y_scale=y_s_1,
                    bias=bias,
                    transpose_x=False,
                    transpose_y=True,
                    output_dtype=output_dtype,
                )
                out = paddle.concat([out_0, out_1], axis=-1)
            else:
                out = fp8_gemm_fused_ptr_scale(
                    x=x_q,
                    y=y,
                    x_scale=x_s,
                    y_scale=y_s,
                    bias=bias,
                    transpose_x=False,
                    transpose_y=True,
                    output_dtype=output_dtype,
                )
        else:
            if x_s is None:
                x, x_s = self.dynamic_quant(x)
            try:
                from paddlenlp_ops import (
                    cutlass_fp8_fp8_half_block_gemm_fused as fp8_block_gemm_fused,
                )
            except:
                assert False, "fp8_block_gemm_fused only supported on sm90"
            out = fp8_block_gemm_fused(
                x,
                y,
                x_s,
                y_s,
                bias=bias,
                transpose_x=False,
                transpose_y=True,
                output_dtype=output_dtype,
                act=act,
            )
        return out

    def compute_qkv_linear(self, ln_out, i, latent_cache=None, **kwargs):
        ln_out_fp8, ln_out_scale = self.dynamic_quant(ln_out)
        if self.config.mla_config.use_mla():
            if self.config.mla_config.q_lora_rank is not None:
                query = self.cutlass_fp8_gemm(
                    x=ln_out_fp8,
                    y=self.q_a_proj_weights[i],
                    x_s=ln_out_scale,
                    y_s=self.q_a_proj_weights_scale[i],
                    bias=None,
                    output_dtype=self._dtype,
                    act="identity",
                )

                query = self.norm_func(
                    x=query,
                    norm_weight=self.q_a_layernorm_weights[i],
                    norm_bias=None,
                    epsilon=self._epsilon,
                    begin_norm_axis=1,
                )[0]

                query = self.cutlass_fp8_gemm(
                    x=query,
                    y=self.q_b_proj_weights[i],
                    y_s=self.q_b_proj_weights_scale[i],
                    bias=None,
                    output_dtype=self._dtype,
                    act="identity",
                )
            else:
                query = self.cutlass_fp8_gemm(
                    x=ln_out_fp8,
                    y=self.q_proj_weights[i],
                    x_s=ln_out_scale,
                    y_s=self.q_proj_weights_scale[i],
                    bias=None,
                    output_dtype=self._dtype,
                    act="identity",
                )

            query = query.reshape([-1, self.num_heads, self.config.mla_config.qk_head_dim])
            query_nope, query_pe = query.split(
                [self.config.mla_config.qk_nope_head_dim, self.config.mla_config.qk_rope_head_dim], axis=-1
            )

            compressed_kv = self.cutlass_fp8_gemm(
                x=ln_out_fp8,
                y=self.kv_a_proj_with_mqa_weights[i],
                x_s=ln_out_scale,
                y_s=self.kv_a_proj_with_mqa_weights_scale[i],
                bias=None,
                output_dtype=self._dtype,
                act="identity",
            )
            compressed_kv, key_pe = compressed_kv.split(
                [self.config.mla_config.kv_lora_rank, self.config.mla_config.qk_rope_head_dim], axis=-1
            )
            key_pe = key_pe.reshape([-1, 1, self.config.mla_config.qk_rope_head_dim])
            compressed_kv = self.norm_func(
                x=compressed_kv,
                norm_weight=self.kv_a_layernorm_weights[i],
                norm_bias=None,
                epsilon=self._epsilon,
                begin_norm_axis=1,
            )[0]
            query_pe, key_pe = self.config.rotary_emb(self.position_ids, query_pe, key_pe)

            if self.config.mla_config.use_absorb():
                from paddlenlp_ops import prefill_mla_write_cache

                prefill_mla_write_cache(
                    compressed_kv,
                    key_pe,
                    latent_cache,
                    kwargs.get("seq_lens_encoder", None),
                    kwargs.get("seq_lens_decoder", None),
                    kwargs.get("padding_offsets", None),
                    kwargs.get("cum_offsets", None),
                    kwargs.get("block_tables", None),
                    "none",
                    kwargs.get("max_input_length", -1),
                )

            key_value = self.cutlass_fp8_gemm(
                x=compressed_kv,
                y=self.kv_b_proj_weights[i],
                y_s=self.kv_b_proj_weights_scale[i],
                bias=None,
                output_dtype=self._dtype,
                act="identity",
            )
            key_value = key_value.reshape(
                [-1, self.num_heads, self.config.mla_config.qk_nope_head_dim + self.config.mla_config.v_head_dim]
            )
            key_nope, value = key_value.split(
                [self.config.mla_config.qk_nope_head_dim, self.config.mla_config.v_head_dim], axis=-1
            )

            query[..., self.config.mla_config.qk_nope_head_dim :] = query_pe
            key = paddle.empty_like(query)
            key[..., : self.config.mla_config.qk_nope_head_dim] = key_nope
            key[..., self.config.mla_config.qk_nope_head_dim :] = key_pe

            if self.config.mla_config.use_absorb():
                value = paddle.nn.functional.pad(
                    value, [0, self.config.mla_config.qk_head_dim - self.config.mla_config.v_head_dim], value=0
                )
                return query, key, value
            else:
                qkv_out = paddle.concat(
                    [
                        query.reshape([-1, self.num_heads * self.config.mla_config.qk_head_dim]),
                        key.reshape([-1, self.num_heads * self.config.mla_config.qk_head_dim]),
                        value.reshape([-1, self.num_heads * self.config.mla_config.v_head_dim]),
                    ],
                    axis=-1,
                )
                return qkv_out
        else:
            qkv_out = self.cutlass_fp8_gemm(
                x=ln_out_fp8,
                y=self.qkv_weights[i],
                x_s=ln_out_scale,
                y_s=self.qkv_weights_scale[i],
                bias=self.qkv_biases[i],
                output_dtype=self._dtype,
                act="identity",
            )
            return qkv_out

    def compute_out_linear(self, fmha_out, i):
        out = self.cutlass_fp8_gemm(
            x=fmha_out,
            y=self.linear_weights[i],
            y_s=self.linear_weights_scale[i],
            bias=None,
            output_dtype=self._dtype,
            act="identity",
        )
        return out

    def compute_mla_absorb(
        self,
        qkv_out,
        caches,
        i,
        **kwargs,
    ):
        from paddlenlp_ops import decode_mla_write_cache, multi_head_latent_attention

        ln_out = qkv_out
        latent_cache = caches[i]

        out_linear_out = paddle.zeros(shape=[ln_out.shape[0], self.embed_dim], dtype=ln_out.dtype)

        if kwargs["max_enc_len_this_time"]:  # prefill phase
            query, key, value = self.compute_qkv_linear(ln_out, i, latent_cache=latent_cache, **kwargs)

            from paddlenlp.utils.env import PREFILL_USE_SAGE_ATTN

            if PREFILL_USE_SAGE_ATTN:
                from paddlenlp_ops import sage_attention_dsk

                query_256 = paddle.nn.functional.pad(paddle.unsqueeze(query, axis=0), (0, 256 - 192))
                key_256 = paddle.nn.functional.pad(paddle.unsqueeze(key, axis=0), (0, 256 - 192))

                value_128, _ = paddle.split(value, [128, 64], axis=-1)
                value_128 = paddle.unsqueeze(value_128, axis=0)

                km = paddle.mean(key_256, axis=1, keepdim=True)
                km = km.squeeze(1)

                fmha_out_prefill = sage_attention_dsk(
                    query_256,
                    key_256,
                    value_128,
                    km,
                    kwargs.get("cu_seqlens_q", None),
                    None,  # vm
                    self.softmax_scale,
                    "per_warp",  # qk_quant_gran
                    "fp16",  # pv_accum_dtype
                    0,  # tensor layout, 0 -> NHD
                    True,  # is_causal
                    True,  # smooth k
                    False,  # smooth v
                    False,  # return lse
                )
                fmha_out_prefill = paddle.nn.functional.pad(fmha_out_prefill, (0, 192 - 128))
                fmha_out_prefill = paddle.squeeze(fmha_out_prefill, axis=0)
            else:
                fmha_out_prefill = paddle.nn.functional.flash_attention.flash_attn_unpadded(
                    query,
                    key,
                    value,
                    kwargs.get("cu_seqlens_q", None),
                    kwargs.get("cu_seqlens_k", None),
                    kwargs.get("max_enc_len_this_time", -1),
                    kwargs.get("max_enc_len_this_time", -1),
                    self.softmax_scale,
                    causal=True,
                    training=False,
                )[0]

            fmha_out_prefill = fmha_out_prefill.reshape([-1, self.num_heads, self.config.mla_config.qk_head_dim])
            fmha_out_prefill = fmha_out_prefill[:, :, : self.config.mla_config.v_head_dim]
            fmha_out_prefill = fmha_out_prefill.reshape([-1, self.num_heads * self.config.mla_config.v_head_dim])

            fmha_out_prefill = fmha_out_prefill * self.mask_encoder_batch.cast(fmha_out_prefill.dtype)

            out_linear_out_prefill = self.compute_out_linear(fmha_out_prefill, i)
            out_linear_out = out_linear_out + out_linear_out_prefill

        if kwargs["max_dec_len_this_time"]:  # decode phase
            if self.config.mla_config.q_lora_rank is not None:
                ln_out_fp8, ln_out_scale = self.dynamic_quant(ln_out)
                query = self.cutlass_fp8_gemm(
                    x=ln_out_fp8,
                    y=self.q_a_proj_weights[i],
                    x_s=ln_out_scale,
                    y_s=self.q_a_proj_weights_scale[i],
                    bias=None,
                    output_dtype=self._dtype,
                    act="identity",
                )
                query = self.norm_func(
                    x=query,
                    norm_weight=self.q_a_layernorm_weights[i],
                    norm_bias=None,
                    epsilon=self._epsilon,
                    begin_norm_axis=1,
                )[0]
                ln_out_or_q_c = query
            else:
                ln_out_or_q_c = ln_out

            ln_out_or_q_c_fp8, ln_out_or_q_c_scale = self.dynamic_quant(ln_out_or_q_c)
            compressed_kv = self.cutlass_fp8_gemm(
                x=ln_out_fp8 if self.config.mla_config.q_lora_rank is not None else ln_out_or_q_c_fp8,
                y=self.kv_a_proj_with_mqa_weights[i],
                x_s=ln_out_scale if self.config.mla_config.q_lora_rank is not None else ln_out_or_q_c_scale,
                y_s=self.kv_a_proj_with_mqa_weights_scale[i],
                bias=None,
                output_dtype=self._dtype,
                act="identity",
            )
            compressed_kv, key_pe = compressed_kv.split(
                [self.config.mla_config.kv_lora_rank, self.config.mla_config.qk_rope_head_dim], axis=-1
            )
            key_pe = key_pe.reshape([-1, 1, self.config.mla_config.qk_rope_head_dim])
            compressed_kv = self.norm_func(
                x=compressed_kv,
                norm_weight=self.kv_a_layernorm_weights[i],
                norm_bias=None,
                epsilon=self._epsilon,
                begin_norm_axis=1,
            )[0]
            query_nope = self.cutlass_fp8_gemm(
                x=ln_out_or_q_c_fp8,
                y=self.q_nope_k_b_proj_weights[i],
                x_s=ln_out_or_q_c_scale,
                y_s=self.q_nope_k_b_proj_weights_scale[i],
                bias=None,
                output_dtype=self._dtype,
                act="identity",
            )
            query_nope = query_nope.reshape(shape=[-1, self.num_heads, self.config.mla_config.kv_lora_rank])
            query_pe = self.cutlass_fp8_gemm(
                x=ln_out_or_q_c_fp8,
                y=self.q_rope_proj_weights[i],
                y_s=self.q_rope_proj_weights_scale[i],
                x_s=ln_out_or_q_c_scale,
                bias=None,
                output_dtype=self._dtype,
                act="identity",
            )
            query_pe = query_pe.reshape(shape=[-1, self.num_heads, self.config.mla_config.qk_rope_head_dim])
            query_pe, key_pe = self.config.rotary_emb(self.position_ids, query_pe, key_pe)

            decode_mla_write_cache(
                compressed_kv,
                key_pe,
                latent_cache,
                kwargs.get("seq_lens_decoder", None),
                kwargs.get("seq_lens_encoder", None),
                kwargs.get("padding_offsets", None),
                kwargs.get("cum_offsets", None),
                kwargs.get("block_tables", None),
                "none",
                kwargs.get("max_input_length", -1),
                self.config.speculate_config.speculate_method is not None,  # speculate_decoder
            )

            q_input = paddle.concat([query_nope, query_pe], axis=-1)
            q_input = q_input.reshape(
                [
                    -1,
                    self.num_heads * (self.config.mla_config.kv_lora_rank + self.config.mla_config.qk_rope_head_dim),
                ]
            )

            fmha_out_decode = multi_head_latent_attention(
                q_input,
                latent_cache,
                latent_cache,
                kwargs.get("seq_lens_encoder", None),
                kwargs.get("seq_lens_decoder", None),
                kwargs.get("seq_lens_this_time", None),
                kwargs.get("cu_seqlens_q", None),
                kwargs.get("padding_offsets", None),
                kwargs.get("cum_offsets", None),
                kwargs.get("block_tables", None),
                kwargs.get("encoder_batch_ids", None),
                kwargs.get("encoder_tile_ids_per_batch", None),
                kwargs.get("encoder_num_blocks", None),
                kwargs.get("kv_batch_ids", None),
                kwargs.get("kv_tile_ids_per_batch", None),
                kwargs.get("kv_num_blocks", None),
                kwargs.get("decoder_batch_ids", None),
                kwargs.get("decoder_tile_ids_per_batch", None),
                kwargs.get("decoder_num_blocks", None),
                kwargs.get("decoder_num_blocks_cpu", None),
                kwargs.get("max_enc_len_this_time", None),
                kwargs.get("max_dec_len_this_time", None),
                kwargs.get("max_len_kv", None),
                None,  # attn_mask
                None,  # qkv_bias
                None,  # qkv_out_scales
                None,  # cache_k_quant_scales
                None,  # cache_v_quant_scales
                None,  # cache_k_dequant_scales
                None,  # cache_v_dequant_scales
                None,  # cache_k_zp
                None,  # cache_v_zp
                None,  # out_shifts
                None,  # out_smooths
                self._fuse_kernel_compute_dtype,
                "none",  # cache_quant_type
                self.config.mla_config.kv_lora_rank,
                kwargs.get("max_input_length", -1),
                self.softmax_scale,  # softmax_scale
                0.0,  # quant_max_bound
                0.0,  # quant_min_bound
                0.0,  # out_linear_in_scale
                self.config.speculate_config.speculate_max_draft_token_num,
                True,  # causal
                self.config.speculate_config.speculate_method is not None,  # speculate_decoder
            )
            fmha_out_decode_fp8, fmha_out_decode_scale = self.dynamic_quant(fmha_out_decode)
            out_linear_out_decode = self.cutlass_fp8_gemm(
                x=fmha_out_decode_fp8,
                y=self.v_b_o_proj_weights[i],
                y_s=self.v_b_o_proj_weights_scale[i],
                x_s=fmha_out_decode_scale,
                bias=None,
                output_dtype=self._dtype,
                act="identity",
            )
            out_linear_out = out_linear_out + out_linear_out_decode

        return out_linear_out

    def compute_ffn1(self, tmp_out, i):
        out = self.cutlass_fp8_gemm(
            x=tmp_out,
            y=self.ffn1_weights[i],
            y_s=self.ffn1_weights_scale[i],
            bias=None,
            output_dtype=self._dtype,
            act="identity",
            ffn1=True,
        )
        return out

    def compute_ffn2(self, ffn1_out, i):
        out = self.cutlass_fp8_gemm(
            x=ffn1_out,
            y=self.ffn2_weights[i],
            y_s=self.ffn2_weights_scale[i],
            bias=None,
            output_dtype=self._dtype,
            act="identity",
        )
        return out

    def compute_fused_moe(self, tmp_out, i):
        e_score_correction_bias = self.e_score_correction_biases[i]

        def get_moe_scores(
            gating_output: paddle.Tensor,
            config: MoeConfig,
        ) -> paddle.Tensor:
            # Compute softmax or sigmoid scores based on the topk_method
            if config.topk_method == "greedy":
                scores = paddle.nn.functional.softmax(gating_output, axis=-1)
                return scores
            elif config.topk_method == "group_limited_greedy":
                scores = paddle.nn.functional.softmax(gating_output, axis=-1)
                scores_with_bias = scores
            elif config.topk_method == "noaux_tc":
                if e_score_correction_bias is None:
                    raise ValueError("e_score_correction_bias must be provided for 'noaux_tc' method.")
                scores = paddle.nn.functional.sigmoid(gating_output)
                scores_with_bias = scores + e_score_correction_bias.unsqueeze(0)
            else:
                raise ValueError(
                    f"Unsupported topk_method: {config.topk_method}. Please choose 'group_limited_greedy' or 'noaux_tc'."
                )
            from paddlenlp_ops import noaux_tc

            scores = noaux_tc(
                scores,
                scores_with_bias,
                config.num_expert_group,
                config.topk_group,
                config.top_k,
                config.routed_scaling_factor,
            )
            return scores

        if self.config.moe_config.topk_method is not None:
            gate_out = paddle.matmul(tmp_out.cast("float32"), self.gate_weights[i])
            # 应用各种策略后重塑的 scores
            scores = get_moe_scores(gate_out, self.config.moe_config)

            if self.moe_quant_type in ["weight_only_int4", "weight_only_int8"]:
                from paddle.incubate.nn.functional import (
                    moe_dispatch,
                    moe_ffn,
                    moe_reduce,
                )

                # topk 在 moe_dispatch 中
                (
                    permute_input,
                    token_nums_per_expert,
                    permute_indices_per_token,
                    top_k_weights,
                    top_k_indices,
                ) = moe_dispatch(tmp_out, scores, self.config.moe_config.top_k, False, topk_only_mode=True)

                ffn_out = moe_ffn(
                    permute_input,
                    token_nums_per_expert,
                    self.ffn1_weights[i],
                    self.ffn2_weights[i],
                    self.ffn1_biases[i],
                    self.ffn1_weights_scale[i] if hasattr(self, "ffn1_weights_scale") else None,
                    self.ffn2_weights_scale[i] if hasattr(self, "ffn2_weights_scale") else None,
                    self.moe_quant_type,
                )

                fused_moe_out = moe_reduce(
                    ffn_out,
                    top_k_weights,
                    permute_indices_per_token,
                    top_k_indices,
                    self.ffn2_biases[i],
                    norm_topk_prob=False,  # 在noaux_tc中做了
                    routed_scaling_factor=1.0,  # 在noaux_tc中做了
                )
            else:
                from paddlenlp.ops.triton_ops.fused_moe import fused_moe

                fused_moe_out = fused_moe(
                    tmp_out,
                    self.ffn1_weights[i],
                    self.ffn2_weights[i],
                    scores,
                    self.config.moe_config.top_k,
                    use_fp8_w8a8=True,
                    w1_scale=self.ffn1_weights_scale[i] if hasattr(self, "ffn1_weights_scale") else None,
                    w2_scale=self.ffn2_weights_scale[i] if hasattr(self, "ffn2_weights_scale") else None,
                    block_shape=self.weight_block_size
                    if sum(self.weight_block_size) != 0
                    else None,  # default block-wise, per-tensor is None
                )
        else:
            assert False, "Not implemented yet"
        return fused_moe_out

    def compute_shared_expert(self, tmp_out, i):
        ffn1_out = self.cutlass_fp8_gemm(
            x=tmp_out,
            y=self.shared_expert_ffn1_weights[i],
            y_s=self.shared_expert_ffn1_weights_scale[i],
            bias=None,
            output_dtype=self._dtype,
            act="identity",
        )
        ffn1_out = fused_bias_act(ffn1_out, None, act_method=self.activation)

        ffn2_out = self.cutlass_fp8_gemm(
            x=ffn1_out,
            y=self.shared_expert_ffn2_weights[i],
            y_s=self.shared_expert_ffn2_weights_scale[i],
            bias=None,
            output_dtype=self._dtype,
            act="identity",
        )
        if self.config.moe_config.shared_expert_with_gate:
            gate_out = paddle.matmul(tmp_out, self.shared_expert_gate_weights[i])
            gate_out = paddle.nn.functional.sigmoid(gate_out)
            return gate_out * ffn2_out
        return ffn2_out
