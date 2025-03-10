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

import os
import shutil
import subprocess

import paddle
from paddle.utils.cpp_extension import CUDAExtension, setup

sm_version = int(os.getenv("CUDA_SM_VERSION", "0"))


def update_git_submodule():
    try:
        subprocess.run(["git", "submodule", "update", "--init"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while updating git submodule: {str(e)}")
        raise


def find_end_files(directory, end_str):
    gen_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(end_str):
                gen_files.append(os.path.join(root, file))
    return gen_files


def get_sm_version():
    if sm_version > 0:
        return sm_version
    else:
        prop = paddle.device.cuda.get_device_properties()
        cc = prop.major * 10 + prop.minor
        return cc


def strtobool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


def get_gencode_flags():
    if not strtobool(os.getenv("FLAG_LLM_PDC", "False")):
        cc = get_sm_version()
        if cc == 90:
            cc = f"{cc}a"
        return ["-gencode", "arch=compute_{0},code=sm_{0}".format(cc)]
    else:
        # support more cuda archs
        return [
            "-gencode",
            "arch=compute_80,code=sm_80",
            "-gencode",
            "arch=compute_75,code=sm_75",
            "-gencode",
            "arch=compute_70,code=sm_70",
        ]


gencode_flags = get_gencode_flags()
library_path = os.environ.get("LD_LIBRARY_PATH", "/usr/local/cuda/lib64")

sources = [
    "./gpu/save_with_output.cc",
    "./gpu/set_value_by_flags.cu",
    "./gpu/token_penalty_multi_scores.cu",
    "./gpu/token_penalty_multi_scores_v2.cu",
    "./gpu/stop_generation_multi_ends.cu",
    "./gpu/fused_get_rope.cu",
    "./gpu/get_padding_offset.cu",
    "./gpu/qkv_transpose_split.cu",
    "./gpu/rebuild_padding.cu",
    "./gpu/transpose_removing_padding.cu",
    "./gpu/write_cache_kv.cu",
    "./gpu/encode_rotary_qk.cu",
    "./gpu/get_padding_offset_v2.cu",
    "./gpu/rebuild_padding_v2.cu",
    "./gpu/set_value_by_flags_v2.cu",
    "./gpu/stop_generation_multi_ends_v2.cu",
    "./gpu/get_output.cc",
    "./gpu/save_with_output_msg.cc",
    "./gpu/write_int8_cache_kv.cu",
    "./gpu/step.cu",
    "./gpu/quant_int8.cu",
    "./gpu/dequant_int8.cu",
    "./gpu/group_quant.cu",
    "./gpu/preprocess_for_moe.cu",
    "./gpu/get_position_ids_and_mask_encoder_batch.cu",
    "./gpu/fused_rotary_position_encoding.cu",
    "./gpu/flash_attn_bwd.cc",
    "./gpu/tune_cublaslt_gemm.cu",
    "./gpu/sample_kernels/top_p_sampling_reject.cu",
    "./gpu/update_inputs_v2.cu",
    "./gpu/noaux_tc.cu",
    "./gpu/set_preids_token_penalty_multi_scores.cu",
    "./gpu/speculate_decoding_kernels/ngram_match.cc",
    "./gpu/speculate_decoding_kernels/speculate_save_output.cc",
    "./gpu/speculate_decoding_kernels/speculate_get_output.cc",
]
sources += find_end_files("./gpu/speculate_decoding_kernels", ".cu")

nvcc_compile_args = gencode_flags
update_git_submodule()
nvcc_compile_args += [
    "-O3",
    "-DNDEBUG",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
    "-Igpu",
    "-Igpu/cutlass_kernels",
    "-Igpu/fp8_gemm_with_cutlass",
    "-Igpu/cutlass_kernels/fp8_gemm_fused/autogen",
    "-Ithird_party/cutlass/include",
    "-Ithird_party/cutlass/tools/util/include",
    "-Ithird_party/nlohmann_json/single_include",
    "-Igpu/sample_kernels",
]

cc = get_sm_version()
cuda_version = float(paddle.version.cuda())

if cc >= 80:
    sources += ["gpu/int8_gemm_with_cutlass/gemm_dequant.cu"]

    sources += ["./gpu/append_attention.cu", "./gpu/multi_head_latent_attention.cu"]

    sources += find_end_files("./gpu/append_attn", ".cu")
    sources += find_end_files("./gpu/append_attn/template_instantiation", ".cu")


fp8_auto_gen_directory = "gpu/cutlass_kernels/fp8_gemm_fused/autogen"
if os.path.isdir(fp8_auto_gen_directory):
    shutil.rmtree(fp8_auto_gen_directory)

if cc == 89 and cuda_version >= 12.4:
    os.system("python utils/auto_gen_fp8_fp8_gemm_fused_kernels.py --cuda_arch 89")
    os.system("python utils/auto_gen_fp8_fp8_dual_gemm_fused_kernels.py --cuda_arch 89")
    sources += find_end_files(fp8_auto_gen_directory, ".cu")
    sources += [
        "gpu/fp8_gemm_with_cutlass/fp8_fp8_half_gemm.cu",
        "gpu/fp8_gemm_with_cutlass/fp8_fp8_half_cuda_core_gemm.cu",
        "gpu/fp8_gemm_with_cutlass/fp8_fp8_fp8_dual_gemm.cu",
    ]

if cc >= 80 and cuda_version >= 12.4:
    nvcc_compile_args += [
        "-std=c++17",
        "--use_fast_math",
        "--threads=8",
        "-D_GLIBCXX_USE_CXX11_ABI=1",
    ]
    sources += ["./gpu/sage_attn_kernels/sageattn_fused.cu"]
    if cc >= 80 and cc < 89:
        sources += [
            "./gpu/sage_attn_kernels/sageattn_qk_int_sv_f16_kernel_sm80.cu"
        ]
        nvcc_compile_args += ["-gencode", f"arch=compute_80,code=compute_80"]
    elif cc >= 89 and cc < 90:
        sources += [
            "./gpu/sage_attn_kernels/sageattn_qk_int_sv_f8_kernel_sm89.cu"
        ]
        nvcc_compile_args += ["-gencode", f"arch=compute_89,code=compute_89"]
    elif cc >= 90:
        sources += [
            "./gpu/sage_attn_kernels/sageattn_qk_int_sv_f8_kernel_sm90.cu",
            "./gpu/sage_attn_kernels/sageattn_qk_int_sv_f8_dsk_kernel_sm90.cu"
        ]
        nvcc_compile_args += ["-gencode", f"arch=compute_90a,code=compute_90a"]

if cc >= 90 and cuda_version >= 12.0:
    os.system("python utils/auto_gen_fp8_fp8_gemm_fused_kernels_sm90.py --cuda_arch 90")
    os.system("python utils/auto_gen_fp8_fp8_gemm_fused_kernels_ptr_scale_sm90.py --cuda_arch 90")
    os.system("python utils/auto_gen_fp8_fp8_dual_gemm_fused_kernels_sm90.py --cuda_arch 90")
    os.system("python utils/auto_gen_fp8_fp8_block_gemm_fused_kernels_sm90.py --cuda_arch 90")
    sources += find_end_files(fp8_auto_gen_directory, ".cu")
    sources += [
        "gpu/fp8_gemm_with_cutlass/fp8_fp8_half_gemm.cu",
        "gpu/fp8_gemm_with_cutlass/fp8_fp8_half_cuda_core_gemm.cu",
        "gpu/fp8_gemm_with_cutlass/fp8_fp8_fp8_dual_gemm.cu",
        "gpu/fp8_gemm_with_cutlass/fp8_fp8_half_block_gemm.cu",
        "gpu/fp8_gemm_with_cutlass/fp8_fp8_half_gemm_ptr_scale.cu",
    ]
    sources += find_end_files("./gpu/mla_attn", ".cu")

ops_name = f"paddlenlp_ops_{sm_version}" if sm_version != 0 else "paddlenlp_ops"
setup(
    name=ops_name,
    ext_modules=CUDAExtension(
        sources=sources,
        extra_compile_args={"cxx": ["-O3", "-fopenmp", "-lgomp", "-std=c++17", "-DENABLE_BF16"], "nvcc": nvcc_compile_args},
        libraries=["cublasLt"],
        library_dirs=[library_path],
    ),
)
