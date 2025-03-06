# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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


import argparse
import copy
import os
import re


def get_candidate_tiles():
    cta_shape = [
        ("<_128, _128, _128>"),
        # ("<_256, _128, _128>"),
    ]
    cluster_shape = [
        ("<_1, _1, _1>"),
        ("<_2, _1, _1>"),
        ("<_1, _2, _1>"),
        ("<_2, _2, _1>"),
        # ("<_1, _8, _1>"),
        # ("<_8, _1, _1>"),
    ]
    base_configs = [(x, y) for x in cta_shape for y in cluster_shape]

    return base_configs


def get_candidate_configs(sm):
    tiles = get_candidate_tiles()
    candidate_configs = list()

    hasbias = ("false", "true")
    KernelSchedule = ("KernelTmaWarpSpecializedCooperativeFP8BlockScaledSubGroupMAccum<1>",)
    EpilogueSchedule = ("TmaWarpSpecializedCooperative",)
    TileSchedule = ("PersistentScheduler", "StreamKScheduler")
    for act_tag in [
        ("noact", "Identity"),
        # ("relu", "ReLu"),
        # ("gelu", "GELU"),
    ]:
        candidate_configs.extend([(hasbias, act_tag, tiles, KernelSchedule, EpilogueSchedule, TileSchedule)])
    return candidate_configs


def get_shape_str(tile_shape):
    blocks, clusters = [s.replace(" ", "").strip("<>").split(",") for s in tile_shape]
    blocks = [elem.strip("_") for elem in blocks]
    clusters = [elem.strip("_") for elem in clusters]
    return blocks, clusters


def check_config_valid(tile_shape, kernel_schedule, epilogue_schedule, tile_schedule):
    blocks, clusters = get_shape_str(tile_shape)
    if int(blocks[0]) < 128 and "Cooperative" in kernel_schedule:
        return False
    # if "Cooperative" in kernel_schedule and "Cooperative" not in epilogue_schedule:
    #     return False
    # if (
    #     tile_shape[0] == "<_256, _128, _128>"
    #     and "Cooperative" not in kernel_schedule
    #     and "Cooperative" not in epilogue_schedule
    # ):
    #     return False
    # flag1 = (int(blocks[0]) == 64 and kernel_schedule == "KernelTmaWarpSpecializedPingpongFP8FastAccum" and epilogue_schedule == "TmaWarpSpecialized")
    # flag2 = (int(blocks[0]) != 64 and kernel_schedule == "KernelTmaWarpSpecializedCooperativeFP8FastAccum" and epilogue_schedule == "TmaWarpSpecializedCooperative")
    # if not (flag1 or flag2):
    #     return False
    return True


# this is a file's header part
CommonHead = """// Generated by auto_gen_fp8_fp8_gemm_fused_kernels_sm90.py - Do not edit.

#pragma once

#include "fp8_gemm_fused/fuse_block_gemm_act_template_3x.h"

"""

GemmDeclare = """
template<>
bool dispatch_fuse_block_gemm_c3x<phi::dtype::{input_type}, phi::dtype::{output_type},
                                 {hasbias},
                                 cutlass::epilogue::thread::{Activation},
                                 Shape{TileShape},
                                 Shape{ClusterShape},
                                 cutlass::gemm::{KernelSchedule},
                                 cutlass::epilogue::{EpilogueSchedule},
                                 cutlass::gemm::{TileSchedule},
                                 {SM}
                                 >(GemmEpilogueAllParams);

"""


LaunchGemmHead = """
#pragma once

#include "fp8_gemm_fused/fp8_fp8_gemm_scale_bias_act.h"

"""

LaunchGemmDeclare = """
bool launch_block_gemm_kernel_sm{sm}_{gemm_config}(const int type_id, GemmEpilogueAllParams params);
"""

LaunchGemmPart0 = """
#pragma once

#include "launch_block_gemm_kernel_sm{sm}.h"

bool launch_block_gemm_kernel_sm{sm}_{gemm_config}(const int type_id, GemmEpilogueAllParams params){
    switch (type_id) {
"""

LaunchGemmPart1 = """
        case {type_id}:
            return dispatch_fuse_block_gemm_c3x<phi::dtype::{input_type}, phi::dtype::{output_type},
                                 {hasbias},
                                 cutlass::epilogue::thread::{Activation},
                                 Shape{TileShape},
                                 Shape{ClusterShape},
                                 cutlass::gemm::{KernelSchedule},
                                 cutlass::epilogue::{EpilogueSchedule},
                                 cutlass::gemm::{TileSchedule},
                                 {SM}
                                 >(params);
            break;
"""

LaunchGemmPart2 = """
        default:
            throw std::runtime_error("cutlass gemm config is invalid.");
            break;
    }
    return false;
}
"""


code_part0 = """// Generated by auto_gen_fp8_fp8_gemm_fused_kernels_sm90.py - Do not edit.

#include <map>
#include "fp8_gemm_fused/fp8_fp8_gemm_scale_bias_act.h"
#include "launch_block_gemm_kernel_sm{sm}.h"

COMMON_DECLARE_string(use_cutlass_device_best_config_path);

std::map<std::string, int> block_gemm_type_map{"""

code_part1 = """
    {"{input_type}_{output_type}_{hasbias}_{act_tag}",   {type_id}}, """

code_part2 = """
};

std::map<std::string, int> block_gemm_config_map{
"""

code_part3 = """    {"{TileShape}, {ClusterShape}, {kernel_schedule}, {epilogue_schedule}, {tile_schedule}", {tile_id}},
"""

code_part4 = """};

bool launch_block_gemm_kernel(const int type_id, const int kernel_id, GemmEpilogueAllParams params){
    switch (kernel_id) {"""

code_part5 = """
        case {tile_id}:
            return launch_block_gemm_kernel_sm{sm}_{gemm_config}(type_id, params);
            break;
"""

code_part6 = """
        default:
            throw std::runtime_error("fp8 gemm_fused Config is invalid.");
            break;
    }
    return false;
}

template <typename T>
T get_relative_best(nlohmann::json* json_data,
                    const std::string& target_key,
                    const int& m,
                    const int& n,
                    const int& k) {
    if (json_data->contains(target_key)) {
        return json_data->at(target_key);
    } else {
        if (k > 3 * n){
            return "<_128, _128, _128>, <_1, _2, _1>, KernelTmaWarpSpecializedCooperativeFP8BlockScaledSubGroupMAccum<1>, TmaWarpSpecializedCooperative, StreamKScheduler";
        }else{
            return "<_128, _128, _128>, <_1, _2, _1>, KernelTmaWarpSpecializedCooperativeFP8BlockScaledSubGroupMAccum<1>, TmaWarpSpecializedCooperative, PersistentScheduler";
        }

    }
}

bool fp8_fp8_block_gemm_scale_bias_act(GemmEpilogueAllParams params) {
  if (block_gemm_type_map.find(params.fuse_gemm_config) == block_gemm_type_map.end()) {
    throw std::runtime_error("fp8 gemm_fused config is invalid.");
  }

  int type_id = block_gemm_type_map[params.fuse_gemm_config];
  int M = (params.M + 31) / 32 * 32;
  int N = params.N;
  int K = params.K;

  int kernel_id;
  std::string mnk_string = "block_gemm_sm90<"+ std::to_string(M)+ ", " +std::to_string(N) + ", "+ std::to_string(K)+ ">";
  std::string best_config;
  CutlassGemmConfigMannager& best_config_mannager = CutlassGemmConfigMannager::getInstance();
  char *config_file_path_c_str = getenv("FLAGS_use_cutlass_device_best_config_path");
  std::string config_file_path = config_file_path_c_str == nullptr ? "" : std::string(config_file_path_c_str);
  if(config_file_path == "tune"){ // tune kernel
    int warm_up_times = 5;
    int tune_times = 10;
    std::string best_kernel_id = "";
    float duratation = 1000000.f;
    // tune all kernel_id kernels
    for(const auto& config_pair : block_gemm_config_map){
        std::cout << "Running tune kernel: " << config_pair.first<< std::endl;
        bool is_valid = true;
        // warm up
        for(int num_time = 0; num_time < warm_up_times; ++num_time){
            if(!launch_block_gemm_kernel(type_id, config_pair.second, params)){
                is_valid = false;
                break;
            }
        }
        if(!is_valid){
            continue;
        }
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaStreamSynchronize(params.stream);
        cudaEventRecord(start, params.stream);
        for(int num_time = 0; num_time < tune_times; ++num_time){
            if(!launch_block_gemm_kernel(type_id, config_pair.second, params)){
                is_valid = false;
                break;
            };
        }
        cudaEventRecord(stop, params.stream);
        cudaEventSynchronize(stop);
        float elapsedTime;
        if(is_valid){
            cudaEventElapsedTime(&elapsedTime, start, stop);
        } else {
            continue;
        }
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        if(elapsedTime < duratation){
            best_kernel_id = config_pair.first;
            duratation = elapsedTime;
        }
    }

    nlohmann::json new_json;
    new_json[mnk_string] = best_kernel_id;
    best_config_mannager.up_date_configs(new_json);
    std::cout <<"Gemm tune result for " << mnk_string<< ": best config is: "<< best_kernel_id << std::endl;
    return true;
  } else { // run kernel
    nlohmann::json* config_json = new nlohmann::json();
    if (config_file_path != "" && config_file_path != "default") {
        config_json = best_config_mannager.get_gemm_best_configs(config_file_path);
    }

    best_config = get_relative_best<std::string>(config_json, mnk_string, M, N, K);

    if (block_gemm_config_map.find(best_config) == block_gemm_config_map.end()) {
        throw std::runtime_error("This config'kernel not be generate, please check auto_gen_fp8_fp8_gemm_fused_kernels_sm90.py and re-generate.");
    } else {
        kernel_id = block_gemm_config_map[best_config];
    }
    return launch_block_gemm_kernel(type_id, kernel_id, params);
  }
}

"""


def SubstituteTemplate(template, values_base):
    values = copy.deepcopy(values_base)
    if values.get("KernelSchedule") is not None and "Auto" in values["KernelSchedule"]:
        values["KernelSchedule"] = "collective::" + values["KernelSchedule"]
    if values.get("EpilogueSchedule") is not None and "Auto" in values["EpilogueSchedule"]:
        values["EpilogueSchedule"] = "collective::" + values["EpilogueSchedule"]
    text = template
    changed = True
    while changed:
        changed = False
        for key, value in values.items():
            regex = f"\\{{{key}\\}}"
            newtext = re.sub(regex, value, text)
            if newtext != text:
                changed = True
            text = newtext
    return text


def parse_args():
    parser = argparse.ArgumentParser(
        description="The argument for generating the generic_mixed_gemm_kernelLauncher instance."
    )
    parser.add_argument(
        "--cuda_arch",
        type=str,
        nargs="+",
        default=["90"],
        help="The CUDA architecture to be generated.",
    )

    args = parser.parse_args()
    return args


# generate source .cu
def generate_source_cu(
    inputs_type: (str),
    outputs_type: (str),
    hasbiases: (str),
    act_tag: (str),
    tiles: (str),
    KernelSchedule: (str),
    EpilogueSchedule: (str),
    TileSchedule: (str),
    sm: str,
):
    all_code = CommonHead

    for input_type in inputs_type:
        for output_type in outputs_type:
            for hasbias in hasbiases:
                for tile_config in tiles:
                    for kernel_schedule in KernelSchedule:
                        for epilogue_schedule in EpilogueSchedule:
                            for tile_schedule in TileSchedule:
                                if not check_config_valid(
                                    tile_config, kernel_schedule, epilogue_schedule, tile_schedule
                                ):
                                    continue
                                value_dict = {
                                    "input_type": input_type,
                                    "output_type": output_type,
                                    "hasbias": hasbias,
                                    "Activation": act_tag[1],
                                    "TileShape": tile_config[0],
                                    "ClusterShape": tile_config[1],
                                    "KernelSchedule": kernel_schedule,
                                    "EpilogueSchedule": epilogue_schedule,
                                    "TileSchedule": tile_schedule,
                                    "SM": sm,
                                    "sm": sm[-2:],
                                }
                                all_code += SubstituteTemplate(GemmDeclare, value_dict)

    return all_code


# generate gemm launch .cu
def generate_launch_gemm_cus(
    generate_dir: (str), inputs_type: (str), outputs_type: (str), fuse_gemm_configs: tuple, sm: str
):
    act_tags = [single_config[1] for single_config in fuse_gemm_configs]

    single_config = fuse_gemm_configs[0]
    hasbiases: (str) = single_config[0]
    tiles: (str) = single_config[2]
    KernelSchedule: (str) = single_config[3]
    EpilogueSchedule: (str) = single_config[4]
    TileSchedule: (str) = single_config[5]
    code_map = {}
    head_path = os.path.join(generate_dir, f"launch_block_gemm_kernel_sm{sm[-2:]}.h")
    head_all_code = LaunchGemmHead
    for tile_config in tiles:
        blocks, clusters = get_shape_str(tile_config)
        gemm_config_str_0 = f"tile{blocks[0]}x{blocks[1]}x{blocks[2]}_cluster{clusters[0]}x{clusters[1]}x{clusters[2]}"
        for kernel_schedule in KernelSchedule:
            gemm_config_str_1 = gemm_config_str_0 + f"_{kernel_schedule}"
            for epilogue_schedule in EpilogueSchedule:
                gemm_config_str_2 = gemm_config_str_1 + f"_{epilogue_schedule}"
                for tile_schedule in TileSchedule:
                    if not check_config_valid(tile_config, kernel_schedule, epilogue_schedule, tile_schedule):
                        continue
                    gemm_config_str = gemm_config_str_2 + f"_{tile_schedule}"
                    value_dict = {
                        "sm": sm[-2:],
                        "gemm_config": gemm_config_str.replace("<", "").replace(">", ""),
                    }
                    head_all_code += SubstituteTemplate(LaunchGemmDeclare, value_dict)
    os.makedirs(generate_dir, exist_ok=True)
    with open(head_path, "w") as f:
        f.write(head_all_code)
        f.close()

    for tile_shape in tiles:
        blocks, clusters = get_shape_str(tile_shape)
        gemm_config_str_0 = f"tile{blocks[0]}x{blocks[1]}x{blocks[2]}_cluster{clusters[0]}x{clusters[1]}x{clusters[2]}"
        for kernel_schedule in KernelSchedule:
            gemm_config_str_1 = gemm_config_str_0 + f"_{kernel_schedule}"
            for epilogue_schedule in EpilogueSchedule:
                gemm_config_str_2 = gemm_config_str_1 + f"_{epilogue_schedule}"
                for tile_schedule in TileSchedule:
                    if not check_config_valid(tile_shape, kernel_schedule, epilogue_schedule, tile_schedule):
                        continue
                    gemm_config_str = gemm_config_str_2 + f"_{tile_schedule}"
                    value_dict = {
                        "sm": sm[-2:],
                        "gemm_config": gemm_config_str.replace("<", "").replace(">", ""),
                    }
                    source_all_code = SubstituteTemplate(LaunchGemmPart0, value_dict)
                    type_id = 0
                    for input_type in inputs_type:
                        for output_type in outputs_type:
                            for act_tag in act_tags:
                                for hasbias in hasbiases:
                                    value_dict = {
                                        "type_id": str(type_id),
                                        "input_type": input_type,
                                        "output_type": output_type,
                                        "hasbias": hasbias,
                                        "Activation": act_tag[1],
                                        "TileShape": tile_shape[0],
                                        "ClusterShape": tile_shape[1],
                                        "KernelSchedule": kernel_schedule,
                                        "EpilogueSchedule": epilogue_schedule,
                                        "TileSchedule": tile_schedule,
                                        "SM": sm,
                                        "sm": sm[-2:],
                                    }
                                    source_all_code += SubstituteTemplate(LaunchGemmPart1, value_dict)
                                    type_id += 1
                    source_all_code += LaunchGemmPart2
                    gemm_config_str = gemm_config_str.replace("<", "").replace(">", "")
                    code_map[gemm_config_str] = source_all_code
                    source_path = os.path.join(
                        generate_dir, f"launch_block_gemm_kernel_sm{sm[-2:]}_{gemm_config_str}.cu"
                    )
                    with open(source_path, "w") as f:
                        f.write(source_all_code)
                        f.close()

    return head_all_code, code_map


# generate fp8_fp8_gemm_scale_bias_act_sm90.cu
def generate_dispatch_gemm_cu(inputs_type: (str), outputs_type: (str), fuse_gemm_configs: tuple, sm: str):
    act_tags = [single_config[1] for single_config in fuse_gemm_configs]

    single_config = fuse_gemm_configs[0]
    hasbiases: (str) = single_config[0]
    tiles: (str) = single_config[2]
    KernelSchedule: (str) = single_config[3]
    EpilogueSchedule: (str) = single_config[4]
    TileSchedule: (str) = single_config[5]
    all_code = SubstituteTemplate(code_part0, {"sm": sm[-2:]})
    type_id = 0
    for input_type in inputs_type:
        for output_type in outputs_type:
            for act_tag in act_tags:
                for hasbias in hasbiases:
                    value_dict = {
                        "act_tag": act_tag[0],
                        "input_type": input_type,
                        "output_type": output_type,
                        "hasbias": hasbias,
                        "type_id": str(type_id),
                    }
                    all_code += SubstituteTemplate(code_part1, value_dict)
                    type_id += 1

    all_code += code_part2
    tile_id = 0
    for tile_shape in tiles:
        for kernel_schedule in KernelSchedule:
            for epilogue_schedule in EpilogueSchedule:
                for tile_schedule in TileSchedule:
                    if not check_config_valid(tile_shape, kernel_schedule, epilogue_schedule, tile_schedule):
                        continue
                    value_dict = {
                        "TileShape": tile_shape[0],
                        "ClusterShape": tile_shape[1],
                        "kernel_schedule": kernel_schedule,
                        "epilogue_schedule": epilogue_schedule,
                        "tile_schedule": tile_schedule,
                        "tile_id": str(tile_id),
                    }
                    all_code += SubstituteTemplate(code_part3, value_dict)
                    tile_id += 1
    all_code += SubstituteTemplate(code_part4, {"sm": sm[-2:]})
    tile_id = 0
    for tile_shape in tiles:
        blocks, clusters = get_shape_str(tile_shape)
        gemm_config_str_0 = f"tile{blocks[0]}x{blocks[1]}x{blocks[2]}_cluster{clusters[0]}x{clusters[1]}x{clusters[2]}"
        for kernel_schedule in KernelSchedule:
            gemm_config_str_1 = gemm_config_str_0 + f"_{kernel_schedule}"
            for epilogue_schedule in EpilogueSchedule:
                gemm_config_str_2 = gemm_config_str_1 + f"_{epilogue_schedule}"
                for tile_schedule in TileSchedule:
                    if not check_config_valid(tile_shape, kernel_schedule, epilogue_schedule, tile_schedule):
                        continue
                    gemm_config_str = gemm_config_str_2 + f"_{tile_schedule}"
                    value_dict = {
                        "sm": sm[-2:],
                        "tile_id": str(tile_id),
                        "gemm_config": gemm_config_str.replace("<", "").replace(">", ""),
                    }
                    all_code += SubstituteTemplate(code_part5, value_dict)
                    tile_id += 1

    all_code += SubstituteTemplate(code_part6, {"sm": sm[-2:]})
    return all_code


if __name__ == "__main__":
    args = parse_args()
    archs = args.cuda_arch
    inputs_type = (
        "float8_e4m3fn",
        # "float8_e5m2",
    )
    outputs_type = ("float16", "bfloat16")
    sm_dict = {"90": "cutlass::arch::Sm90"}

    for sm in archs:
        if sm == "90":
            fuse_gemm_configs = get_candidate_configs(sm)
            for fuse_gemm_config in fuse_gemm_configs:
                file_name = f"gpu/cutlass_kernels/fp8_gemm_fused/autogen/generic_block_gemm_kernel_sm{sm}_{fuse_gemm_config[1][0]}.cu"
                all_code = generate_source_cu(
                    inputs_type,
                    outputs_type,
                    fuse_gemm_config[0],
                    fuse_gemm_config[1],
                    fuse_gemm_config[2],
                    fuse_gemm_config[3],
                    fuse_gemm_config[4],
                    fuse_gemm_config[5],
                    sm_dict[sm],
                )
                file_dir = os.path.dirname(file_name)
                os.makedirs(file_dir, exist_ok=True)
                with open(file_name, "w") as f:
                    f.write(all_code)
                    f.close()
            # Compile parallelization
            generate_launch_gemm_cus(
                "gpu/cutlass_kernels/fp8_gemm_fused/autogen", inputs_type, outputs_type, fuse_gemm_configs, sm_dict[sm]
            )

            # hard code for act_tag
            file_name = f"gpu/cutlass_kernels/fp8_gemm_fused/autogen/fp8_fp8_block_gemm_scale_bias_act_sm{sm}.cu"
            all_code = generate_dispatch_gemm_cu(
                inputs_type,
                outputs_type,
                fuse_gemm_configs,
                sm_dict[sm],
            )
            file_dir = os.path.dirname(file_name)
            os.makedirs(file_dir, exist_ok=True)
            with open(file_name, "w") as f:
                f.write(all_code)
                f.close()
        else:
            raise ValueError(f"Unsupported SM: {sm}")
