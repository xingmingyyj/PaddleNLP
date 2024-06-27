# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

from map_generator import MapGenerator
from param_translator import ParamTranslator

if __name__ == "__main__":
    dy_hand_log_path = "/root/paddlejob/workspace/xingmingyyj/PaddleNLP/llm/llama/auto_parallel/dy_log_1"
    dy_hand_log_name = [
        "workerlog.0",
        "workerlog.1",
        "workerlog.2",
        "workerlog.3",
        "workerlog.4",
        "workerlog.5",
        "workerlog.6",
        "workerlog.7",
    ]
    dy_hand_checkpoint_path = "/root/paddlejob/workspace/xingmingyyj/PaddleNLP/llm/llama/auto_parallel/checkpoints/llama2_pretrain_ckpts/checkpoint-2"
    dy_hand_model_state_name = [
        "model_state.tp00_pp00.pdparams",
        "model_state.tp01_pp00.pdparams",
        "model_state.tp00_pp01.pdparams",
        "model_state.tp01_pp01.pdparams",
        "model_state.tp00_pp02.pdparams",
        "model_state.tp01_pp02.pdparams",
        "model_state.tp00_pp03.pdparams",
        "model_state.tp01_pp03.pdparams",
    ]
    dy_hand_optim_state_name = [
        "optimizer.tp00_pp00.pdopt",
        "optimizer.tp01_pp00.pdopt",
        "optimizer.tp00_pp01.pdopt",
        "optimizer.tp01_pp01.pdopt",
        "optimizer.tp00_pp02.pdopt",
        "optimizer.tp01_pp02.pdopt",
        "optimizer.tp00_pp03.pdopt",
        "optimizer.tp01_pp03.pdopt",
    ]

    dy2st_checkpoint_path = "/root/paddlejob/workspace/xingmingyyj/PaddleNLP/llm/llama/auto_parallel/output/dy2st_log2/checkpoint-2/dist_ckpt"
    dy2st_model_state_name = [
        "0_0.model.param",
        "1_0.model.param",
        "2_0.model.param",
        "3_0.model.param",
        "4_0.model.param",
        "5_0.model.param",
        "6_0.model.param",
        "7_0.model.param",
    ]

    rank_name_maps = []
    for rank in range(8):
        print("===gen rank {} name_map ===".format(rank))
        joined_dy_hand_log_path = os.path.join(dy_hand_log_path, dy_hand_log_name[rank])
        joined_dy_hand_model_state_path = os.path.join(dy_hand_checkpoint_path, dy_hand_model_state_name[rank])
        joined_dy_hand_optim_state_path = os.path.join(dy_hand_checkpoint_path, dy_hand_optim_state_name[rank])
        joined_dy2st_auto_model_state_path = os.path.join(dy2st_checkpoint_path, dy2st_model_state_name[rank])
        rank_name_maps.append(
            MapGenerator(
                joined_dy_hand_log_path,
                joined_dy_hand_model_state_path,
                joined_dy_hand_optim_state_path,
                joined_dy2st_auto_model_state_path,
            ).get_map(5)
        )

    need_repartition_tensor_names = [
        "llama.embed_tokens.weight",
        "embedding_0.w_0",
        "embedding_0.w_0_fp32_master_0_moment1_0",
        "embedding_0.w_0_fp32_master_0_moment2_0",
    ]
    need_repartition_files = [
        {
            "model": os.path.join(dy_hand_checkpoint_path, dy_hand_model_state_name[0]),
            "optim": os.path.join(dy_hand_checkpoint_path, dy_hand_optim_state_name[0]),
        },
        {
            "model": os.path.join(dy_hand_checkpoint_path, dy_hand_model_state_name[1]),
            "optim": os.path.join(dy_hand_checkpoint_path, dy_hand_optim_state_name[1]),
        },
    ]

    print("==========translate param=================")

    saved_checkpoint_path = "/root/paddlejob/workspace/xingmingyyj/PaddleNLP/llm/llama/auto_parallel/output/dy2st_log2/checkpoint_trans/dist_ckpt"

    ParamTranslator(
        rank_name_maps,
        need_repartition_tensor_names,
        need_repartition_files,
        model_state_dict_file_names=dy_hand_model_state_name,
        optim_state_dict_file_names=dy_hand_optim_state_name,
        dy_hand_checkpoint_path=dy_hand_checkpoint_path,
        saved_checkpoint_path=saved_checkpoint_path,
        save_file_names=dy2st_model_state_name,
    ).do_translate()
