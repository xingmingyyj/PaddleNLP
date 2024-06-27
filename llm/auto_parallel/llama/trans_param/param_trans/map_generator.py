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

import logging
import re

import paddle
from extractor import Extractor

level = logging.DEBUG
logger = logging.getLogger(__file__)
logger.setLevel(level)
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(level)
logger.addHandler(consoleHandler)


class MapGenerator:
    def __init__(
        self, dy_hand_log_path, dy_hand_model_state_path, dy_hand_optim_state_path, dy2st_auto_model_state_path
    ):
        self.dy_hand_log_path = dy_hand_log_path
        self.dy_hand_model_state_path = dy_hand_model_state_path
        self.dy_hand_optim_state_path = dy_hand_optim_state_path
        self.dy2st_auto_model_state_path = dy2st_auto_model_state_path
        self.extractor = Extractor(open(dy_hand_log_path, "r").read(), None)

    def get_state_info(self, state):
        state_info = {}
        if "master_weights" in state.keys():
            logger.debug("master weights")
            master_weights = state.pop("master_weights")
            for k, v in master_weights.items():
                state_info[k] = [v.shape, v.dtype]
        free_keys_pattern = ["LearningRate", "eager_", "LR_Scheduler"]
        need_pop_keys = []
        for free_key in free_keys_pattern:
            for key in state.keys():
                if re.search(free_key, key):
                    need_pop_keys.append(key)
        for key in need_pop_keys:
            state.pop(key)
        for k, v in state.items():
            state_info[k] = [v.shape, v.dtype]
        return state_info

    def sort_model_state_keys(self, model_state_keys, master_weights, sorted_master_weights_names):
        assert len(master_weights) == len(model_state_keys)
        assert len(sorted_master_weights_names) == len(master_weights)
        mapping = {}
        for i in range(len(master_weights)):
            mapping[master_weights[i]] = model_state_keys[i]
        sorted_model_state_keys = []
        for name in sorted_master_weights_names:
            sorted_model_state_keys.append(mapping[name])
        return sorted_model_state_keys

    def dy_generate_sorted_keys(self, model_state_keys, master_weights, sorted_master_weights_names):
        sorted_model_state_keys = self.sort_model_state_keys(
            model_state_keys, master_weights, sorted_master_weights_names
        )
        sorted_optim_states_keys = self.generator_sotrd_optim_states(sorted_master_weights_names)
        for k in sorted_optim_states_keys:
            sorted_model_state_keys.append(k)
        return sorted_model_state_keys

    def generator_sotrd_optim_states(self, sorted_master_weights_names):
        sorted_optim_states = []
        for name in sorted_master_weights_names:
            sorted_optim_states.append(name)
            sorted_optim_states.append(name + "_fp32_master_0_moment1_0")
            sorted_optim_states.append(name + "_fp32_master_0_moment2_0")
            sorted_optim_states.append(name + "_fp32_master_0_beta1_pow_acc_0")
            sorted_optim_states.append(name + "_fp32_master_0_beta2_pow_acc_0")
        return sorted_optim_states

    def dy2st_gen_sorted_keys(self, dy2st_auto_model_state_keys, sorted_master_weights_names):
        model_state_keys = dy2st_auto_model_state_keys[0 : len(sorted_master_weights_names)]
        optimizer_state_keys = []
        for k in model_state_keys:
            optimizer_state_keys.append(k + "_fp32_master_1")
            optimizer_state_keys.append(k + "_fp32_master_1_moment1_0")
            optimizer_state_keys.append(k + "_fp32_master_1_moment2_0")
            optimizer_state_keys.append(k + "_fp32_master_1_beta1_pow_acc_0")
            optimizer_state_keys.append(k + "_fp32_master_1_beta2_pow_acc_0")
        for k in optimizer_state_keys:
            model_state_keys.append(k)
        return model_state_keys

    def verify(self, dy_hand_to_dy2st_auto_keys_mapping, dy_state_info, dy2st_auto_model_state_infos):
        if dy_hand_to_dy2st_auto_keys_mapping.keys() != dy_state_info.keys():
            logger.debug("The resulting mapping is incomplete!")
            return False
        free_check_keys_pattern = ["embed"]
        for k, v in dy_state_info.items():
            if re.search(free_check_keys_pattern[0], k):
                continue
            logger.debug("verify {}:{}".format(k, v))
            logger.debug(
                "verify {}:{}".format(
                    dy_hand_to_dy2st_auto_keys_mapping[k],
                    dy2st_auto_model_state_infos[dy_hand_to_dy2st_auto_keys_mapping[k]],
                )
            )
            if v != dy2st_auto_model_state_infos[dy_hand_to_dy2st_auto_keys_mapping[k]]:
                logger.debug("The mapping is not correct!")
                return False
        return True

    def get_map(self, max_step):
        dy_model_state = paddle.load(self.dy_hand_model_state_path)
        dy_model_state_keys = list(dy_model_state.keys())
        dy_model_state_infos = self.get_state_info(dy_model_state)
        del dy_model_state
        dy_optim_state = paddle.load(self.dy_hand_optim_state_path)
        master_weights = list(dy_optim_state["master_weights"].keys())
        dy_optim_state_info = self.get_state_info(dy_optim_state)
        del dy_optim_state
        dy_model_state_infos.update(dy_optim_state_info)
        dy_state_info = dy_model_state_infos

        logger.debug("Dynamic hand parallel param infos")
        logger.debug("Model state keys")
        logger.debug(dy_model_state_keys)
        logger.debug(
            "Expect to parse the dynamic graph execution log to get the order in which these tensors will be used during the dynamic graph execution！"
        )
        logger.debug(master_weights)

        dy2st_auto_model_state = paddle.load(self.dy2st_auto_model_state_path)
        dy2st_auto_model_state_infos = self.get_state_info(dy2st_auto_model_state)
        dy2st_auto_model_state_keys = list(dy2st_auto_model_state_infos.keys())
        del dy2st_auto_model_state

        logger.debug("Dy2st auto parallel param infos")
        logger.debug("Model state keys")
        logger.debug(dy2st_auto_model_state_infos.keys())

        self.extractor.tensor_names = master_weights
        sorted_master_weights_names = self.extractor.get_sorted_tensor_names()

        logger.debug("Try to get dy_hand_to_dy2st_auto_keys_mapping")

        for step in range(max_step):
            logger.debug("Step " + str(step))
            logger.debug("Executed order of master_weights_names")
            logger.debug(sorted_master_weights_names)

            dy_sorted_keys = self.dy_generate_sorted_keys(
                dy_model_state_keys, master_weights, sorted_master_weights_names
            )
            logger.debug("Dy_sorted_keys")
            for k in dy_sorted_keys:
                logger.debug(k)

            dy2st_sorted_keys = self.dy2st_gen_sorted_keys(dy2st_auto_model_state_keys, sorted_master_weights_names)

            if set(dy_sorted_keys) != set(dy_state_info.keys()) or set(dy2st_sorted_keys) != set(
                dy2st_auto_model_state_keys
            ):
                logger.debug("The execution sequence of log parsing is incomplete. You need to try again！")
                self.extractor.filted_running_api_infos.pop(0)
                sorted_master_weights_names = self.extractor.get_sorted_tensor_names()
                continue

            dy_hand_to_dy2st_auto_keys_mapping = {}

            for i in range(len(dy_sorted_keys)):
                dy_hand_to_dy2st_auto_keys_mapping[dy_sorted_keys[i]] = dy2st_sorted_keys[i]

            logger.debug("Dy hand to dy2st auto param mapping")

            for k, v in dy_hand_to_dy2st_auto_keys_mapping.items():
                logger.debug(k + " , " + v)

            logger.debug("Verify the mapping is correct or not")
            if self.verify(dy_hand_to_dy2st_auto_keys_mapping, dy_state_info, dy2st_auto_model_state_infos):
                logger.debug("Mapping is correct")
                break
            else:
                logger.debug("Mapping is incorrect")
                self.extractor.filted_running_api_infos.pop(0)
                sorted_master_weights_names = self.extractor.get_sorted_tensor_names()
                assert step != max_step - 1, (
                    "Tried " + str(max_step) + " several times, but still can't find a suitable mapping!"
                )
        logger.debug("Find a reasonable mapping")
        return dy_hand_to_dy2st_auto_keys_mapping
