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

import re


class RunningAPIInfo:
    def __init__(self, api_name, inputs, outputs):
        self.api_name = api_name
        self.inputs = inputs
        self.outputs = outputs

    def __str__(self):
        inputs_str = ""
        for input in self.inputs:
            inputs_str += str(input) + ", "
        outputs_str = ""
        for output in self.outputs:
            outputs_str += str(output) + ", "
        input_str = "{ " + inputs_str + " }"
        output_str = "{ " + outputs_str + " }"
        return f"API Name: {self.api_name}, Inputs: {input_str}, Outputs: {output_str}"


class TensorInfo:
    def __init__(self, name, ptr):
        self.name = name
        self.ptr = ptr

    def __str__(self):
        return f"TensorInfo(name={self.name}, ptr={self.ptr})"


class Extractor:
    def __init__(self, log, tensor_names):
        self.log = log
        self.tensor_names = tensor_names
        self.api_name_pattern = re.compile("Finish AD API: (\w+) ")
        self.filted_running_api_infos = None

    def cut_inputs_str(self, running_api_info, begin_index):
        step = 1
        count = 1
        while count != 0 and begin_index + step < len(running_api_info) - 1:
            if running_api_info[begin_index + step] == "{":
                count += 1
            elif running_api_info[begin_index + step] == "}":
                count -= 1
            step += 1
        return running_api_info[begin_index + len("{") : begin_index + step - len("}")].strip(), begin_index + step

    def cut_tensor_str(self, tensor_str):
        if "Output: [" in tensor_str:
            inputs_str = tensor_str[0 : tensor_str.find("Output: [") - len(", ")]
            outputs_str = tensor_str[tensor_str.find("Output: [") :]
        else:
            inputs_str = tensor_str
            outputs_str = ""
        input_tensor_strs = []
        begin_index = inputs_str.find("(")
        step = 1
        count = 1
        while begin_index != -1 and begin_index + step < len(inputs_str) - 1:
            if inputs_str[begin_index + step] == "(":
                count += 1
            elif inputs_str[begin_index + step] == ")":
                count -= 1
            step += 1
            if count == 0:
                input_tensor_strs.append(inputs_str[begin_index : begin_index + step].strip())
                begin_index = inputs_str.find("(", begin_index + step)
                if begin_index == -1:
                    break
                else:
                    step = 1
                    count = 1
        output_tensor_strs = []
        begin_index = outputs_str.find("(")
        step = 1
        count = 1
        while begin_index != -1 and begin_index + step < len(outputs_str) - 1:
            if outputs_str[begin_index + step] == "(":
                count += 1
            elif outputs_str[begin_index + step] == ")":
                count -= 1
            step += 1
            if count == 0:
                output_tensor_strs.append(outputs_str[begin_index : begin_index + step].strip())
                begin_index = outputs_str.find("(", begin_index + step)
                if begin_index == -1:
                    break
                else:
                    step = 1
                    count = 1
        return input_tensor_strs, output_tensor_strs

    def parse_tensor_info(self, tensor_strs):
        tensor_infos = []
        for ts in tensor_strs:
            if ts.find("TensorInfo") == -1:
                tensor_infos.append(TensorInfo(None, None))
                continue
            ts = ts[0 : ts.find("TensorInfo")]
            name_pattern = re.compile("\{Name: (\w+(\.\w+)*),")
            ptr_pattern = re.compile("Ptr: (\w+),")
            name_match = re.search(name_pattern, ts)
            assert name_match is not None
            ptr_match = re.search(ptr_pattern, ts)
            assert ptr_match is not None
            tensor_infos.append(TensorInfo(name_match.group(1), ptr_match.group(1)))
        return tensor_infos

    def parse_running_api_info(self):
        KEYWORD = "Finish AD API:"
        MAX_API_NAME_LENGTH = 100

        running_api_infos = []
        split_log = []
        log = self.log.replace("\n", " ")
        posistion = 0
        index = log.find(KEYWORD, posistion)
        while index != -1:
            api_name_str = log[index : index + len(KEYWORD) + MAX_API_NAME_LENGTH]
            api_name_match = re.search(self.api_name_pattern, api_name_str)
            if api_name_match:
                api_name = api_name_match.group(1)
            else:
                raise Exception("Can not find api name")
            begin_index = log.find("{ Input", index)
            inputs_str, end_index = self.cut_inputs_str(log, begin_index)
            split_log.append((api_name, inputs_str))
            posistion = end_index + 1
            index = log.find(KEYWORD, posistion)
        for i in split_log:
            input_tensor_strs, output_tensor_strs = self.cut_tensor_str(i[1])
            input_tensor_infos = self.parse_tensor_info(input_tensor_strs)
            output_tensor_infos = self.parse_tensor_info(output_tensor_strs)
            running_api_infos.append(RunningAPIInfo(i[0], input_tensor_infos, output_tensor_infos))

        return running_api_infos

    def rename(self, running_api_info):
        ptr_name_dict = {}
        for api in running_api_info:
            for input_tensor in api.inputs:
                if input_tensor.ptr is not None and input_tensor.name != "None":
                    ptr_name_dict[input_tensor.ptr] = input_tensor.name
            for output_tensor in api.outputs:
                if output_tensor.ptr is not None and output_tensor.name != "None":
                    ptr_name_dict[output_tensor.ptr] = output_tensor.name
        for api in running_api_info:
            for input_tensor in api.inputs:
                if input_tensor.ptr is not None and input_tensor.name == "None":
                    if input_tensor.ptr in ptr_name_dict:
                        input_tensor.name = ptr_name_dict[input_tensor.ptr]
            for output_tensor in api.outputs:
                if output_tensor.ptr is not None and output_tensor.name == "None":
                    if output_tensor.ptr in ptr_name_dict:
                        output_tensor.name = ptr_name_dict[output_tensor.ptr]
        return running_api_info

    def filter(self, running_api_info):
        def is_computational_api(api):
            computational_api = False
            for input_tensor in api.inputs:
                if input_tensor.name in self.tensor_names:
                    computational_api = True
            for output_tensor in api.outputs:
                if output_tensor.name in self.tensor_names or "eager_tmp" in output_tensor.name:
                    computational_api = False
            return computational_api

        return list(filter(is_computational_api, running_api_info))

    def get_sorted_tensor_names(self):
        sorted_tensor_names = []
        if self.filted_running_api_infos is None:
            running_api_infos = self.parse_running_api_info()
            renamed_running_api_infos = self.rename(running_api_infos)
            self.filted_running_api_infos = self.filter(renamed_running_api_infos)

        for api in self.filted_running_api_infos:
            for input_tensor in api.inputs:
                if input_tensor.name not in sorted_tensor_names and input_tensor.name in self.tensor_names:
                    sorted_tensor_names.append(input_tensor.name)

        assert len(sorted_tensor_names) == len(self.tensor_names)
        return sorted_tensor_names
