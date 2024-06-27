// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved. */

/*This code is copied fron NVIDIA apex:
 *     https://github.com/NVIDIA/apex
 *     with minor changes. */

#include <cassert>
#include <vector>

#include "layer_norm_cuda.h"  // NOLINT
#include "paddle/extension.h"

#ifdef CUSTOM_OP_WITH_SPMD
#include "paddle/phi/api/ext/spmd_infer.h"
#include "paddle/phi/infermeta/spmd_rules/rules.h"
#endif

#define CHECK_CUDA(x) PD_CHECK(!x.is_cpu(), #x " must be a CUDA tensor")

static void GetRowsCols(const std::vector<int64_t> &shape,
                        int *p_rows,
                        int *p_cols) {
  int rows = 1;
  for (int i = 0; i + 1 < shape.size(); ++i) {
    rows *= shape[i];
  }
  int cols = shape[shape.size() - 1];
  *p_rows = rows;
  *p_cols = cols;
}

/**
 * Print Input Output (level 0 means least info, level 2 means most info)
 * **/
std::string TensorStr(const paddle::Tensor& t) {
  std::string tensor_name_str = "";
  if (t.name() == "") {
    tensor_name_str = "None";
  } else {
    tensor_name_str = t.name();
  }
  const char* TENSOR_INFO_TEMPLATE =
      "Type: %s, Dtype: %s, Place: %s, Shape: %s, DistAttr: %s";
  std::string tensor_info_str = "";
  if (t.defined()) {
      if (t.initialized()) {
        tensor_info_str += paddle::string::Sprintf(TENSOR_INFO_TEMPLATE,
                                                   t.impl()->type_info().name(),
                                                   t.dtype(),
                                                   t.place().DebugString(),
                                                   t.dims(),
                                                   "Unknown");
      } else {
        tensor_info_str += paddle::string::Sprintf(TENSOR_INFO_TEMPLATE,
                                                   t.impl()->type_info().name(),
                                                   "Unknown",
                                                   "Unknown",
                                                   "Unknown",
                                                   "Unknown");
      }
  } else {
    tensor_info_str += "Unknown";
  }
    const char* TENSOR_PRINT_TEMPLATE =
        "{Name: %s, Initialized: %d, Ptr: %d,"
        "TensorInfo: [ %s ], ADInfo:[ %s ]}";
      return paddle::string::Sprintf(TENSOR_PRINT_TEMPLATE,
                                     tensor_name_str,
                                     t.initialized(),
                                     t.impl(),
                                     tensor_info_str,
                                     "Unknown");
}



std::vector<paddle::Tensor> RMSLnFwd(const paddle::Tensor &x,
                                     const paddle::Tensor &scale,
                                     float epsilon) {
  const auto &scale_shape = scale.shape();
  const auto &x_shape = x.shape();
  PD_CHECK(scale_shape.size() == 1);
  PD_CHECK(scale_shape[0] == x_shape[x_shape.size() - 1]);
  CHECK_CUDA(x);
  CHECK_CUDA(scale);

  int rows, cols;
  GetRowsCols(x_shape, &rows, &cols);

  auto place = x.place();
  auto y = paddle::empty(x_shape, scale.type(), place);
  auto variance_shape = x_shape;
  variance_shape.pop_back();
  auto invvar = paddle::empty(variance_shape, paddle::DataType::FLOAT32, place);
  cuda_rms_norm(x, scale, rows, cols, epsilon, &y, &invvar);

  std::cout << "Finish AD API: " << "external_rms_norm"<<std::endl;
 // Before log info
    const char* INPUT_PRINT_TEMPLATE = "{ Input: [%s],  \n Output: [%s] } ";
    std::string input_str = "";
    std::string output_str = "";
    const char* TENSOR_X_TEMPLATE = " \n( x , [%s]), ";
    std::string input_x_str = paddle::string::Sprintf(
        TENSOR_X_TEMPLATE, TensorStr(x));
    input_str += input_x_str;
    const char* TENSOR_SCALE_TEMPLATE = " \n( scale , [%s]), ";
    std::string input_scale_str = paddle::string::Sprintf(
        TENSOR_SCALE_TEMPLATE, TensorStr(scale));
    input_str += input_scale_str;
   
    const char* TENSOR_Y_TEMPLATE = " \n( y , [%s]), ";
    std::string output_y_str = paddle::string::Sprintf(
        TENSOR_Y_TEMPLATE, TensorStr(y));
    output_str += output_y_str;
    const char* TENSOR_INVVAR_TEMPLATE = " \n( invvar , [%s]), ";
    std::string output_invvar_str = paddle::string::Sprintf(
        TENSOR_INVVAR_TEMPLATE, TensorStr(invvar));
    output_str += output_y_str;
    std::cout << paddle::string::Sprintf(
        INPUT_PRINT_TEMPLATE, input_str, output_str);
  
  return {y, invvar};
}


std::vector<paddle::Tensor> LnFwd(const paddle::Tensor &x,
                                  const paddle::Tensor &scale,
                                  const paddle::Tensor &bias,
                                  float epsilon) {
  const auto &scale_shape = scale.shape();
  const auto &bias_shape = bias.shape();
  const auto &x_shape = x.shape();
  PD_CHECK(scale_shape == bias_shape);
  PD_CHECK(scale_shape.size() == 1);
  PD_CHECK(scale_shape[0] == x_shape[x_shape.size() - 1]);
  CHECK_CUDA(x);
  CHECK_CUDA(scale);
  CHECK_CUDA(bias);

  int rows, cols;
  GetRowsCols(x_shape, &rows, &cols);

  auto place = x.place();
  auto y = paddle::empty(x_shape, scale.type(), place);
  auto mean = paddle::empty({rows}, paddle::DataType::FLOAT32, place);
  auto invvar = paddle::empty_like(mean);

  cuda_layer_norm(x, scale, bias, rows, cols, epsilon, &y, &mean, &invvar);
  return {y, mean, invvar};
}

std::vector<std::vector<int64_t>> LnFwdInferShape(
    std::vector<int64_t> x_shape,
    std::vector<int64_t> scale_shape,
    std::vector<int64_t> bias_shape,
    float epsilon) {
  int rows, cols;
  GetRowsCols(x_shape, &rows, &cols);
  return {x_shape, {rows}, {rows}};
}

std::vector<std::vector<int64_t>> RMSLnFwdInferShape(
    std::vector<int64_t> x_shape,
    std::vector<int64_t> scale_shape,
    float epsilon) {
  auto variance_shape = x_shape;
  variance_shape.pop_back();
  return {x_shape, variance_shape};
}

std::vector<paddle::DataType> LnFwdInferDtype(paddle::DataType x_dtype,
                                              paddle::DataType scale_dtype,
                                              paddle::DataType bias_dtype) {
  return {x_dtype, paddle::DataType::FLOAT32, paddle::DataType::FLOAT32};
}

std::vector<paddle::DataType> RMSLnFwdInferDtype(paddle::DataType x_dtype,
                                                 paddle::DataType scale_dtype) {
  return {x_dtype, paddle::DataType::FLOAT32};
}

std::vector<paddle::Tensor> LnBwd(const paddle::Tensor &x,
                                  const paddle::Tensor &scale,
                                  const paddle::Tensor &bias,
                                  const paddle::Tensor &mean,
                                  const paddle::Tensor &invvar,
                                  const paddle::Tensor &dy,
                                  float epsilon) {
  CHECK_CUDA(dy);
  CHECK_CUDA(mean);
  CHECK_CUDA(invvar);
  CHECK_CUDA(x);
  CHECK_CUDA(scale);
  CHECK_CUDA(bias);

  int rows, cols;
  GetRowsCols(x.shape(), &rows, &cols);

  auto grad_x = paddle::empty_like(x);
  auto grad_scale = paddle::empty_like(scale);
  auto grad_bias = paddle::empty_like(bias);

  cuda_layer_norm_gradient(x,
                           scale,
                           bias,
                           mean,
                           invvar,
                           dy,
                           rows,
                           cols,
                           epsilon,
                           &grad_x,
                           &grad_scale,
                           &grad_bias);
  return {grad_x, grad_scale, grad_bias};
}

std::vector<paddle::Tensor> RMSLnBwd(const paddle::Tensor &x,
                                     const paddle::Tensor &scale,
                                     const paddle::Tensor &invvar,
                                     const paddle::Tensor &dy,
                                     float epsilon) {
  CHECK_CUDA(dy);
  CHECK_CUDA(invvar);
  CHECK_CUDA(x);
  CHECK_CUDA(scale);

  int rows, cols;
  GetRowsCols(x.shape(), &rows, &cols);

  auto grad_x = paddle::empty_like(x);
  auto grad_scale = paddle::empty_like(scale);

  cuda_rms_norm_gradient(
      x, scale, invvar, dy, rows, cols, epsilon, &grad_x, &grad_scale);
  return {grad_x, grad_scale};
}

std::vector<std::vector<int64_t>> LnBwdInferShape(
    std::vector<int64_t> input_shape,
    std::vector<int64_t> gamma_shape,
    std::vector<int64_t> beta_shape,
    std::vector<int64_t> mean_shape,
    std::vector<int64_t> invvar_shape,
    std::vector<int64_t> dout_shape,
    float epsilon) {
  return {input_shape, gamma_shape, beta_shape};
}

std::vector<std::vector<int64_t>> RMSLnBwdInferShape(
    std::vector<int64_t> input_shape,
    std::vector<int64_t> gamma_shape,
    std::vector<int64_t> invvar_shape,
    std::vector<int64_t> dout_shape,
    float epsilon) {
  return {input_shape, gamma_shape};
}


PD_BUILD_OP(fused_ln)
    .Inputs({"x", "scale", "bias"})
    .Outputs({"y", "mean", "invvar"})
    .Attrs({"epsilon: float"})
    .SetKernelFn(PD_KERNEL(LnFwd))
    .SetInferShapeFn(PD_INFER_SHAPE(LnFwdInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(LnFwdInferDtype));

PD_BUILD_GRAD_OP(fused_ln)
    .Inputs({"x", "scale", "bias", "mean", "invvar", paddle::Grad("y")})
    .Outputs({paddle::Grad("x"), paddle::Grad("scale"), paddle::Grad("bias")})
    .Attrs({"epsilon: float"})
    .SetKernelFn(PD_KERNEL(LnBwd))
    .SetInferShapeFn(PD_INFER_SHAPE(LnBwdInferShape));

PD_BUILD_OP(fused_rms_norm)
    .Inputs({"x", "scale"})
    .Outputs({"y", "invvar"})
    .Attrs({"epsilon: float"})
    .SetKernelFn(PD_KERNEL(RMSLnFwd))
    .SetInferShapeFn(PD_INFER_SHAPE(RMSLnFwdInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RMSLnFwdInferDtype))
#ifdef CUSTOM_OP_WITH_SPMD
    .SetInferSpmdFn(PD_INFER_SPMD_RULE(phi::distributed::RmsNormInferSpmd))
#endif
    ;

PD_BUILD_GRAD_OP(fused_rms_norm)
    .Inputs({"x", "scale", "invvar", paddle::Grad("y")})
    .Outputs({paddle::Grad("x"), paddle::Grad("scale")})
    .Attrs({"epsilon: float"})
    .SetKernelFn(PD_KERNEL(RMSLnBwd))
    .SetInferShapeFn(PD_INFER_SHAPE(RMSLnBwdInferShape))
#ifdef CUSTOM_OP_WITH_SPMD
    .SetInferSpmdFn(PD_INFER_SPMD_RULE(phi::distributed::RmsNormGradInferSpmd))
#endif
    ;


// https://github.com/NVIDIA/apex/blob/85e9eddece9d4ac72b48c2407f8162f2173e1bf4/csrc/layer_norm_cuda_kernel.cu#L679
