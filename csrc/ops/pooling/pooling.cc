/*
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "csrc/ops/pooling/pooling.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "ATen/core/ATen_fwd.h"
#include "absl/log/absl_check.h"
#include "absl/status/statusor.h"
#include "csrc/common/dimension_types.h"
#include "csrc/common/error_utils.h"
#include "csrc/ops/op_builder_utils.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/integrations/cpp/builder/AttrTypeBuilderUtil.h"
#include "stablehlo/integrations/cpp/builder/MlirBuilder.h"
#include "stablehlo/integrations/cpp/builder/StablehloBuilder.h"

namespace torch_tpu {

mlir::Attribute GetMaxPoolInitValueAttr(mlir::Type element_type,
                                        mlir::OpBuilder& builder) {
  if (const auto float_type = llvm::dyn_cast<mlir::FloatType>(element_type)) {
    return mlir::FloatAttr::get(
        element_type, llvm::APFloat::getInf(float_type.getFloatSemantics(),
                                            /*Negative=*/true));
  }
  return GetMinFiniteValueAttr(element_type, builder);
}

// Adds a batch dimension of size 1 if the input tensor doesn't have it.
// Returns the batch input and the original dimension size.
absl::StatusOr<BatchInput> CreateBatchInput(mlir::MlirOp input,
                                            const int64_t spatial_dim_count) {
  const mlir::RankedTensorType input_type = GetTensorTypeOrDie(input);
  auto num_dims = input_type.getRank();

  TT_RET_CHECK(  // ERROR_COV_INFEASIBLE=Rank is already checked by
                 // GetPoolingOutputSize.
      num_dims == spatial_dim_count + 1 || num_dims == spatial_dim_count + 2,
      error::kInvalidArgument)
      << "expected non-empty " << spatial_dim_count + 1 << "D or "
      << spatial_dim_count + 2 << "D (batch mode) tensor for input, got "
      << num_dims << "D tensor";

  // If the input doesn't have a batch dimension, add a size 1 batch dimension
  if (num_dims == spatial_dim_count + 1) {
    Dimensions new_shape = {1};
    new_shape.insert(new_shape.end(), input_type.getShape().begin(),
                     input_type.getShape().end());
    auto batch_input = mlir::stablehlo::Reshape(input, new_shape);
    return BatchInput{batch_input, num_dims};
  }
  return BatchInput{input, num_dims};
}

// Expands the attribute to match the dim size.
absl::StatusOr<Dimensions> ExpandAttribute(const Dimensions& values,
                                           const int64_t dim) {
  if (values.size() == 1 && dim > 1) {
    return Dimensions(dim, values[0]);
  }

  Dimensions expanded_values = values;
  ABSL_CHECK_EQ(values.size(), dim)  // CRASH_OK
      << "the attribute was not expanded correctly to match the dim size"
      << ", got expanded values size " << expanded_values.size()
      << " and expected dim size " << dim;
  return expanded_values;
}

// Removes the batch dimension if it doesn't have it at the beginning.
mlir::MlirOp RemoveTrivialBatch(mlir::MlirOp batch_op,
                                int64_t original_dim_size,
                                int64_t spatial_dim_count) {
  if (original_dim_size == spatial_dim_count + 2) {
    return batch_op;
  }

  const mlir::RankedTensorType current_type = GetTensorTypeOrDie(batch_op);
  llvm::SmallVector<int64_t> squeezed_shape;
  squeezed_shape.append(current_type.getShape().begin() + 1,
                        current_type.getShape().end());
  return mlir::stablehlo::Reshape(batch_op, squeezed_shape);
}

namespace {

int64_t ComputeExtraCeilModePadding(const int64_t input_size,
                                    const int64_t kernel_size,
                                    const int64_t stride, const int64_t padding,
                                    const int64_t dilation) {
  if (stride <= 1) {
    return 0;
  }
  const int64_t k_eff = (kernel_size - 1) * dilation + 1;
  const int64_t output_size_num = input_size + 2 * padding - k_eff;
  int64_t output_size_rem = output_size_num % stride;
  if (output_size_rem < 0) {
    output_size_rem += stride;
  }
  if (output_size_rem == 0) {
    return 0;
  }
  const int64_t extra_padding = stride - output_size_rem;
  const int64_t target_output_size =
      (output_size_num + extra_padding) / stride + 1;
  const int64_t last_window_start_index = (target_output_size - 1) * stride;
  if (last_window_start_index < padding + input_size) {
    return extra_padding;
  }
  return 0;
}

// Computes the scalar i32 runtime extra right padding for `ceil_mode` on a
// dynamic spatial dimension of size `input_size_i32`, mirroring
// `ComputeExtraCeilModePadding`.
mlir::MlirOp ComputeDynamicExtraCeilModePadding(mlir::MlirOp input_size_i32,
                                                const int64_t kernel_size,
                                                const int64_t stride,
                                                const int64_t padding,
                                                const int64_t dilation) {
  mlir::MlirBuilder& builder = input_size_i32.getBuilder();
  const mlir::Type i32_type = builder.getOpBuilder().getI32Type();
  const int64_t k_eff = (kernel_size - 1) * dilation + 1;

  mlir::MlirOp zero = MakeScalarConstant(builder, 0, i32_type);
  mlir::MlirOp one = MakeScalarConstant(builder, 1, i32_type);
  mlir::MlirOp stride_op = MakeScalarConstant(builder, stride, i32_type);
  mlir::MlirOp pad_op = MakeScalarConstant(builder, padding, i32_type);
  mlir::MlirOp offset_op =
      MakeScalarConstant(builder, 2 * padding - k_eff, i32_type);

  mlir::MlirOp num = mlir::stablehlo::Add(input_size_i32, offset_op);
  mlir::MlirOp rem = mlir::stablehlo::Rem(num, stride_op);
  mlir::MlirOp rem_neg = mlir::stablehlo::Compare(
      rem, zero, mlir::stablehlo::ComparisonDirection::LT);
  mlir::MlirOp rem_plus_stride = mlir::stablehlo::Add(rem, stride_op);
  rem = mlir::stablehlo::Select(rem_neg, rem_plus_stride, rem);

  mlir::MlirOp rem_nz = mlir::stablehlo::Compare(
      rem, zero, mlir::stablehlo::ComparisonDirection::NE);
  mlir::MlirOp stride_minus_rem = mlir::stablehlo::Subtract(stride_op, rem);
  mlir::MlirOp cand_extra =
      mlir::stablehlo::Select(rem_nz, stride_minus_rem, zero);

  mlir::MlirOp padded_num = mlir::stablehlo::Add(num, cand_extra);
  mlir::MlirOp quot = mlir::stablehlo::Div(padded_num, stride_op);
  mlir::MlirOp target_out = mlir::stablehlo::Add(quot, one);
  mlir::MlirOp last_idx = mlir::stablehlo::Subtract(target_out, one);
  mlir::MlirOp last_start = mlir::stablehlo::Mul(last_idx, stride_op);

  mlir::MlirOp in_limit = mlir::stablehlo::Add(input_size_i32, pad_op);
  mlir::MlirOp in_image = mlir::stablehlo::Compare(
      last_start, in_limit, mlir::stablehlo::ComparisonDirection::LT);
  mlir::MlirOp use_extra = mlir::stablehlo::And(rem_nz, in_image);
  return mlir::stablehlo::Select(use_extra, cand_extra, zero);
}

// Lifts bounded dynamic dimensions on `operand` to their static upper bounds
// and records each dimension's original runtime size in `runtime_sizes`.
void LiftDynamicDimsToStaticBounds(
    mlir::MlirOp& operand,
    std::vector<std::optional<mlir::MlirOp>>& runtime_sizes) {
  mlir::MlirBuilder& builder = operand.getBuilder();
  const mlir::Type i32_type = builder.getOpBuilder().getI32Type();
  const mlir::RankedTensorType operand_type = GetTensorTypeOrDie(operand);
  const Dimensions operand_bounds = GetStaticShape(operand);
  const int64_t rank = operand_type.getRank();
  for (int64_t dim = 0; dim < rank; ++dim) {
    if (!operand_type.isDynamicDim(dim)) {
      continue;
    }
    runtime_sizes[dim] = mlir::stablehlo::GetDimensionSize(operand, dim);
    mlir::MlirOp bound_op =
        MakeScalarConstant(builder, operand_bounds[dim], i32_type);
    operand = mlir::stablehlo::SetDimensionSize(operand, bound_op, dim);
  }
}

// Appends `iota(dim) < cur_size` onto `in_bounds_mask` for `static_i32_type`.
void AppendDimInBoundsMask(mlir::MlirBuilder& builder,
                           const mlir::RankedTensorType& static_i32_type,
                           const int64_t dim, mlir::MlirOp cur_size,
                           std::optional<mlir::MlirOp>& in_bounds_mask) {
  mlir::MlirOp idx = mlir::stablehlo::Iota(builder, static_i32_type, dim);
  mlir::MlirOp size_bcast = mlir::stablehlo::BroadcastInDim(
      static_i32_type, cur_size, /*broadcast_dimensions=*/{});
  mlir::MlirOp dim_in_bounds = mlir::stablehlo::Compare(
      idx, size_bcast, mlir::stablehlo::ComparisonDirection::LT);
  if (in_bounds_mask.has_value()) {
    mlir::MlirOp prev_mask = *in_bounds_mask;
    in_bounds_mask = mlir::stablehlo::And(prev_mask, dim_in_bounds);
  } else {
    in_bounds_mask = dim_in_bounds;
  }
}

// Restores bounded dynamic dimension sizes recorded in `runtime_sizes`.
mlir::MlirOp RestoreDynamicDimSizes(
    mlir::MlirOp operand,
    const std::vector<std::optional<mlir::MlirOp>>& runtime_sizes) {
  const int64_t rank = static_cast<int64_t>(runtime_sizes.size());
  for (int64_t dim = 0; dim < rank; ++dim) {
    if (runtime_sizes[dim].has_value()) {
      mlir::MlirOp new_size = *runtime_sizes[dim];
      operand = mlir::stablehlo::SetDimensionSize(operand, new_size, dim);
    }
  }
  return operand;
}

}  // namespace

// Calculates the adjusted (left_padding, right_padding) pairs given the padding
// (which is symmetric) and ceil mode.
// Additional high padding could be required when ceil mode is set.
// This ensures that the last pooling is fully inside the image.
std::vector<PaddingParams> CeilModePadding(
    const mlir::RankedTensorType& input_shape,  // (N, C, H, W)/(N, C, D, H, W)
    const mlir::ArrayRef<int64_t> kernel_size,  // [K_h, K_w]
    const mlir::ArrayRef<int64_t> stride,       // [S_h, S_w]
    const mlir::ArrayRef<int64_t> padding,      // [P_h, P_w]
    const mlir::ArrayRef<int64_t> dilation,     // [D_h, D_w]
    const bool ceil_mode) {
  std::vector<PaddingParams> ceil_mode_padding;
  ceil_mode_padding.reserve(padding.size());

  for (size_t i = 0; i < padding.size(); ++i) {
    const int64_t left_padding = padding[i];
    int64_t right_padding = padding[i];
    const int64_t spatial_dim = 2 + static_cast<int64_t>(i);

    if (ceil_mode && stride[i] > 1 && !input_shape.isDynamicDim(spatial_dim)) {
      right_padding += ComputeExtraCeilModePadding(
          input_shape.getDimSize(spatial_dim), kernel_size[i], stride[i],
          padding[i], dilation[i]);
    }
    ceil_mode_padding.push_back({left_padding, right_padding});
  }

  return ceil_mode_padding;
}

bool HasDynamicCeilModePadding(const mlir::RankedTensorType& input_shape,
                               const PoolingWindowParams& params) {
  if (!params.ceil_mode) {
    return false;
  }
  for (size_t i = 0; i < params.stride.size(); ++i) {
    const int64_t spatial_dim = 2 + static_cast<int64_t>(i);
    if (params.stride[i] > 1 && input_shape.isDynamicDim(spatial_dim)) {
      return true;
    }
  }
  return false;
}

mlir::MlirOp PadDynamicCeilModeSpatialDims(mlir::MlirOp operand,
                                           mlir::MlirOp pad_value,
                                           mlir::MlirOp batch_input,
                                           const PoolingWindowParams& params) {
  const mlir::RankedTensorType batch_type = GetTensorTypeOrDie(batch_input);
  if (!HasDynamicCeilModePadding(batch_type, params)) {
    return operand;
  }

  mlir::MlirBuilder& builder = operand.getBuilder();
  const mlir::Type i32_type = builder.getOpBuilder().getI32Type();
  const int64_t rank = GetTensorTypeOrDie(operand).getRank();
  const Dimensions operand_bounds = GetStaticShape(operand);
  const Dimensions batch_bounds = GetStaticShape(batch_input);

  std::vector<std::optional<mlir::MlirOp>> runtime_sizes(rank);
  LiftDynamicDimsToStaticBounds(operand, runtime_sizes);

  Dimensions high_pad(rank, 0);
  bool needs_static_pad = false;
  std::optional<mlir::MlirOp> in_bounds_mask;
  const mlir::RankedTensorType static_i32_type =
      mlir::RankedTensorType::get(operand_bounds, i32_type);

  for (size_t i = 0; i < params.stride.size(); ++i) {
    const int64_t dim = 2 + static_cast<int64_t>(i);
    if (params.stride[i] <= 1 || !batch_type.isDynamicDim(dim)) {
      continue;
    }
    ABSL_CHECK(runtime_sizes[dim].has_value());  // CRASH_OK
    mlir::MlirOp cur_size = *runtime_sizes[dim];
    AppendDimInBoundsMask(builder, static_i32_type, dim, cur_size,
                          in_bounds_mask);

    const int64_t bound_extra = ComputeExtraCeilModePadding(
        batch_bounds[dim], params.kernel_size[i], params.stride[i],
        params.padding[i], params.dilation[i]);
    if (bound_extra > 0) {
      high_pad[dim] = bound_extra;
      needs_static_pad = true;
    }

    mlir::MlirOp batch_dim_size =
        mlir::stablehlo::GetDimensionSize(batch_input, dim);
    mlir::MlirOp runtime_extra = ComputeDynamicExtraCeilModePadding(
        batch_dim_size, params.kernel_size[i], params.stride[i],
        params.padding[i], params.dilation[i]);
    runtime_sizes[dim] = mlir::stablehlo::Add(cur_size, runtime_extra);
  }

  if (in_bounds_mask.has_value()) {
    mlir::MlirOp mask = *in_bounds_mask;
    mlir::MlirOp pad_bcast = mlir::stablehlo::BroadcastInDim(
        GetTensorTypeOrDie(operand), pad_value, /*broadcast_dimensions=*/{});
    operand = mlir::stablehlo::Select(mask, operand, pad_bcast);
  }

  if (needs_static_pad) {
    const Dimensions zero_pad(rank, 0);
    operand =
        mlir::stablehlo::Pad(operand, pad_value, zero_pad, high_pad, zero_pad);
  }

  return RestoreDynamicDimSizes(operand, runtime_sizes);
}

// Creates the padding config to be used to pad the input tensor and iota tensor
// consistently, which is also required by StableHLO's ReduceWindowOp.
ReduceWindowAttributes GetReduceWindowAttributes(
    mlir::MlirBuilder& builder, Dimensions kernel_size_attr,
    Dimensions stride_attr, Dimensions dilation_attr,
    std::vector<PaddingParams> padding_pairs, const int64_t spatial_dim_count,
    const int64_t total_num_dims) {
  Dimensions kernel_size(2, 1);
  kernel_size.insert(kernel_size.end(), kernel_size_attr.begin(),
                     kernel_size_attr.end());

  Dimensions stride;
  stride.resize(2, 1);
  stride.insert(stride.end(), stride_attr.begin(), stride_attr.end());

  Dimensions dilation(2, 1);
  dilation.insert(dilation.end(), dilation_attr.begin(), dilation_attr.end());

  Dimensions base_dilations(total_num_dims, 1);

  // Create a flat list of padding values for StableHLO's ReduceWindowOp
  llvm::SmallVector<int64_t> flat_padding_list;
  flat_padding_list.reserve(total_num_dims * 2);

  for (int i = 0; i < total_num_dims - spatial_dim_count; ++i) {
    flat_padding_list.push_back(0);
    flat_padding_list.push_back(0);
  }
  for (const auto& dim_padding : padding_pairs) {
    flat_padding_list.push_back(dim_padding.left);   // low
    flat_padding_list.push_back(dim_padding.right);  // high
  }

  auto reduce_padding_attr = mlir::DenseIntElementsAttr::get(
      mlir::makeTensorType(builder.getContext(), {total_num_dims, 2},
                           builder.getOpBuilder().getI64Type()),
      flat_padding_list);

  return ReduceWindowAttributes{
      .window_dimensions =
          mlir::DenseI64ArrayAttr::get(&builder.getContext(), kernel_size),
      .window_strides =
          mlir::DenseI64ArrayAttr::get(&builder.getContext(), stride),
      .base_dilations =
          mlir::DenseI64ArrayAttr::get(&builder.getContext(), base_dilations),
      .window_dilations =
          mlir::DenseI64ArrayAttr::get(&builder.getContext(), dilation),
      .padding = reduce_padding_attr};
}

namespace {

int64_t GetSingleDim(const int64_t in, const int64_t i,
                     at::IntArrayRef kernel_size, at::IntArrayRef stride,
                     at::IntArrayRef padding, at::IntArrayRef dilation,
                     const bool ceil_mode) {
  const int64_t k = kernel_size.size() == 1 ? kernel_size[0] : kernel_size[i];
  const int64_t s =
      stride.empty() ? k : (stride.size() == 1 ? stride[0] : stride[i]);
  const int64_t p = padding.size() == 1 ? padding[0] : padding[i];
  const int64_t d = dilation.size() == 1 ? dilation[0] : dilation[i];
  const int64_t numerator = in + 2 * p - d * (k - 1) - 1;

  if (!ceil_mode) {
    return numerator / s + 1;
  }

  const int64_t out = (numerator + s - 1) / s + 1;
  if ((out - 1) * s >= in + p) {
    return out - 1;
  }
  return out;
}

}  // namespace

absl::StatusOr<Dimensions> GetPoolingOutputSize(
    at::IntArrayRef input_size, at::IntArrayRef kernel_size,
    at::IntArrayRef stride, at::IntArrayRef padding, at::IntArrayRef dilation,
    const bool ceil_mode, const int64_t spatial_dim_count) {
  const int64_t dims_num = input_size.size();
  TT_RET_CHECK(
      dims_num == spatial_dim_count + 1 || dims_num == spatial_dim_count + 2,
      error::kInvalidArgument)
      << "expected non-empty " << spatial_dim_count + 1 << "D or "
      << spatial_dim_count + 2 << "D (batch mode) tensor for input, got "
      << dims_num << "D tensor";

  // If (N, C, ...), spatial starts at 2. If (C, ...), starts at 1.
  const int64_t spatial_start_index =
      (dims_num == spatial_dim_count + 2) ? 2 : 1;

  Dimensions output_size(dims_num);
  std::copy(input_size.begin(), input_size.begin() + spatial_start_index,
            output_size.begin());

  // Calculates Spatial Dims (D, H, W).
  for (int i = 0; i < spatial_dim_count; ++i) {
    const int64_t in = input_size[spatial_start_index + i];
    const int64_t out =
        GetSingleDim(in, i, kernel_size, stride, padding, dilation, ceil_mode);
    output_size[spatial_start_index + i] = std::max<int64_t>(out, 1);
  }
  return output_size;
}

}  // namespace torch_tpu
