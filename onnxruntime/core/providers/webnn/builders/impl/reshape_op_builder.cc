// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/initializer.h"
#include "core/providers/common.h"
#include "core/providers/cpu/tensor/reshape_helper.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {

class ReshapeOpBuilder : public BaseOpBuilder {
  // Add operator related.
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related.
 private:
  bool IsOpSupportedImpl(const GraphViewer& graph_viewer, const Node& node,
                         const WebnnDeviceType /* device_type */, const logging::Logger& logger) const override;
};

// Add operator related.

void ReshapeOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  model_builder.AddInitializerToSkip(node.InputDefs()[1]->Name());
}

Status ReshapeOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                               const Node& node,
                                               const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& initializers(model_builder.GetInitializerTensors());
  const auto& target_shape_tensor = *initializers.at(input_defs[1]->Name());
  const auto& target_shape_tensor_dims = target_shape_tensor.dims();
  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val new_shape = emscripten::val::array();

  const std::string& input_name = input_defs[0]->Name();
  const std::string& output_name = node.OutputDefs()[0]->Name();

  // Retrieve per-axis dynamic dim info for the input operand (may be nullptr if fully static).
  const OperandDimInfo* input_dim_info = model_builder.GetOperandDimInfo(input_name);

  // Fallback: if no dim info was propagated (e.g., due to rank changes between graph input and here),
  // try to construct it directly from the input's ONNX shape proto + FreeDimensionBounds.
  OperandDimInfo constructed_dim_info;
  if (input_dim_info == nullptr) {
    if (model_builder.BuildDimInfoFromNodeArg(*input_defs[0], constructed_dim_info)) {
      input_dim_info = &constructed_dim_info;
    }
  }

  // Do nothing if target shape is an empty shape, which means converting to a scalar.
  if (!target_shape_tensor_dims.empty()) {
    const int64_t* raw_target_shape = target_shape_tensor.int64_data().empty()
                                          ? reinterpret_cast<const int64_t*>(target_shape_tensor.raw_data().data())
                                          : target_shape_tensor.int64_data().data();

    const auto size = target_shape_tensor_dims[0];
    std::vector<int64_t> input_shape;
    ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");

    // Check if input has any dynamic dimensions.
    bool has_dynamic_input = input_dim_info != nullptr;

    if (has_dynamic_input) {
      // Dynamic path: handle 0, static, and -1 axes with dynamic dimension tracking
      // 1. For each input axis: classify as 0-consumed (by a target 0 axis), static, or dynamic.
      //    Multiply static input dims into static_input_product.
      //    There should be exactly one dynamic input dim contributing to the -1 output axis.
      // 2. For each output axis: classify as 0 (pass-through from input), static, or -1 (infer).
      //    Multiply static output dims into static_output_product.
      // 3. R = static_input_product / static_output_product.
      // 4. For the -1 axis, new_multiplier = old_multiplier * R (expansion) or old_multiplier / R (contraction).
      // 5. Build the WebNN dynamic dimension descriptor and register derived dim info.

      int64_t static_input_product = 1;
      int64_t static_output_product = 1;
      int minus_one_axis = -1;
      // Track which dynamic dim from input contributes to -1 axis.
      const DynamicDimInfo* contributing_dyn_dim = nullptr;
      int contributing_dyn_count = 0;

      // Pass 1: Classify input axes.
      // Build a set of axes consumed by 0 in target shape.
      InlinedHashSet<size_t> zero_consumed_axes;
      for (int64_t i = 0; i < size; ++i) {
        if (raw_target_shape[i] == 0 && i < static_cast<int64_t>(input_shape.size())) {
          zero_consumed_axes.insert(static_cast<size_t>(i));
        }
      }

      for (size_t i = 0; i < input_shape.size(); ++i) {
        if (zero_consumed_axes.count(i)) {
          // This axis is consumed by a 0 in target: its value (static or dynamic) passes through.
          continue;
        }
        bool is_dyn = (input_dim_info->size() > i && (*input_dim_info)[i].has_value());
        if (is_dyn) {
          contributing_dyn_dim = &(*input_dim_info)[i].value();
          contributing_dyn_count++;
        } else {
          // Static input axis (input_shape[i] should be > 0).
          ORT_RETURN_IF(input_shape[i] <= 0,
                        "Unexpected non-positive static input dim at axis ", i);
          static_input_product *= input_shape[i];
        }
      }

      // Pass 2: Classify output axes.
      for (int64_t i = 0; i < size; ++i) {
        if (raw_target_shape[i] == 0) {
          // Pass-through from input (handled separately).
        } else if (raw_target_shape[i] == -1) {
          minus_one_axis = static_cast<int>(i);
        } else {
          static_output_product *= raw_target_shape[i];
        }
      }

      // Build output dim info.
      OperandDimInfo output_dim_info(static_cast<size_t>(size), std::nullopt);

      // Populate dim info for 0 (pass-through) axes.
      for (int64_t i = 0; i < size; ++i) {
        if (raw_target_shape[i] == 0) {
          if (input_dim_info->size() > static_cast<size_t>(i) &&
              (*input_dim_info)[static_cast<size_t>(i)].has_value()) {
            output_dim_info[static_cast<size_t>(i)] = (*input_dim_info)[static_cast<size_t>(i)];
          }
        }
      }

      // Handle the -1 axis.
      if (minus_one_axis >= 0) {
        if (contributing_dyn_count == 0) {
          // All input dims feeding -1 are static. Nothing to set in output_dim_info.
        } else if (contributing_dyn_count == 1 && contributing_dyn_dim != nullptr) {
          // Exactly one dynamic dim contributing: compute the rational ratio.
          // new = old_num/old_den * static_input_product / static_output_product
          //     = (old_num * static_input_product) / (old_den * static_output_product)
          ORT_RETURN_IF(static_output_product == 0, "static_output_product is zero, cannot compute -1 axis");

          int64_t new_num64 = static_cast<int64_t>(contributing_dyn_dim->numerator) * static_input_product;
          int64_t new_den64 = static_cast<int64_t>(contributing_dyn_dim->denominator) * static_output_product;
          ORT_RETURN_IF(new_den64 == 0, "Computed zero denominator for -1 axis");

          // Simplify via GCD before narrowing to int32_t to prevent overflow.
          DynamicDimInfo new_dim = DynamicDimInfo::CreateSimplified(
              contributing_dyn_dim->base_name, new_num64, new_den64,
              contributing_dyn_dim->base_min, contributing_dyn_dim->base_max);

          // Validate that the resulting dimension produces integer values.
          ORT_RETURN_IF(new_dim.numerator <= 0 || new_dim.denominator <= 0,
                        "Computed non-positive rational scale for -1 axis: ",
                        new_dim.numerator, "/", new_dim.denominator);
          ORT_RETURN_IF((static_cast<int64_t>(new_dim.base_min) * new_dim.numerator) % new_dim.denominator != 0,
                        "Dynamic dim min (", new_dim.base_min,
                        ") not divisible by scale denominator (", new_dim.denominator, ")");
          ORT_RETURN_IF((static_cast<int64_t>(new_dim.base_max) * new_dim.numerator) % new_dim.denominator != 0,
                        "Dynamic dim max (", new_dim.base_max,
                        ") not divisible by scale denominator (", new_dim.denominator, ")");

          output_dim_info[static_cast<size_t>(minus_one_axis)] = std::move(new_dim);
        } else {
          // Multiple dynamic dims contributing to -1.
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "Reshape with -1 axis: multiple dynamic input dimensions (",
                                 contributing_dyn_count, ") contributing is not supported");
        }
      }

      // Build new_shape in correct order.
      // Cache the output shape proto for resolving real dim_param names.
      const auto* output_shape_proto = node.OutputDefs()[0]->Shape();

      // Helper: resolve the WebNN dimension name for a given axis.
      // Prefers the real ONNX dim_param if available, otherwise uses DimName().
      auto resolve_dim_name = [output_shape_proto](const DynamicDimInfo& dim, int axis) -> std::string {
        if (output_shape_proto != nullptr &&
            axis < output_shape_proto->dim_size() &&
            !output_shape_proto->dim(axis).has_dim_value() &&
            !output_shape_proto->dim(axis).dim_param().empty()) {
          return output_shape_proto->dim(axis).dim_param();
        }
        return dim.DimName();
      };

      // Helper: push an MLDynamicDimension descriptor for a dynamic axis.
      auto push_dynamic_dim = [&new_shape, &resolve_dim_name](const DynamicDimInfo& dim, int axis) {
        emscripten::val dim_obj = emscripten::val::object();
        dim_obj.set("name", emscripten::val(resolve_dim_name(dim, axis)));
        dim_obj.set("minSize", dim.CurrentMin());
        dim_obj.set("maxSize", dim.CurrentMax());
        new_shape.call<void>("push", dim_obj);
      };

      for (int64_t i = 0; i < size; ++i) {
        if (raw_target_shape[i] == 0) {
          // Pass-through from input. If this axis is dynamic, we must push an MLDynamicDimension
          // descriptor because input["shape"][i] returns 0 for dynamic dims in WebNN.
          if (output_dim_info[static_cast<size_t>(i)].has_value()) {
            push_dynamic_dim(output_dim_info[static_cast<size_t>(i)].value(), static_cast<int>(i));
          } else {
            new_shape.call<void>("push", input["shape"][static_cast<int>(i)]);
          }
        } else if (raw_target_shape[i] == -1) {
          if (minus_one_axis >= 0 && output_dim_info[static_cast<size_t>(i)].has_value()) {
            push_dynamic_dim(output_dim_info[static_cast<size_t>(i)].value(), static_cast<int>(i));
          } else {
            // Static -1 axis: compute the inferred value.
            ORT_RETURN_IF(static_output_product == 0, "static_output_product is zero");
            int64_t inferred = static_input_product / static_output_product;
            new_shape.call<void>("push", SafeInt<uint32_t>(inferred));
          }
        } else {
          uint32_t dim_value = SafeInt<uint32_t>(raw_target_shape[i]);
          new_shape.call<void>("push", dim_value);
        }
      }

      // Register output dim info if it has any dynamic dimensions.
      bool has_output_dynamic = false;
      for (const auto& d : output_dim_info) {
        if (d.has_value()) {
          has_output_dynamic = true;
          break;
        }
      }
      if (has_output_dynamic) {
        model_builder.SetOperandDimInfo(output_name, std::move(output_dim_info));
      }
    } else {
      // Safety check: verify the input is truly fully static. If the input actually has
      // dynamic dimensions but we have no dim info, the static path would produce incorrect results.
      // Fail explicitly rather than silently computing wrong shapes.
      if (std::any_of(input_shape.begin(), input_shape.end(), [](int64_t d) { return d <= 0; })) {
        bool target_needs_dynamic = false;
        for (int64_t i = 0; i < size; ++i) {
          if (raw_target_shape[i] == 0 || raw_target_shape[i] == -1) {
            target_needs_dynamic = true;
            break;
          }
        }
        if (target_needs_dynamic) {
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                                 "Reshape node '", node.Name(), "': input '", input_name,
                                 "' has dynamic dimensions but no dimension tracking info is available. "
                                 "This may happen when dimension info propagation is broken by rank-changing "
                                 "operations between two Reshapes. Ensure FreeDimensionBounds are configured "
                                 "for all base dynamic dimensions, or add custom dim info propagation for "
                                 "the intermediate rank-changing operation.");
        }
      }

      TensorShapeVector target_shape{raw_target_shape, raw_target_shape + size};
      ReshapeHelper helper(TensorShape(input_shape), target_shape);

      for (size_t axis = 0; axis < static_cast<size_t>(size); ++axis) {
        if (target_shape[axis] == 0) {
          new_shape.call<void>("push", input["shape"][axis]);
        } else {
          uint32_t dim_value = SafeInt<uint32_t>(target_shape[axis]);
          new_shape.call<void>("push", dim_value);
        }
      }
    }
  }

  emscripten::val options = emscripten::val::object();
  options.set("label", node.Name());
  emscripten::val console = emscripten::val::global("console");
  console.call<void>("log", emscripten::val("[WebNN][ReshapeOpBuilder] node=" + node.Name() + " input.shape="), input["shape"]);
  console.call<void>("log", emscripten::val("[WebNN][ReshapeOpBuilder] node=" + node.Name() + " new_shape="), new_shape);

  emscripten::val output = model_builder.GetBuilder().call<emscripten::val>("reshape",
                                                                            input,
                                                                            new_shape,
                                                                            options);
  model_builder.AddOperand(output_name, std::move(output));
  return Status::OK();
}

// Operator support related.

bool ReshapeOpBuilder::IsOpSupportedImpl(const GraphViewer& graph_viewer,
                                         const Node& node,
                                         const WebnnDeviceType /* device_type */,
                                         const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& perm_name = input_defs[1]->Name();
  const auto* perm_init = graph_viewer.GetConstantInitializer(perm_name);
  if (!perm_init) {
    LOGS(logger, VERBOSE) << "New shape of reshape must be a constant initializer";
    return false;
  }

  const auto& perm_tensor = *perm_init;
  std::vector<uint8_t> unpacked_tensor;
  if (!UnpackInitializerData(perm_tensor, unpacked_tensor, graph_viewer, logger)) {
    return false;
  }

  const int64_t* raw_new_shape = reinterpret_cast<const int64_t*>(unpacked_tensor.data());
  const auto& perm_dims = perm_tensor.dims();

  // WebNN reshape does not support 0 as dimension.
  NodeAttrHelper helper(node);
  const bool allow_zero = helper.Get("allowzero", 0) == 1;
  if (allow_zero && !perm_dims.empty()) {
    for (int64_t i = 0; i < perm_dims[0]; i++) {
      if (raw_new_shape[i] == 0) {
        LOGS_DEFAULT(VERBOSE) << "Reshape doesn't support 0 reshape dimension when allowzero is enabled";
        return false;
      }
    }
  }

  return true;
}

void CreateReshapeOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<ReshapeOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
