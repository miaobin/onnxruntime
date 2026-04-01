// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) Intel Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/providers/common.h>

#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/shared/utils/utils.h"

#include "base_op_builder.h"

namespace onnxruntime {
namespace webnn {
// Add operator related.

Status BaseOpBuilder::AddToModelBuilder(ModelBuilder& model_builder, const Node& node,
                                        const logging::Logger& logger) const {
  ORT_RETURN_IF_NOT(
      IsOpSupported(model_builder.GetGraphViewer(), node, model_builder.GetWebnnDeviceType(),
                    model_builder.GetOpSupportLimits(), logger),
      "Unsupported operator ", node.OpType());
  ORT_RETURN_IF_ERROR(AddToModelBuilderImpl(model_builder, node, logger));

  // Auto-propagate dynamic dimension info through shape-preserving operations.
  // If the first input has dim info and the first output does not yet have dim info,
  // and the output has the same rank as the input, copy the dim info to the output.
  // This enables Reshape chains separated by ops like LayerNorm, Add, etc. to work.
  //
  // TODO: Current limitations:
  // - Only propagates from the first input (input_defs[0]). For binary ops like Add(A, B),
  //   if only B carries dim info, it won't be propagated.
  // - Only sets dim info on the first output (output_defs[0]). Multi-output ops like
  //   SkipSimplifiedLayerNormalization may need dim info on additional outputs.
  // - Does not handle rank-changing ops (Squeeze, Unsqueeze, Flatten, etc.). Those ops
  //   would need custom dim info logic in their respective op builders.
  // - Axis-reordering ops (e.g., Transpose) have the same rank but permute axis semantics.
  //   Blindly copying dim info would assign dynamic dims to wrong axes.
  const auto& input_defs = node.InputDefs();
  const auto& output_defs = node.OutputDefs();
  if (!input_defs.empty() && !output_defs.empty() &&
      input_defs[0]->Name() != "" && output_defs[0]->Name() != "") {
    const auto* input_dim_info = model_builder.GetOperandDimInfo(input_defs[0]->Name());
    if (input_dim_info != nullptr &&
        model_builder.GetOperandDimInfo(output_defs[0]->Name()) == nullptr) {
      // Check that the output has the same rank as the input dim info.
      const auto* output_shape = output_defs[0]->Shape();
      if (output_shape != nullptr &&
          static_cast<size_t>(output_shape->dim_size()) == input_dim_info->size()) {
        // Copy dim info from input to output (shape-preserving op).
        OperandDimInfo output_dim_info(*input_dim_info);
        model_builder.SetOperandDimInfo(output_defs[0]->Name(), std::move(output_dim_info));
      }
    }
  }

  return Status::OK();
}

// Operator support related.

bool BaseOpBuilder::IsOpSupported(const GraphViewer& graph_viewer, const Node& node,
                                  const WebnnDeviceType device_type, const emscripten::val& wnn_limits,
                                  const logging::Logger& logger) const {
  if (!HasSupportedInputs(graph_viewer, node, wnn_limits, logger))
    return false;

  if (!HasSupportedOutputs(node, wnn_limits, logger))
    return false;

  if (!HasSupportedOpSet(node, logger))
    return false;

  return IsOpSupportedImpl(graph_viewer, node, device_type, logger);
}

bool BaseOpBuilder::HasSupportedInputs(const GraphViewer& graph_viewer, const Node& node,
                                       const emscripten::val& wnn_limits, const logging::Logger& logger) const {
  const auto node_name = MakeString("Node [", node.Name(), "] type [", node.OpType(), "]");
  for (const auto* input : node.InputDefs()) {
    if (!IsTensorShapeSupported(*input, node_name, logger, allow_empty_tensor_as_input_)) {
      return false;
    }
  }

  return HasSupportedInputsImpl(graph_viewer, node, wnn_limits, logger);
}

bool BaseOpBuilder::HasSupportedInputsImpl(const GraphViewer&, const Node& node,
                                           const emscripten::val& wnn_limits,
                                           const logging::Logger& logger) const {
  // We only check the type of input 0 by default, specific op builder can override this.
  const auto& input = *node.InputDefs()[0];
  const std::string_view op_type = node.OpType();
  int32_t input_type;
  if (!GetType(input, input_type, logger))
    return false;

  const std::string_view webnn_op_type = GetWebNNOpType(op_type);
  const std::string_view webnn_input_name = GetWebNNOpFirstInputName(op_type);
  return IsDataTypeSupportedByWebNNOp(op_type, webnn_op_type, input_type, wnn_limits,
                                      webnn_input_name, "input", logger) &&
         IsInputRankSupportedByOp(node, wnn_limits, logger);
}

bool BaseOpBuilder::HasSupportedOutputs(const Node& node, const emscripten::val& wnn_limits,
                                        const logging::Logger& logger) const {
  const auto node_name = MakeString("Node [", node.Name(), "] type [", node.OpType(), "]");
  for (const auto* output : node.OutputDefs()) {
    if (!IsTensorShapeSupported(*output, node_name, logger)) {
      return false;
    }
  }

  return HasSupportedOutputsImpl(node, wnn_limits, logger);
}

bool BaseOpBuilder::HasSupportedOutputsImpl(const Node& node,
                                            const emscripten::val& wnn_limits,
                                            const logging::Logger& logger) const {
  // We only check the type of output 0 by default, specific op builder can override this.
  const auto& output = *node.OutputDefs()[0];
  const std::string_view op_type = node.OpType();
  int32_t output_type;
  if (!GetType(output, output_type, logger))
    return false;

  return IsDataTypeSupportedByOp(op_type, output_type, wnn_limits, "output", "Output", logger);
}

bool BaseOpBuilder::HasSupportedOpSet(const Node& node,
                                      const logging::Logger& logger) const {
  auto since_version = node.SinceVersion();
  if (since_version < GetMinSupportedOpSet(node) || since_version > GetMaxSupportedOpSet(node)) {
    LOGS(logger, VERBOSE) << "Current opset since version of "
                          << node.OpType() << " is " << since_version
                          << ", WebNN EP only supports for its opset ["
                          << GetMinSupportedOpSet(node) << ", "
                          << GetMaxSupportedOpSet(node) << "]";
    return false;
  }

  return true;
}

}  // namespace webnn
}  // namespace onnxruntime
