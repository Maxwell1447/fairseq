
/**
 * Copyright 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <torch/extension.h>
#include <tuple>

torch::Tensor GetGraphCuda(
    torch::Tensor x,
    const int pad
);

std::tuple<torch::Tensor, torch::Tensor> ComputeDistCuda(
    torch::Tensor graph,
    torch::Tensor pos,
    torch::Tensor graph_mask,
    const int p
);

torch::Tensor ComputeDistBackwardCuda(
    torch::Tensor arg_out,
    torch::Tensor graph_mask,
    torch::Tensor pos,
    torch::Tensor grad_out,
    const int p
);