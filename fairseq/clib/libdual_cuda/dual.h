/**
 * Copyright 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <torch/extension.h>

torch::Tensor GetBOWCuda(
        torch::Tensor sequence,
        const int voc_size,
        const int bos,
        const int eos,
        const int pad
);
