/**
 * Copyright 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 This code is partially adpoted from https://github.com/1ytic/pytorch-edit-distance
 */

#include "dual.h"
#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor GetBOW(
        torch::Tensor sequence,
        const int voc_size,
        const int bos,
        const int eos,
        const int pad
) {

    CHECK_INPUT(sequence);
    return GetBOWCuda(sequence, voc_size, bos, eos, pad);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_bow_mask_from_sequence", &GetBOW, "Get Bag of words From Sequence");
}
