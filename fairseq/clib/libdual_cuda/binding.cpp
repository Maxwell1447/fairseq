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
#define CHECK_CPU(x) TORCH_CHECK(x.device() == torch::kCPU, #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT_CUDA(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_INPUT_CPU(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)

template <typename scalar_t>
torch::Tensor GetBOWCPU(
    torch::Tensor sequence,
    const int voc_size,
    const int bos,
    const int eos,
    const int pad
) {
    const int batch_size = sequence.size(0);
    const int seq_length = sequence.size(1);
    at::TensorOptions options(sequence.device());
    options = options
        .dtype(sequence.dtype())
        .layout(sequence.layout());
    auto masked_bow = torch::zeros({batch_size, voc_size}, options);

    const scalar_t* sequence_ptr = sequence.data_ptr<scalar_t>();
    scalar_t* masked_bow_ptr = masked_bow.data_ptr<scalar_t>();


    long index;
    long tok_idx;
    long bow_index;
    for (long b = 0; b < batch_size; b++) {
        for (long i = 0; i < seq_length; i++) {
            index = b * seq_length + i;
            tok_idx = sequence_ptr[index];
            if ((tok_idx < voc_size) && (tok_idx >= 0)) {
                bow_index = b * voc_size + tok_idx;
                masked_bow_ptr[bow_index] = (tok_idx != bos) && (tok_idx != eos) && (tok_idx != pad);
            }
        }
    }
    return masked_bow;
}


torch::Tensor GetBOW(
    torch::Tensor sequence,
    const int voc_size,
    const int bos,
    const int eos,
    const int pad
) {
    if (sequence.device().is_cuda()) {
        CHECK_INPUT_CUDA(sequence);
        return GetBOWCuda(sequence, voc_size, bos, eos, pad);
    }
    else {
        CHECK_INPUT_CPU(sequence);
        if (sequence.dtype() == torch::kInt64)
            return GetBOWCPU<long>(sequence, voc_size, bos, eos, pad);
        else if (sequence.dtype() == torch::kInt32)
            return GetBOWCPU<int>(sequence, voc_size, bos, eos, pad);
        else if (sequence.dtype() == torch::kInt16)
            return GetBOWCPU<short>(sequence, voc_size, bos, eos, pad);
        else
            throw (sequence.dtype());
    }
}
    


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_bow_mask_from_sequence", &GetBOW, "Get Bag of words From Sequence");
}
