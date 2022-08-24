/**
* Copyright 2017-present, Facebook, Inc.
* All rights reserved.
*
* This source code is licensed under the license found in the
* LICENSE file in the root directory of this source tree.
*/

#include "dual.h"
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <utility>      // std::pair



template <typename scalar_t>
__global__ void get_bag_of_word_kernel(
        const scalar_t* __restrict__ sequence,
        scalar_t* __restrict__ masked_bow,
        const int vocab_size,
        const int N,
        const int bos,
        const int eos,
        const int pad
    ) {

    // const int b = blockIdx.x;
    // const int i = threadIdx.x;
    // const index = b * sequence_size + i;
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int tok_idx = sequence[index];
    const int bow_index = blockIdx.x * vocab_size + tok_idx;
    if (index < N) {
        masked_bow[bow_index] = (tok_idx != bos) && (tok_idx != eos) && (tok_idx != pad);
    }
}


torch::Tensor GetBOWCuda(
        torch::Tensor sequence,
        const int voc_size,
        const int bos,
        const int eos,
        const int pad
) {

    const auto batch_size = sequence.size(0);
    const auto seq_length = sequence.size(1);
    at::TensorOptions options(sequence.device());
    options = options
        .dtype(sequence.dtype())
        .layout(sequence.layout())
        .device(sequence.device());
    auto masked_bow = torch::zeros({batch_size, voc_size}, options);
    auto stream = at::cuda::getCurrentCUDAStream(sequence.device().index());
    // ScalarType::Bool
    AT_DISPATCH_ALL_TYPES(sequence.scalar_type(), "get_bag_of_word_kernel", ([&] {
        get_bag_of_word_kernel<scalar_t><<<batch_size, seq_length, 0, stream>>>(
            sequence.data_ptr<scalar_t>(),
            masked_bow.data_ptr<scalar_t>(),
            masked_bow.size(1),
            sequence.size(0) * sequence.size(1),
            bos,
            eos,
            pad
        );
    }));

    return masked_bow;
}
