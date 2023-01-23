#include "align.h"
// // #include <THC/THC.h>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <device_launch_parameters.h>
#include <utility>      // std::pair
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/detail/IndexUtils.cuh>
#include <ATen/cuda/detail/TensorInfo.cuh>
#include <c10/cuda/CUDACachingAllocator.h>
#include "atomics.cuh"

#define THREADS 1024
#define BLOCKS(N) (N + THREADS - 1) / THREADS

template <typename scalar_t>
__device__ scalar_t pow(scalar_t  x, const int p) {
    x = abs(x);
    scalar_t power = 1.;
    for (int i = 0; i < p; i++)
        power *= x;
    return power;
}

template <typename scalar_t>
__global__ void GetGraphKernel(
        const scalar_t* __restrict__ x,
        scalar_t* __restrict__ graph,
        const int B,
        const int N,
        const int L,
        const int pad
    ) {

    const int index = blockIdx.x * blockDim.x + threadIdx.x;

    const int mul2 = N * L;
    const int mul1 = mul2 * L;
    const int mul0 = mul1 * N;

    const int max = B * mul0;

    const int b = index / mul0;
    const int index0 = index % mul0;
    const int n = index0 / mul1;
    const int index1 = index0 % mul1;
    const int i = index1 / mul2;
    const int index2 = index1 % mul2;
    const int m = index2 / L;
    // const int j = index2 % L;

    if ((index < max)) {
        graph[index] = (
            (
                x[b * mul2 + n * L + i] == 
                x[b * mul2 + index2]
            )
            && (x[b * mul2 + n * L + i] != pad)
            && (n != m)
        );
    }
}

template <typename scalar_t>
__global__ void ComputeDistKernel(
        const scalar_t* __restrict__ pos,
        const at::cuda::detail::TensorInfo<int64_t, int> graph_info,
        scalar_t* __restrict__ out,
        const int B,
        const int N,
        const int L,
        const int p
    ) {

    //// B = blockDim.x
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int mul2 = N * L;
    const int mul1 = mul2 * L;
    const int mul0 = mul1 * N;

    const int max = B * mul0;

    const int b = index / mul0;
    const int index0 = index % mul0;
    const int n = index0 / mul1;
    const int index1 = index0 % mul1;
    const int i = index1 / mul2;
    const int index2 = index1 % mul2;
    const int m = index2 / L;
    const int j = index2 % L;

    const int offset = at::cuda::detail::IndexToOffset<int64_t, int, -1>::get(index, graph_info);
    if ((index < max) && (graph_info.data[offset] > 0)) {
        atomMin(out + b * mul2 + n * L + i, pow<scalar_t>(
            pos[b * mul2 + n * L + i] - 
            pos[b * mul2 + m * L + j],
            p
        ));
    }
}

template <typename scalar_t>
__global__ void ComputeDistArgKernel(
        const scalar_t* __restrict__ pos,
        const at::cuda::detail::TensorInfo<int64_t, int> graph_info,
        const scalar_t* __restrict__ out,
        int64_t* __restrict__ arg_out,
        const int B,
        const int N,
        const int L,
        const int p
    ) {

    //// B = blockDim.x
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int mul2 = N * L;
    const int mul1 = mul2 * L;
    const int mul0 = mul1 * N;

    const int max = B * mul0;

    const int b = index / mul0;
    const int index0 = index % mul0;
    const int n = index0 / mul1;
    const int index1 = index0 % mul1;
    const int i = index1 / mul2;
    const int index2 = index1 % mul2;
    const int m = index2 / L;
    const int j = index2 % L;

    const int offset = at::cuda::detail::IndexToOffset<int64_t, int, -1>::get(index, graph_info);
    if (
        (index < max) && 
        (graph_info.data[offset] > 0) && 
        (
            (out[b * mul2 + n * L + i] ==
                pow<scalar_t>(
                    pos[b * mul2 + n * L + i] -
                    pos[b * mul2 + m * L + j],
                    p
                )
            )
        )
    ) {
        arg_out[b * mul2 + n * L + i] = m * L + j;
    }
}

/////////// Backward Kernels

template <typename scalar_t>
__global__ void ComputeDistBackwardSignKernel(
        const scalar_t* __restrict__ grad_out,
        const scalar_t* __restrict__ pos,
        const at::cuda::detail::TensorInfo<int64_t, int> arg_info,
        const at::cuda::detail::TensorInfo<int64_t, int> graph_mask_info,
        scalar_t* __restrict__ grad_sign,
        const int B,
        const int N,
        const int L,
        const int p
    ) {

    //// B = blockDim.x
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int mul2 = N * L;
    const int mul1 = mul2 * L;
    const int mul0 = mul1 * N;

    const int max = B * mul0;
    const int b = index / mul0;
    const int index0 = index % mul0;
    const int n = index0 / mul1;
    const int index1 = index0 % mul1;
    const int i = index1 / mul2;
    const int index2 = index1 % mul2;

    const int offset = at::cuda::detail::IndexToOffset<int64_t, int, -1>::get(b * mul2 + n * L + i, arg_info);
    const int index_arg = arg_info.data[offset];

    const int offset_graph_mask = at::cuda::detail::IndexToOffset<int64_t, int, -1>::get(b * mul2 + n * L + i, graph_mask_info);

    if ((index < max) && (graph_mask_info.data[offset_graph_mask] > 0) && (index_arg == index2)) {
        const scalar_t diff = pos[b * mul2 + n * L + i] - pos[b * mul2 + index2];
        grad_sign[index] = ((scalar_t)(diff > 0) - (scalar_t)(diff < 0)) * pow(diff, p - 1) * grad_out[b * mul2 + n * L + i];
    }
}

template <typename scalar_t>
__global__ void ComputeDistBackwardKernel(
        const scalar_t* __restrict__ grad_sign,
        scalar_t* __restrict__ grad_in,
        const int B,
        const int N,
        const int L
    ) {

    //// B = blockDim.x
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int mul2 = N * L;
    const int mul1 = mul2 * L;
    const int mul0 = mul1 * N;

    const int max = B * mul0;
    const int b = index / mul0;
    const int index0 = index % mul0;
    const int n = index0 / mul1;
    const int index1 = index0 % mul1;
    const int i = index1 / mul2;
    const int index2 = index1 % mul2;
    const int m = index2 / L;
    const int j = index2 % L;

    if ((index < max)) {
        atomAdd(
            &grad_in[b * mul2 + n * L + i],
            grad_sign[index] -
            grad_sign[b * mul0 + m * mul1 + j * mul2 + n * L + i]
        );
    }
}

////////////////////////////////////////////////////////////////

torch::Tensor GetGraphCuda(
    torch::Tensor x,
    const int pad
) {

    const auto batch_size = x.size(0);
    const auto N = x.size(1);
    const auto seq_length = x.size(2);
    at::TensorOptions options(x.device());
    options = options
        .dtype(x.dtype())
        .layout(x.layout());
    torch::Tensor graph = torch::zeros({batch_size, N, seq_length, N, seq_length}, options);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(x.device().index());
    const auto numel = batch_size * seq_length * N * seq_length * N;
    AT_DISPATCH_ALL_TYPES(x.scalar_type(), "GetGraphKernel", ([&] {
        GetGraphKernel<scalar_t><<<BLOCKS(numel), THREADS, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            graph.data_ptr<scalar_t>(),
            batch_size,
            N,
            seq_length,
            pad
        );
    }));

    return graph;
}

std::tuple<torch::Tensor, torch::Tensor> ComputeDistCuda(
    torch::Tensor graph,
    torch::Tensor pos,
    torch::Tensor graph_mask,
    const int p
) {

    const auto batch_size = graph.size(0);
    const auto N = graph.size(1);
    const auto seq_length = graph.size(2);
    at::TensorOptions options(pos.device());
    options = options
        .dtype(pos.dtype())
        .layout(pos.layout());
    torch::Tensor out = torch::full({batch_size, N, seq_length}, seq_length + 1024, options);
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(pos.device().index());
    // ScalarType::Bool
    auto graph_info = at::cuda::detail::getTensorInfo<int64_t, int>(graph);
    const auto numel = batch_size * seq_length * N * seq_length * N;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(pos.scalar_type(), "ComputeDistKernel", ([&] {
        ComputeDistKernel<scalar_t><<<BLOCKS(numel), THREADS, 0, stream>>>(
            pos.data_ptr<scalar_t>(),
            graph_info,
            out.data_ptr<scalar_t>(),
            batch_size,
            N,
            seq_length,
            p
        );
    }));
    at::TensorOptions options2(pos.device());
    options2 = options2
        .dtype(torch::kInt64)
        .layout(pos.layout());
    torch::Tensor arg_out = torch::zeros({batch_size, N, seq_length}, options2);
    // torch::Tensor arg_out_2 = torch::zeros({batch_size, N, seq_length}, options2);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(pos.scalar_type(), "ComputeDistArgKernel", ([&] {
        ComputeDistArgKernel<scalar_t><<<BLOCKS(numel), THREADS, 0, stream>>>(
            pos.data_ptr<scalar_t>(),
            graph_info,
            out.data_ptr<scalar_t>(),
            arg_out.data_ptr<int64_t>(),
            batch_size,
            N,
            seq_length,
            p
        );
    }));
    return std::make_tuple(out, arg_out);
}

torch::Tensor ComputeDistBackwardCuda(
    torch::Tensor arg_out,
    torch::Tensor graph_mask,
    torch::Tensor pos,
    torch::Tensor grad_out,
    const int p
) {

    const auto batch_size = pos.size(0);
    const auto N = pos.size(1);
    const auto seq_length = pos.size(2);
    at::TensorOptions options = grad_out.options();    
    
    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream(pos.device().index());
    // ScalarType::Bool
    auto arg_info = at::cuda::detail::getTensorInfo<int64_t, int>(arg_out);
    auto graph_mask_info = at::cuda::detail::getTensorInfo<int64_t, int>(graph_mask);
    const auto numel = batch_size * seq_length * N;
    const auto numel_more = numel * seq_length * N;

    torch::Tensor grad_sign = torch::zeros({batch_size, N, seq_length, N, seq_length}, options);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(pos.scalar_type(), "ComputeDistBackwardSignKernel", ([&] {
        ComputeDistBackwardSignKernel<scalar_t><<<BLOCKS(numel_more), THREADS, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            pos.data_ptr<scalar_t>(),
            arg_info,
            graph_mask_info,
            grad_sign.data_ptr<scalar_t>(),
            batch_size,
            N,
            seq_length,
            p
        );
    }));
    torch::Tensor grad_in = torch::zeros({batch_size, N, seq_length}, options);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(pos.scalar_type(), "ComputeDistBackwardKernel", ([&] {
        ComputeDistBackwardKernel<scalar_t><<<BLOCKS(numel_more), THREADS, 0, stream>>>(
            grad_sign.data_ptr<scalar_t>(),
            grad_in.data_ptr<scalar_t>(),
            batch_size,
            N,
            seq_length
        );
    }));

    return grad_in;
}
