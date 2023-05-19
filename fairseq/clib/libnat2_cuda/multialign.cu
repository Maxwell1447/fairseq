#include "multialign.h"

// #include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <utility>      // std::pair
#include <c10/cuda/CUDAStream.h>

#define THREAD 1024

template <typename scalar_t>
__global__ void get_mask_kernel(
    const scalar_t* __restrict__ y_ref,
    const scalar_t* __restrict__ ys,
    int* mask,
    const int B,
    const int L_ref,
    const int N,
    const int L,
    const int pad)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = idx % L_ref;
    idx /= L_ref;
    const int i = idx % L;
    idx /= L;
    const int n = idx % N;
    const int b = idx / N;
    if (b < B)
    {
        const scalar_t ref = y_ref[b * L_ref + j];
        const scalar_t ret = ys[b * N * L + n * L + i];
        if (((ref == ret) && (ret != pad) && (ret != pad)))
            mask[b * N * L + n * L + i] = 1;
    }
}

template <typename scalar_t>
__global__ void get_ref_mask_kernel(
    const scalar_t* __restrict__ y_ref,
    const scalar_t* __restrict__ ys,
    const scalar_t* __restrict__ Is,
    int* mask,
    const int B,
    const int L_ref,
    const int N,
    const int L_short,
    const int L,
    const int pad)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = idx % L_ref;
    idx /= L_ref;
    const int i = idx % L_short;
    idx /= L_short;
    const int n = idx % N;
    const int b = idx / N;
    if (b < B)
    {
        const scalar_t ref = y_ref[b * L_ref + j];
        const scalar_t ret = ys[b * N * L + n * L + Is[b * N * L_short + n * L_short + i]];
        if (((ref == ret) && (ret != pad) && (ret != pad)))
            mask[b * N * L_ref + j * N + n] = 1;
        //////////// DEBUG
        // if (n == 0 && ((j == 0 && i == 0) || (j == 4 && i == 3)))
        //     printf(
        //         "[%d, %d, %d, %d] Is=%d y_ref=%d ys=%d\n",
        //         b, j, n, i,
        //         (int)Is[b * N * L_short + n * L_short + i],
        //         (int)ref, (int)ret
        //     );
        // if (n == 0 && (ref != pad) && b == 0 && i == 0)
        //     printf(
        //         "y_ref[0, %d]=%d\n",
        //         (int)j, (int)ref
        //     );
        ////////////
    }
}

template <typename scalar_t>
__global__ void build_min_preference_kernel(
    const scalar_t* __restrict__ y_ref,
    const scalar_t* __restrict__ ys,
    const scalar_t* __restrict__ Is,
    int* V,
    const int B,
    const int L_ref,
    const int N,
    const int L_short,
    const int L,
    const int pad)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = idx % L_short;
    idx /= L_short;
    const int i_tilde = idx % L_short;
    idx /= L_short;
    const int j = idx % L_ref;
    idx /= L_ref;
    const int n = idx % N;
    const int b = idx / N;
    if (b < B && i_tilde >= i)
    {
        const scalar_t ref = y_ref[b * L_ref + j];
        const scalar_t ret = ys[b * N * L + n * L + Is[b * N * L_short + n * L_short + i_tilde]];
        //////////// DEBUG
        // if (b == 0 && n == 1 && i_tilde == 0 && j == 0)
        //     printf(
        //         "[%d, %d, %d, %d] %d  y_ref=%d  ys=%d  Is=%d\n",
        //         b, n, i, j, i_tilde,
        //         (int)ref, (int)ret,
        //         Is[b * N * L_short + n * L_short + i_tilde]
        //     );
        ////////////
        if (((ref == ret) && (ret != pad) && (ret != pad)))
            atomicMin(&V[b * N * L_ref * L_short + n * L_ref * L_short + j * L_short + i], i_tilde);
    }
}

__global__ void get_G_len(const int *G_offset, const int b, int *G_len)
{
    G_len[0] = G_offset[b + 1] - G_offset[b];
}

__device__ __forceinline__ void update_forward_sent(
    float * __restrict__ D,
    int * __restrict__ D_idx,
    const int idx_src,
    const int idx_tgt,
    const float cost)
{
    int* address_as_i = (int*) &D[idx_tgt];
    int old = *address_as_i, assumed;
    float val = D[idx_src] + cost;
    // float local_max;
    do {
        assumed = old;
        old = atomicCAS(
            address_as_i,
            assumed,
            __float_as_int(fmaxf(val, __int_as_float(assumed))));
        
    } while (assumed != old);

    __syncthreads();
    // TODO verify
    if (val == D[idx_tgt])
    {
        // D_idx[idx_tgt] = idx_src;
        // atomicMin(&D_idx[idx_tgt], idx_src);
        atomicMax(&D_idx[idx_tgt], idx_src);
        //////////// DEBUG
        // int L_short = 5;
        // printf(
        //     "D_idx[%d,%d,%d,%d] = (%d,%d,%d,%d) %f\n",
        //     (idx_tgt / 2 / (L_short + 1) / (L_short + 1)), (idx_tgt / 2 / (L_short + 1)) % (L_short + 1), (idx_tgt / 2) % (L_short + 1), idx_tgt % 2,
        //     (idx_src / 2 / (L_short + 1) / (L_short + 1)), (idx_src / 2 / (L_short + 1)) % (L_short + 1), (idx_src / 2) % (L_short + 1), idx_src % 2,
        //     val
        // );
        ////////////
    }
}

__global__ void forward_step_kernel(
    const int batch,
    const int m,
    const int * __restrict__ lens_short,
    const int * __restrict__ G,
    const int * __restrict__ G_offset,
    const int * __restrict__ V,
    const int N,
    const int L_N,
    const int L_short,
    const int L_ref,
    float * __restrict__ D,
    int * __restrict__ D_idx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ((idx < 2 * L_N) && ((idx == 0 && m == 0) || (m != 0 && D_idx[m * 2 * L_N + idx] != -1)))
    {
        const int k = G[G_offset[batch] + m];
        const int n = k % N;
        const int j = k / N;

        // case no G_m edge
        const bool same_j =  ((m > 0) && (G[G_offset[batch] + m - 1] / N == j));
        const int b = idx % 2;
        const int no_egde_idx = idx + b * ((int)same_j - 1);
        /////// DEBUG
        // if (N == 2)
        // {
        //     printf(
        //         "%d (%d,%d,%d,%d) 0 -> (%d,%d,%d,%d) D=%f\n",
        //         batch,
        //         m, (idx / 2 / (L_short + 1)) % (L_short + 1), (idx / 2) % (L_short + 1), b,
        //         m + 1, (no_egde_idx / 2 / (L_short + 1)) % (L_short + 1), (no_egde_idx / 2) % (L_short + 1), no_egde_idx % 2,
        //         D[m * 2 * L_N + idx]
        //     );
        // }
        ///////
        // send forward to no_edge_idx
        update_forward_sent(D, D_idx, m * 2 * L_N + idx, (m + 1) * 2 * L_N + no_egde_idx, 0);

        // case G_m chosen
        int L_n = L_N * 2;
        for (int nn = 0; nn <= n; nn++)
            L_n /= (L_short + 1);
        int i_n = (idx / L_n) % (L_short + 1);
        int i_star = V[batch * N * L_short * L_ref + n * L_short * L_ref + j * L_short + i_n];
        if (i_star != L_short) // if edge exists
        {
            const int edge_idx = idx + (i_star + 1 - i_n) * L_n + 1 - b;
            /////// DEBUG
            // if (N == 2 && b == 0 && m == 1 && ((idx / 2 / (L_short + 1)) % (L_short + 1) == 0) && ((idx / 2) % (L_short + 1) == 0))
            // {
            //     printf(
            //         "%d (%d,%d,%d,%d) %d -> (%d,%d,%d,%d) i_n=%d, i_*=%d=V[%d,%d,%d,%d]\n",
            //         batch,
            //         m, (idx / 2 / (L_short + 1)) % (L_short + 1), (idx / 2) % (L_short + 1), b,
            //         (int)((b == 0) || !same_j),
            //         m + 1, (edge_idx / 2 / (L_short + 1)) % (L_short + 1), (edge_idx / 2) % (L_short + 1), 1,
            //         i_n, i_star,
            //         batch, n, j, i_n
            //     );
            // }
            // if (N == 2)
            // {
            //     printf(
            //         "%d (%d,%d,%d,%d) %d|1 -> (%d,%d,%d,%d) \t D=%f\n",
            //         batch,
            //         m, (idx / 2 / (L_short + 1)) % (L_short + 1), (idx / 2) % (L_short + 1), b,
            //         (int)((b == 0) || !same_j),
            //         m + 1, (edge_idx / 2 / (L_short + 1)) % (L_short + 1), (edge_idx / 2) % (L_short + 1), edge_idx % 2,
            //         D[m * 2 * L_N + idx]
            //     );
            // }
            ///////
            // send forward to edge_idx
            update_forward_sent(D, D_idx, m * 2 * L_N + idx, (m + 1) * 2 * L_N + edge_idx, (float)((b == 0) || !same_j) + 0.00001);
            /////// DEBUG
            // if (N == 2)
            // {
            //     __syncthreads();
            //     printf(
            //         "%d (%d,%d,%d,%d) %d -> D[%d,%d,%d,%d]=%f\n",
            //         batch,
            //         (D_idx[(m + 1) * 2 * L_N + edge_idx] / 2 / (L_short + 1) / (L_short + 1)),
            //         (D_idx[(m + 1) * 2 * L_N + edge_idx] / 2 / (L_short + 1)) % (L_short + 1),
            //         (D_idx[(m + 1) * 2 * L_N + edge_idx] / 2) % (L_short + 1),
            //         D_idx[(m + 1) * 2 * L_N + edge_idx] % 2,
            //         m + 1, (edge_idx / 2 / (L_short + 1)) % (L_short + 1), (edge_idx / 2) % (L_short + 1), edge_idx % 2,
            //         D[(m + 1) * 2 * L_N + edge_idx]
            //     );
            // }
            ///////
        }
    }
}

__global__ void backward_kernel(
    const int batch,
    const int * __restrict__ Is,
    const int * __restrict__ lens_short,
    const int * __restrict__ G,
    const int * __restrict__ G_offset,
    const int * __restrict__ D_idx,
    const int N,
    const int L_N,
    const int L_short,
    const int L_max,
    int idx,
    long * __restrict__ graph_left,
    long * __restrict__ graph_right,
    const long pad,
    const long unk)
{
    int last_idx = idx;
    int delta;
    int i_n;
    int n;
    int j;
    int G_len = G_offset[batch + 1] - G_offset[batch];
    /////// DEBUG
    // if (N == 2)
    // {
    //     printf(
    //         "%d (%d,%d,%d,%d) === (%d,%d,%d,%d)\n",
    //         batch,
    //         (last_idx / 2 / (L_short + 1) / (L_short + 1)),
    //         (last_idx / 2 / (L_short + 1)) % (L_short + 1),
    //         (last_idx / 2) % (L_short + 1),
    //         last_idx % 2,
    //         (idx / 2 / (L_short + 1) / (L_short + 1)),
    //         (idx / 2 / (L_short + 1)) % (L_short + 1),
    //         (idx / 2) % (L_short + 1),
    //         idx % 2
    //     );
    // }
    ///////
    // backward + fill
    int m = G_len - 1;
    while (idx > 0)
    {
        last_idx = idx;
        idx = D_idx[last_idx];
        if (idx < 0)
            break;
        /////// DEBUG
        // if (N == 2)
        // {
        //     printf(
        //         "%d (%d,%d,%d,%d) -> (%d,%d,%d,%d)\n",
        //         batch,
        //         (last_idx / 2 / (L_short + 1) / (L_short + 1)),
        //         (last_idx / 2 / (L_short + 1)) % (L_short + 1),
        //         (last_idx / 2) % (L_short + 1),
        //         last_idx % 2,
        //         (idx / 2 / (L_short + 1) / (L_short + 1)),
        //         (idx / 2 / (L_short + 1)) % (L_short + 1),
        //         (idx / 2) % (L_short + 1),
        //         idx % 2
        //     );
        // }
        ///////

        delta = (last_idx - idx - 2 * L_N + (idx % 2) - (last_idx % 2)) / 2;
        if (delta != 0)
        {
            int n;
            int L_n = 2;
            for (n = 0; n < N; n++)
            {
                if ((delta != 0) && ((delta / (L_short + 1)) == 0))
                    break;
                delta /= (L_short + 1);
                L_n *= (L_short + 1);
            }
            i_n = ((last_idx / L_n) % (L_short + 1)) - 1;
            j = G[G_offset[batch] + m] / N;
            n = G[G_offset[batch] + m] % N;
            //////////// DEBUG
            // printf(
            //     "ref: %d %d     other: %d %d %d            %d %d    %d ?= %d   L_n=%d L_s+1=%d\n",
            //     batch, j,
            //     batch, n, Is[batch * N * L_short + n * L_short + i_n],
            //     i_n, delta,
            //     last_idx,
            //     idx,
            //     L_n,
            //     (L_short + 1)
            // );
            //////////// 
            graph_left[
                batch * N * L_max
                + n * L_max
                + Is[batch * N * L_short + n * L_short + i_n]
            ] = 1;
            graph_right[
                batch * N * L_max
                + n * L_max
                + j
            ] = 1;
        }
        m--;
    }
}

__global__ void get_last_index(
    const int * __restrict__ lens_short,
    const int batch,
    const int N,
    const int L_short,
    int L_j,
    int * last_index)
{
    for (int n = 0; n < N; n++)
    {
        L_j /= (L_short + 1);
        last_index[0] += lens_short[batch * N + n] * L_j * 2;
    }
}

torch::Tensor GetMaskCuda(
    torch::Tensor& y_ref,
    torch::Tensor& ys,
    const int pad)
{
    const auto batch_size = y_ref.size(0);
    const auto ref_length = y_ref.size(1);
    const auto N = ys.size(1);
    const auto length = ys.size(2);
    at::TensorOptions options(y_ref.device());
    options = options
        .dtype(torch::kInt32)
        .layout(y_ref.layout());
    auto mask = torch::zeros({batch_size, N, length}, options);
    auto stream = at::cuda::getCurrentCUDAStream(y_ref.device().index());
    auto BLOCK = (batch_size * ref_length * N * length + THREAD - 1) / THREAD;
    AT_DISPATCH_INTEGRAL_TYPES(y_ref.scalar_type(), "get_mask_kernel", ([&] {
        get_mask_kernel<scalar_t><<<BLOCK, THREAD, 0, stream>>>(
            y_ref.data_ptr<scalar_t>(),
            ys.data_ptr<scalar_t>(),
            mask.data_ptr<int>(),
            batch_size,
            ref_length,
            N,
            length,
            pad
        );
    }));
    return mask;
}

torch::Tensor GetRefMaskCuda(
    torch::Tensor& y_ref,
    torch::Tensor& ys,
    torch::Tensor& Is,
    const int max_short_len,
    const int pad)
{
    const auto batch_size = y_ref.size(0);
    const auto ref_length = y_ref.size(1);
    const auto N = ys.size(1);
    at::TensorOptions options(y_ref.device());
    options = options
        .dtype(torch::kInt32)
        .layout(y_ref.layout());
    auto mask = torch::zeros({batch_size, ref_length, N}, options);
    auto stream = at::cuda::getCurrentCUDAStream(y_ref.device().index());
    auto BLOCK = (batch_size * ref_length * N * max_short_len + THREAD - 1) / THREAD;
    AT_DISPATCH_INTEGRAL_TYPES(y_ref.scalar_type(), "get_ref_mask_kernel", ([&] {
        get_ref_mask_kernel<scalar_t><<<BLOCK, THREAD, 0, stream>>>(
            y_ref.data_ptr<scalar_t>(),
            ys.data_ptr<scalar_t>(),
            Is.data_ptr<scalar_t>(),
            mask.data_ptr<int>(),
            batch_size,
            ref_length,
            N,
            max_short_len,
            ys.size(2),
            pad
        );
    }));
    return mask;
}

torch::Tensor BuildMinPreferenceCuda(
    torch::Tensor& y_ref,
    torch::Tensor& ys,
    torch::Tensor& Is,
    const int max_short_len,
    const int pad)
{
    const auto batch_size = y_ref.size(0);
    const auto ref_length = y_ref.size(1);
    const auto N = ys.size(1);
    at::TensorOptions options(y_ref.device());
    options = options
        .dtype(torch::kInt32)
        .layout(y_ref.layout());
    auto V = torch::full({batch_size, N, ref_length, max_short_len}, max_short_len, options);
    auto stream = at::cuda::getCurrentCUDAStream(y_ref.device().index());
    auto BLOCK = (batch_size * ref_length * N * max_short_len * max_short_len + THREAD - 1) / THREAD;
    // std::cerr << batch_size << ", " << ref_length << ", "<< N << ", " << max_short_len << std::endl;
    AT_DISPATCH_INTEGRAL_TYPES(y_ref.scalar_type(), "build_min_preference_kernel", ([&] {
        build_min_preference_kernel<scalar_t><<<BLOCK, THREAD, 0, stream>>>(
            y_ref.data_ptr<scalar_t>(),
            ys.data_ptr<scalar_t>(),
            Is.data_ptr<scalar_t>(),
            V.data_ptr<int>(),
            batch_size,
            ref_length,
            N,
            max_short_len,
            ys.size(2),
            pad
        );
    }));
    return V;
}

void getOpsFromSingle(
    const int b,
    const int *Is,
    const int *lens_short,
    const int *G,
    const int *G_offset,
    const int *V,
    const int L_short,
    const int L_ref,
    const int L_max,
    const int N,
    long *graph_left,
    long *graph_right,
    const long pad,
    const long unk)
{
    float *D;
    int *D_idx;
    int L_N = 1;
    int *d_G_len;
    cudaMalloc((void **) &d_G_len, sizeof(int));
    get_G_len<<<1, 1>>>(G_offset, b, d_G_len);
    int *h_G_len = (int *)malloc(sizeof(int));
    cudaMemcpy(h_G_len, d_G_len, sizeof(int), cudaMemcpyDeviceToHost);
    int G_len = *h_G_len;
    free(h_G_len);
    cudaFree(d_G_len);
    for (int n = 0; n < N; n++)
        L_N *= (L_short + 1);
    cudaMalloc((void **) &D, ((G_len + 1) * L_N * 2) * sizeof(float));
    cudaMalloc((void **) &D_idx, ((G_len + 1) * L_N * 2) * sizeof(int));
    cudaMemset(D, -1, ((G_len + 1) * L_N * 2) * sizeof(float));
    cudaMemset(D, 0, sizeof(float));
    cudaMemset(D_idx, -1, ((G_len + 1) * L_N * 2) * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // FORWARD
    for (int m = 0; (m < G_len) && (m < L_ref * L_short); m++)
    {
        // std::cerr << m << " ------------------------------" << std::endl;
        auto BLOCK = (L_N * 2 + THREAD - 1) / THREAD;
        forward_step_kernel<<<BLOCK, THREAD, 0, stream>>>(
            b,
            m,
            lens_short,
            G,
            G_offset,
            V,
            N,
            L_N,
            L_short,
            L_ref,
            D,
            D_idx);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
            std::cerr << "CUDA Runtime Error in Forward " << cudaGetErrorString(err)
                      << std::endl;
    }

    // BACKWARD
    int last_idx = G_len * L_N * 2 + 1;
    int L_j = L_N;
    int* d_last_index;
    cudaMalloc((void **) &d_last_index, sizeof(int));
    cudaMemset(d_last_index, 0, sizeof(int));

    get_last_index<<<1, 1, 0, stream>>>(
        lens_short,
        b,
        N,
        L_short,
        L_j,
        d_last_index);
    int* h_last_index = (int *)malloc(sizeof(int));
    cudaMemcpy(h_last_index, d_last_index, sizeof(int), cudaMemcpyDeviceToHost);
    last_idx += *h_last_index;
    free(h_last_index);

    backward_kernel<<<1, 1, 0, stream>>>(
        b,
        Is,
        lens_short,
        G,
        G_offset,
        D_idx,
        N,
        L_N,
        L_short,
        L_max,
        last_idx,
        graph_left,
        graph_right,
        pad,
        unk);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cerr << "CUDA Runtime Error in Backward " << cudaGetErrorString(err)
                  << std::endl;
    
    
    cudaFree(d_last_index);
    cudaFree(D);
    cudaFree(D_idx);
}