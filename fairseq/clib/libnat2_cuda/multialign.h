#pragma once

#define STRIP_ERROR_MESSAGES

#include <torch/torch.h>
#include <torch/extension.h>

torch::Tensor GetMaskCuda(
    torch::Tensor& y_ref,
    torch::Tensor& ys,
    const int pad);

torch::Tensor GetRefMaskCuda(
    torch::Tensor& y_ref,
    torch::Tensor& ys,
    torch::Tensor& Is,
    const int max_short_len,
    const int pad);

torch::Tensor BuildMinPreferenceCuda(
    torch::Tensor& y_ref,
    torch::Tensor& ys,
    torch::Tensor& Is,
    const int max_short_len,
    const int pad);

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
    const long unk);
