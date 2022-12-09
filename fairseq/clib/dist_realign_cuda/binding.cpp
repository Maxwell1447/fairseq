#include "align.h"
#include <torch/extension.h>
#include <torch/script.h>
#include <vector>

using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(x.device() == torch::kCPU, #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT_CUDA(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_INPUT_CPU(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)


// inline std::vector<int64_t> list2vec(const c10::List<int64_t> list) {
//   std::vector<int64_t> result;
//   result.reserve(list.size());
//   for (size_t i = 0; i < list.size(); i++)
//     result.push_back(list[i]);
//   return result;
// }

torch::Tensor GetGraph(
    torch::Tensor x,
    const int pad
) {
    CHECK_INPUT_CUDA(x);
    return GetGraphCuda(x, pad);
}

std::tuple<torch::Tensor, torch::Tensor> ComputeDist(
    torch::Tensor graph,
    torch::Tensor pos,
    torch::Tensor graph_mask,
    const int p
) {
    CHECK_INPUT_CUDA(graph);
    CHECK_INPUT_CUDA(pos);
    CHECK_INPUT_CUDA(graph_mask);
    return ComputeDistCuda(graph, pos, graph_mask, p);
}

torch::Tensor ComputeDistBackward(
    torch::Tensor arg_out,
    torch::Tensor graph,
    torch::Tensor pos,
    torch::Tensor grad_out,
    const int p
) {
    return ComputeDistBackwardCuda(arg_out, graph, pos, grad_out, p);
}


class ScatterDist : public torch::autograd::Function<ScatterDist> {
public:
  static variable_list forward(
    AutogradContext *ctx,
    Variable pos,
    Variable graph,
    Variable graph_mask,
    const int p
) {

    // index = broadcast(index, src, dim);
    // ctx->saved_data["pos_shape"] = pos.sizes();
    // ctx->saved_data["pos_dim"] = pos.dim();
    std::tuple<torch::Tensor, torch::Tensor> res = ComputeDist(graph, pos, graph_mask, p);
    torch::Tensor out = std::get<0>(res);
    torch::Tensor arg_out = std::get<1>(res);
    at::TensorOptions options(arg_out.device());
    options = options
        .dtype(arg_out.dtype())
        .layout(arg_out.layout());
    torch::Tensor p_tensor =  torch::full({1}, p, options);
    ctx->save_for_backward({pos, arg_out, graph_mask, p_tensor});
    // ctx->mark_non_differentiable({arg_out});

    return {out};
  }

  static variable_list backward(AutogradContext *ctx, variable_list grad_outs) {
    auto grad_out = grad_outs[0];
    auto saved = ctx->get_saved_variables();
    auto pos = saved[0];
    auto arg_out = saved[1];
    auto graph_mask = saved[2];
    // const int p = saved[3].index({0});
    const int p = 2;
    // auto pos_shape = list2vec(ctx->saved_data["pos_shape"].toIntList());
    // pos_shape[pos.dim() - 1] += 1;
    // auto grad_in = torch::zeros(pos_shape, grad_out.options());
    // printf("Entering ComputeDistBackwardCuda\n");
    torch::Tensor grad_in = ComputeDistBackward(arg_out, graph_mask, pos, grad_out, p);
    // grad_in.scatter_(dim, arg_out, grad_out);
    // grad_in = grad_in.narrow(pos.dim() - 1, 0, pos_shape[pos.dim() - 1] - 1);
    return {grad_in, Variable(), Variable(), Variable()};
  }
};


torch::Tensor scatter_dist_lp(
    torch::Tensor graph,
    torch::Tensor pos,
    torch::Tensor graph_mask,
    const int p
) {
    return ScatterDist::apply(pos, graph, graph_mask, p)[0];
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // m.def("get_interpolators", &GetInterpolators, "get interpolators to compute log prob");
    m.def("get_graph", &GetGraph, "get graph tensor");
    m.def("scatter_dist_lp", &scatter_dist_lp, "get graph tensor");
}
