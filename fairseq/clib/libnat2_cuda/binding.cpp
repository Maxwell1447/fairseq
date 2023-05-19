#include "multialign.h"
#include <pybind11/pybind11.h>
#include <thread>
#include <vector>
#include <memory>
#include <utility>
#include <iostream>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(x.device() == torch::kCPU, #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT_CUDA(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_INPUT_CPU(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)

namespace py = pybind11;

torch::Tensor GetMask(
    torch::Tensor y_ref,
    torch::Tensor ys,
    const int pad)
{
    CHECK_INPUT_CUDA(y_ref);
    CHECK_INPUT_CUDA(ys);
    return GetMaskCuda(y_ref, ys, pad);
}

torch::Tensor GetRefMask(
    torch::Tensor y_ref,
    torch::Tensor ys,
    torch::Tensor Is,
    const int max_short_len,
    const int pad)
{
    CHECK_INPUT_CUDA(y_ref);
    CHECK_INPUT_CUDA(ys);
    CHECK_INPUT_CUDA(Is);
    return GetRefMaskCuda(y_ref, ys, Is, max_short_len, pad);
}

torch::Tensor BuildMinPreference(
    torch::Tensor y_ref,
    torch::Tensor ys,
    torch::Tensor Is,
    const int max_short_len,
    const int pad)
{
    CHECK_INPUT_CUDA(y_ref);
    CHECK_INPUT_CUDA(ys);
    CHECK_INPUT_CUDA(Is);
    return BuildMinPreferenceCuda(y_ref, ys, Is, max_short_len, pad);
}


/////////////////////////////////////////////////////////
void getOpsFromBatch(
    const int *Is,
    const int *lens_short,
    const int *G,
    const int *G_offsets,
    const int *V,
    const int L_short,
    const int L_ref,
    const int L_max,
    const int B,
    const int N,
    long *graph_left,
    long *graph_right,
    const long pad,
    const long unk)
{
  std::vector<std::thread> threads(B);
  std::thread t;
  for (int b = 0; b < B; ++b)
  {
    getOpsFromSingle(
        b,
        Is,
        lens_short,
        G,
        G_offsets,
        V,
        L_short,
        L_ref,
        L_max,
        N,
        graph_left,
        graph_right,
        pad,
        unk);
    // t = std::thread(
    //     getOpsFromSingle,
    //     &ys[b * N * L],
    //     &y[b * L_ref],
    //     L_short,
    //     L_ref,
    //     N,
    //     &del[b * N * L_max],
    //     &ins[b * N * (L_max - 1)],
    //     &cmb[b * N * L_max],
    //     &s_del[b * N * L_max],
    //     &s_plh[b * N * L_max],
    //     &s_cmb[b * L_max],
    //     pad,
    //     unk);
    // threads.push_back(std::move(t));
  }
//   for (std::thread &t : threads)
//   {
//     t.join();
//   }
}

class EditOpsBatchCuda
{
public:
  const int B;
  const int N;
  const int L_ref;
  const int L;
  const int L_short;

  torch::Tensor graph_left;
  torch::Tensor graph_right;

  EditOpsBatchCuda() : B(0), N(0), L_ref(0), L(0), L_short(0) {};
  EditOpsBatchCuda(
      torch::Tensor y,
      torch::Tensor ys,
      torch::Tensor Is,
      torch::Tensor lens_short,
      torch::Tensor G,
      torch::Tensor G_offsets,
      torch::Tensor V,
      const int max_len_short,
      const int max_len,
      const long pad,
      const long unk) : 
        B(ys.size(0)),
        N(ys.size(1)),
        L_ref(y.size(1)),
        L(ys.size(2)),
        L_short(max_len_short)
  {
    auto options = torch::TensorOptions()
                       .layout(ys.layout())
                       .dtype(torch::kI64)
                       .device(ys.device());

    graph_left = torch::zeros({B, N, max_len}, options);
    graph_right = torch::zeros({B, N, max_len}, options);

    getOpsFromBatch(
        Is.data_ptr<int>(),
        lens_short.data_ptr<int>(),
        G.data_ptr<int>(),
        G_offsets.data_ptr<int>(),
        V.data_ptr<int>(),
        L_short,
        L_ref,
        max_len,
        B,
        N,
        graph_left.data_ptr<long>(),
        graph_right.data_ptr<long>(),
        pad,
        unk);
  };
    torch::Tensor getGraphLeft() { return graph_left; };
    torch::Tensor getGraphRight() { return graph_right; };
};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("get_mask", &GetMask, "get mask");
    m.def("get_ref_mask", &GetRefMask, "get mask");
    m.def("build_min_preference", &BuildMinPreference, "build min preference mask");
    py::class_<EditOpsBatchCuda>(m, "MultiLevEditOpsCuda")
      .def(py::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, const int, const int, const long, const long>())
      .def(py::init<>())
      .def("get_graph_left", &EditOpsBatchCuda::getGraphLeft)
      .def("get_graph_right", &EditOpsBatchCuda::getGraphRight);
}