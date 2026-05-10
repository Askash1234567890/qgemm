#include <torch/extension.h>

torch::Tensor quantized_gemm_forward(
    torch::Tensor A,
    torch::Tensor B,
    float scale_a,
    float scale_b,
    int   tile_size);

std::tuple<torch::Tensor, torch::Tensor> quantized_gemm_backward(
    torch::Tensor grad_output,
    torch::Tensor A_dq,
    torch::Tensor B_dq,
    torch::Tensor mask_A,
    torch::Tensor mask_B);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "qgemm: INT8 fake-quant CUDA kernels";

    m.def("forward", &quantized_gemm_forward,
          "INT8 tiled matmul forward (tile_size=16 or 32)",
          py::arg("A"), py::arg("B"),
          py::arg("scale_a"), py::arg("scale_b"),
          py::arg("tile_size") = 32);

    m.def("backward", &quantized_gemm_backward,
          "STE backward: dA=(dY@B_dq.T)*mask_A, dB=(A_dq.T@dY)*mask_B",
          py::arg("grad_output"),
          py::arg("A_dq"), py::arg("B_dq"),
          py::arg("mask_A"), py::arg("mask_B"));
}
