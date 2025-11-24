// Copyright © 2025 Apple Inc.

#include <algorithm>
#include <complex>
#include <numeric>

#include <cuda_runtime.h>
#include <cufft.h>

#include "mlx/allocator.h"
#include "mlx/backend/cuda/cuda_utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/primitives.h"
#include "mlx/ops.h"

namespace mlx::core {

namespace {

// Utility to check cuFFT errors
void check_cufft_error(const char* name, cufftResult err) {
  if (err != CUFFT_SUCCESS) {
    std::string err_msg;
    switch (err) {
      case CUFFT_INVALID_PLAN: err_msg = "CUFFT_INVALID_PLAN"; break;
      case CUFFT_ALLOC_FAILED: err_msg = "CUFFT_ALLOC_FAILED"; break;
      case CUFFT_INVALID_TYPE: err_msg = "CUFFT_INVALID_TYPE"; break;
      case CUFFT_INVALID_VALUE: err_msg = "CUFFT_INVALID_VALUE"; break;
      case CUFFT_INTERNAL_ERROR: err_msg = "CUFFT_INTERNAL_ERROR"; break;
      case CUFFT_EXEC_FAILED: err_msg = "CUFFT_EXEC_FAILED"; break;
      case CUFFT_SETUP_FAILED: err_msg = "CUFFT_SETUP_FAILED"; break;
      case CUFFT_INVALID_SIZE: err_msg = "CUFFT_INVALID_SIZE"; break;
      case CUFFT_UNALIGNED_DATA: err_msg = "CUFFT_UNALIGNED_DATA"; break;
      default: err_msg = "CUFFT_UNKNOWN_ERROR"; break;
      // Handle cases where err might be an int that doesn't map directly if needed
    }
    throw std::runtime_error(
        std::string("[cuFFT Error] ") + name + " failed with error: " + err_msg);
  }
}

#define CHECK_CUFFT_ERROR(cmd) check_cufft_error(#cmd, (cmd))

// CUDA Kernel for scaling (Normalization)
template <typename T>
__global__ void scale_kernel(T* data, size_t n, float scale) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx] *= scale;
  }
}

template <>
__global__ void scale_kernel<cuComplex>(cuComplex* data, size_t n, float scale) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    data[idx].x *= scale;
    data[idx].y *= scale;
  }
}

} // namespace

void FFT::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& s = stream();
  auto& encoder = cu::get_command_encoder(s);
  
  auto& in = inputs[0];
  
  // 1. Identify dimensions
  std::vector<int> batch_dims;
  std::vector<int> fft_dims(axes_.begin(), axes_.end());
  
  std::vector<bool> is_fft_dim(in.ndim(), false);
  for (int ax : axes_) is_fft_dim[ax] = true;
  
  for (int i = 0; i < in.ndim(); ++i) {
    if (!is_fft_dim[i]) batch_dims.push_back(i);
  }

  // 2. Permute: [Batch Dims..., FFT Dims...]
  std::vector<int> perm = batch_dims;
  perm.insert(perm.end(), fft_dims.begin(), fft_dims.end());

  array in_transposed = transpose(in, perm, s);
  array in_contiguous = copy(in_transposed, s);

  // 3. Calculate Intermediate Output Shape
  std::vector<int> out_transposed_vec(in_contiguous.shape().begin(), in_contiguous.shape().end());
  
  if (in.dtype() == float32 && out.dtype() == complex64) { // R2C
    out_transposed_vec.back() = in_contiguous.shape().back() / 2 + 1;
  } else if (in.dtype() == complex64 && out.dtype() == float32) { // C2R
    out_transposed_vec.back() = (in_contiguous.shape().back() - 1) * 2;
  }
  
  Shape out_transposed_shape(out_transposed_vec.begin(), out_transposed_vec.end());

  // Allocate intermediate buffer
  array out_contiguous(out_transposed_shape, out.dtype(), nullptr, {});
  out_contiguous.set_data(allocator::malloc(out_contiguous.nbytes()));

  // 4. Setup cuFFT Plan
  int rank = axes_.size();
  std::vector<int> n(rank);
  
  for (int i = 0; i < rank; ++i) {
    int dim_idx = batch_dims.size() + i;
    if (out.dtype() == float32 && i == rank - 1) { // C2R Case
        n[i] = out_transposed_vec[dim_idx]; 
    } else {
        n[i] = in_contiguous.shape(dim_idx);
    }
  }

  int batch = 1;
  for (int dim : batch_dims) {
    batch *= in.shape(dim);
  }
  
  long long idist = in_contiguous.size() / batch;
  long long odist = out_contiguous.size() / batch;

  cufftHandle plan;
  cufftType type;
  if (in.dtype() == complex64 && out.dtype() == complex64) type = CUFFT_C2C;
  else if (in.dtype() == float32 && out.dtype() == complex64) type = CUFFT_R2C;
  else if (in.dtype() == complex64 && out.dtype() == float32) type = CUFFT_C2R;
  else throw std::runtime_error("[FFT] Invalid dtype combination.");

  encoder.set_input_array(in_contiguous);
  encoder.set_output_array(out_contiguous);

  CHECK_CUFFT_ERROR(cufftPlanMany(
      &plan, rank, n.data(),
      nullptr, 1, idist,
      nullptr, 1, odist,
      type, batch));

  CHECK_CUFFT_ERROR(cufftSetStream(plan, encoder.stream()));

  // 5. Execute FFT
  if (type == CUFFT_C2C) {
    int direction = inverse_ ? CUFFT_INVERSE : CUFFT_FORWARD;
    CHECK_CUFFT_ERROR(cufftExecC2C(
        plan,
        (cufftComplex*)in_contiguous.data<complex64_t>(),
        (cufftComplex*)out_contiguous.data<complex64_t>(),
        direction));
  } else if (type == CUFFT_R2C) {
    CHECK_CUFFT_ERROR(cufftExecR2C(
        plan,
        (cufftReal*)in_contiguous.data<float>(),
        (cufftComplex*)out_contiguous.data<complex64_t>()));
  } else { // C2R
    CHECK_CUFFT_ERROR(cufftExecC2R(
        plan,
        (cufftComplex*)in_contiguous.data<complex64_t>(),
        (cufftReal*)out_contiguous.data<float>()));
  }

  // 6. Scaling Kernel
  if (inverse_) {
    float scale = 1.0f;
    size_t nelem = 1;
    for (auto s : n) nelem *= s;
    scale /= nelem;

    size_t total_elements = out_contiguous.size();
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;

    if (out.dtype() == complex64) {
      scale_kernel<cuComplex><<<blocks, threads, 0, encoder.stream()>>>(
          (cuComplex*)out_contiguous.data<complex64_t>(), total_elements, scale);
    } else {
      scale_kernel<float><<<blocks, threads, 0, encoder.stream()>>>(
          out_contiguous.data<float>(), total_elements, scale);
    }
  }

  CHECK_CUFFT_ERROR(cufftDestroy(plan));

  // 7. Transpose Back and Copy to Out
  std::vector<int> inv_perm(in.ndim());
  for (int i = 0; i < perm.size(); ++i) {
    inv_perm[perm[i]] = i;
  }

  // This creates a strided view
  array out_reordered = transpose(out_contiguous, inv_perm, s);

  // 'copy' collapses the strided view into a new contiguous array
  array final_temp = copy(out_reordered, s);

  // Now we allocate 'out' and move the bits from final_temp to 'out'
  // This preserves 'out's identity (graph node) while giving it the correct data.
  out.set_data(allocator::malloc(out.nbytes()));

  encoder.set_input_array(final_temp);
  encoder.set_output_array(out);

  if (cudaMemcpyAsync(
          out.data<uint8_t>(),
          final_temp.data<uint8_t>(),
          out.nbytes(),
          cudaMemcpyDeviceToDevice,
          encoder.stream()) != cudaSuccess) {
     throw std::runtime_error("[FFT] Final copy failed");
  }
}

} // namespace mlx::core