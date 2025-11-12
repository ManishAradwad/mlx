// Copyright © 2025 Apple Inc.

#include <cufft.h>
#include <numeric>

#include "mlx/allocator.h"
#include "mlx/backend/cuda/cuda_utils.h"
#include "mlx/backend/cuda/device.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

// Helper function to check cuFFT errors
void check_cufft_error(const char* name, cufftResult err) {
  if (err != CUFFT_SUCCESS) {
    const char* err_msg;
    switch (err) {
      case CUFFT_INVALID_PLAN:
        err_msg = "CUFFT_INVALID_PLAN";
        break;
      case CUFFT_ALLOC_FAILED:
        err_msg = "CUFFT_ALLOC_FAILED";
        break;
      case CUFFT_INVALID_TYPE:
        err_msg = "CUFFT_INVALID_TYPE";
        break;
      case CUFFT_INVALID_VALUE:
        err_msg = "CUFFT_INVALID_VALUE";
        break;
      case CUFFT_INTERNAL_ERROR:
        err_msg = "CUFFT_INTERNAL_ERROR";
        break;
      case CUFFT_EXEC_FAILED:
        err_msg = "CUFFT_EXEC_FAILED";
        break;
      case CUFFT_SETUP_FAILED:
        err_msg = "CUFFT_SETUP_FAILED";
        break;
      case CUFFT_INVALID_SIZE:
        err_msg = "CUFFT_INVALID_SIZE";
        break;
      case CUFFT_UNALIGNED_DATA:
        err_msg = "CUFFT_UNALIGNED_DATA";
        break;
      case CUFFT_INCOMPLETE_PARAMETER_LIST:
        err_msg = "CUFFT_INCOMPLETE_PARAMETER_LIST";
        break;
      case CUFFT_INVALID_DEVICE:
        err_msg = "CUFFT_INVALID_DEVICE";
        break;
      case CUFFT_PARSE_ERROR:
        err_msg = "CUFFT_PARSE_ERROR";
        break;
      case CUFFT_NO_WORKSPACE:
        err_msg = "CUFFT_NO_WORKSPACE";
        break;
      case CUFFT_NOT_IMPLEMENTED:
        err_msg = "CUFFT_NOT_IMPLEMENTED";
        break;
      case CUFFT_LICENSE_ERROR:
        err_msg = "CUFFT_LICENSE_ERROR";
        break;
      case CUFFT_NOT_SUPPORTED:
        err_msg = "CUFFT_NOT_SUPPORTED";
        break;
      default:
        err_msg = "CUFFT_UNKNOWN_ERROR";
        break;
    }
    throw std::runtime_error(
        std::string("[cuFFT Error] ") + name + " failed with error: " +
        err_msg);
  }
}

#define CHECK_CUFFT_ERROR(cmd) check_cufft_error(#cmd, (cmd))

} // namespace

void FFT::eval_gpu(const std::vector<array>& inputs, array& out) {
  auto& in = inputs[0];
  auto& s = stream();
  
  // Get CUDA encoder/device
  auto& d = cu::device(s.device);
  auto& encoder = cu::get_command_encoder(s);
  
  // Allocate output memory
  out.set_data(allocator::malloc(out.nbytes()));
  
  // Set input and output arrays for the encoder (keeps them alive)
  encoder.set_input_array(in);
  encoder.set_output_array(out);
  
  // Get the FFT size from the axes
  int n = out.dtype() == float32 ? out.shape(axes_[0]) : in.shape(axes_[0]);
  
  // Calculate batch size: total elements / n
  size_t total_size = in.dtype() == float32 ? in.size() : 
                      (out.dtype() == float32 ? out.size() : in.size());
  int batch = total_size / n;
  
  // Create cuFFT plan
  cufftHandle plan;
  cufftResult result;
  
  // Determine the transform type and create appropriate plan
  if (in.dtype() == complex64 && out.dtype() == complex64) {
    // Complex to Complex transform
    result = cufftPlan1d(&plan, n, CUFFT_C2C, batch);
    CHECK_CUFFT_ERROR(result);
    
    // Set the stream for this plan
    CHECK_CUFFT_ERROR(cufftSetStream(plan, encoder.stream()));
    
    // Execute C2C FFT
    int direction = inverse_ ? CUFFT_INVERSE : CUFFT_FORWARD;
    result = cufftExecC2C(
        plan,
        reinterpret_cast<cufftComplex*>(in.data<complex64_t>()),
        reinterpret_cast<cufftComplex*>(out.data<complex64_t>()),
        direction);
    CHECK_CUFFT_ERROR(result);
    
  } else if (in.dtype() == float32 && out.dtype() == complex64) {
    // Real to Complex transform
    result = cufftPlan1d(&plan, n, CUFFT_R2C, batch);
    CHECK_CUFFT_ERROR(result);
    
    // Set the stream for this plan
    CHECK_CUFFT_ERROR(cufftSetStream(plan, encoder.stream()));
    
    // Execute R2C FFT
    result = cufftExecR2C(
        plan,
        const_cast<float*>(in.data<float>()),
        reinterpret_cast<cufftComplex*>(out.data<complex64_t>()));
    CHECK_CUFFT_ERROR(result);
    
  } else if (in.dtype() == complex64 && out.dtype() == float32) {
    // Complex to Real transform (inverse)
    result = cufftPlan1d(&plan, n, CUFFT_C2R, batch);
    CHECK_CUFFT_ERROR(result);
    
    // Set the stream for this plan
    CHECK_CUFFT_ERROR(cufftSetStream(plan, encoder.stream()));
    
    // Execute C2R FFT
    result = cufftExecC2R(
        plan,
        reinterpret_cast<cufftComplex*>(in.data<complex64_t>()),
        out.data<float>());
    CHECK_CUFFT_ERROR(result);
    
  } else {
    cufftDestroy(plan);
    throw std::runtime_error(
        "[FFT] Received unexpected input and output type combination.");
  }
  
  // Clean up the plan
  CHECK_CUFFT_ERROR(cufftDestroy(plan));
}

} // namespace mlx::core
