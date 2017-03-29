#pragma once

#include "onmt/StorageLoader.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename ModelT>
    QuantizedLinearGPU<MatFwd, MatIn, ModelT>::QuantizedLinearGPU(th::Table* data, cublasHandle_t& handle)
      : Module<MatFwd>("onmt.QuantizedLinearGPU")
      , _handle(handle)
      , _s(StorageLoader<MatIn, ModelT>::get_matrix(data, "s"))
      , _bias(StorageLoader<MatIn, ModelT>::get_matrix(data, "bias"))
      , _weight_device(nullptr)
      , _output_device(nullptr)
      , _allocated_batches(0)
    {
      const int8_t* weight = StorageLoader<int8_t, unsigned char>::get_matrix(data,
                                                                              "weight",
                                                                              _output_size,
                                                                              _input_size);
      _weight_device = cuda::to_device<int8_t>(weight, _input_size, _output_size);
    }

    template <typename MatFwd, typename MatIn, typename ModelT>
    QuantizedLinearGPU<MatFwd, MatIn, ModelT>::~QuantizedLinearGPU()
    {
      CUDA_CHECK(cudaFree(_weight_device));
      CUDA_CHECK(cudaFree(_output_device));
    }

    template <typename MatFwd, typename MatIn, typename ModelT>
    void QuantizedLinearGPU<MatFwd, MatIn, ModelT>::realloc_output(int num_batches) const
    {
      CUDA_CHECK(cudaFree(_output_device));
      _output_device = cuda::to_device<int32_t>(nullptr, _output_size, num_batches);
      _allocated_batches = num_batches;
    }

    template <typename MatFwd, typename MatIn, typename ModelT>
    MatFwd QuantizedLinearGPU<MatFwd, MatIn, ModelT>::forward_impl(MatFwd& input) const
    {
      const int batch_size = input.rows();

      if (batch_size > _allocated_batches)
        this->realloc_output(batch_size);

      std::vector<ModelT> maxs;

      for (int i = 0; i < input.rows(); ++i)
      {
        maxs.push_back(input.row(i).array().abs().maxCoeff());
        input.row(i) *= 127 / maxs.back();
      }

      void* quantized_input_device = cuda::to_device<int8_t>(input.template cast<int8_t>().eval().data(), input.cols(), input.rows());

      int32_t alpha = 1;
      int32_t beta = 0;

      CUBLAS_CHECK(cublasGemmEx(_handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                _output_size, batch_size, _input_size,
                                &alpha,
                                _weight_device, CUDA_R_8I, _input_size,
                                quantized_input_device, CUDA_R_8I, _input_size,
                                &beta,
                                _output_device, CUDA_R_32I, _output_size,
                                CUDA_R_32I,
                                CUBLAS_GEMM_DFALT));

      CUDA_CHECK(cudaFree(quantized_input_device));

      Eigen::RowMajorMat<int32_t> output(batch_size, _output_size);
      cuda::to_host<int32_t>(_output_device, output.data(), _output_size, batch_size);

      MatFwd out = output.template cast<float>();
      for (int i = 0; i < out.rows(); ++i)
      {
        out.row(i) = (out.row(i).array() * _s.array() * maxs[i]) / (127 * 127);
        if (_bias.rows() > 0)
          out.row(i) += _bias.transpose();
      }

      return out;
    }


  }
}
