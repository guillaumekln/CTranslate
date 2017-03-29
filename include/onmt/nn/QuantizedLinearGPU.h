#pragma once

#include "onmt/nn/Module.h"
#include "onmt/th/Obj.h"
#include "onmt/cuda/Utils.h"

namespace onmt
{
  namespace nn
  {

    template <typename MatFwd, typename MatIn, typename ModelT>
    class QuantizedLinearGPU: public Module<MatFwd>
    {
    public:
      QuantizedLinearGPU(th::Table* data, cublasHandle_t& handle);
      ~QuantizedLinearGPU();

      virtual MatFwd forward_impl(MatFwd& input) const override;

    private:
      void realloc_output(int num_batches) const;

      cublasHandle_t& _handle;

      MatIn _s;
      MatIn _bias;

      int8_t* _weight_device;

      int _output_size;
      int _input_size;

      // Reuse allocated gemm output.
      mutable int32_t* _output_device;
      mutable size_t _allocated_batches;
    };

  }
}

#include "onmt/nn/QuantizedLinearGPU.hxx"
