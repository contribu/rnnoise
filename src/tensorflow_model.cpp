#include "tensorflow_model.h"

#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>
#include "tensorflow/c/c_api.h"

namespace {
    void free_null(void *data, size_t length) {}
    void dealloc_tensor_null(void *data, size_t length, void *arg) {}
    
    void throw_tf_exception(const char *message, TF_Status *status) {
        std::stringstream ss;
        ss << message;
        if (status) {
            ss << " " << TF_Message(status);
        }
        throw std::logic_error(ss.str());
    }
    
    struct StatusDeleter {
        void operator ()(TF_Status *ptr) const {
            TF_DeleteStatus(ptr);
        }
    };
    typedef std::unique_ptr<TF_Status, StatusDeleter> StatusPtr;
    
    struct BufferDeleter {
        void operator ()(TF_Buffer *ptr) const {
            TF_DeleteBuffer(ptr);
        }
    };
    typedef std::unique_ptr<TF_Buffer, BufferDeleter> BufferPtr;
}

namespace rnnoise {
    TensorflowModel::TensorflowModel(const char *pb_data, int pb_data_size): graph_(nullptr), session_(nullptr) {
        StatusPtr status(TF_NewStatus());
        
        BufferPtr pb_buf(TF_NewBuffer());
        pb_buf->data = pb_data;
        pb_buf->length = pb_data_size;
        pb_buf->data_deallocator = free_null;
        
        graph_ = TF_NewGraph();
        TF_ImportGraphDefOptions *graph_opts = TF_NewImportGraphDefOptions();
        TF_GraphImportGraphDef((TF_Graph *)graph_, pb_buf.get(), graph_opts, status.get());
        TF_DeleteImportGraphDefOptions(graph_opts);
        
        if (TF_GetCode(status.get()) != TF_OK) {
            throw_tf_exception("TF_GraphImportGraphDef failed", status.get());
        }
        
        TF_SessionOptions *sess_opts = TF_NewSessionOptions();
        session_ = TF_NewSession((TF_Graph *)graph_, sess_opts, status.get());
        TF_DeleteSessionOptions(sess_opts);
        if (TF_GetCode(status.get()) != TF_OK) {
            throw_tf_exception("TF_NewSession failed", status.get());
        }
    }
    
    TensorflowModel::~TensorflowModel() {
        StatusPtr status(TF_NewStatus());
        TF_DeleteGraph((TF_Graph * )graph_);
        TF_DeleteSession((TF_Session *)session_, status.get());
    }
    
    void TensorflowModel::Predict(const Input *inputs, int input_count, const Output *outputs, int output_count) {
        StatusPtr status(TF_NewStatus());
        
        std::vector<TF_Output> input_outputs;
        std::vector<TF_Tensor*> input_tensors;
        for (int i = 0; i < input_count; i++) {
            TF_Operation* input_op = TF_GraphOperationByName((TF_Graph *)graph_, inputs[i].name.c_str());
            if (!input_op) {
                throw_tf_exception(("TF_GraphOperationByName failed " + inputs[i].name).c_str(), nullptr);
            }
            TF_Output input_output = {input_op, 0};
            input_outputs.push_back(input_output);
            
            TF_Tensor *input_tensor = TF_NewTensor(TF_FLOAT, inputs[i].dims.data(), inputs[i].dims.size(), (void *)inputs[i].data, 4 * inputs[i].size(), dealloc_tensor_null, nullptr);
            if (!input_tensor) {
                throw_tf_exception("TF_NewTensor failed", nullptr);
            }
            input_tensors.push_back(input_tensor);
        }
        
#if 0
        size_t pos = 0;
        TF_Operation* oper;
        while ((oper = TF_GraphNextOperation((TF_Graph *)graph_, &pos)) != nullptr) {
            std::cout << "Input: " << TF_OperationName(oper) << "\n";
        }
#endif
    
        std::vector<TF_Output> output_outputs;
        std::vector<TF_Tensor*> output_tensors;
        for (int i = 0; i < output_count; i++) {
            TF_Operation *output_op = TF_GraphOperationByName((TF_Graph *)graph_, outputs[i].name.c_str());
            if (!output_op) {
                throw_tf_exception(("TF_GraphOperationByName failed " + outputs[i].name).c_str(), nullptr);
            }
            TF_Output output_output = {output_op, 0};
            output_outputs.push_back(output_output);
            
            output_tensors.push_back(nullptr);
        }
        
        TF_SessionRun((TF_Session *)session_, nullptr,
                      input_outputs.data(), input_tensors.data(), input_outputs.size(),
                      output_outputs.data(), output_tensors.data(), output_outputs.size(),
                      nullptr, 0, nullptr, status.get());
        for (int i = 0; i < input_count; i++) {
            TF_DeleteTensor(input_tensors[i]);
        }
        if (TF_GetCode(status.get()) != TF_OK) {
            throw_tf_exception("TF_SessionRun failed", status.get());
        }
        
        for (int i = 0; i < output_count; i++) {
            float *tensor_value_ptr = static_cast<float *>(TF_TensorData(output_tensors[i]));
            const int size = TF_TensorByteSize(output_tensors[i]) / sizeof(float);
            for (int j = 0; j < size; j++) {
                outputs[i].data[j] = tensor_value_ptr[j];
            }
            TF_DeleteTensor(output_tensors[i]);
        }
    }
}
