#ifndef tensorflow_model_hpp
#define tensorflow_model_hpp

#include <string>
#include <vector>
#include <stdint.h>

namespace rnnoise {
class TensorflowModel {
public:
    struct Input {
        Input(): data(nullptr) {}
        int size() const {
            int s = 1;
            for (int i = 0; i < dims.size(); i++) {
                s *= dims[i];
            }
            return s;
        }
        
        std::string name;
        std::vector<int64_t> dims;
        const float *data;
    };
    
    struct Output {
        Output(): data(nullptr) {}
        std::string name;
        float *data;
    };
    
    TensorflowModel(const char *pb_data, int pb_data_size);
    ~TensorflowModel();
    void Predict(const Input *inputs, int input_count, const Output *outputs, int output_count);
private:
    void *graph_;
    void *session_;
};
}

#endif /* tensorflow_model_hpp */
