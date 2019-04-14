#include "gtest/gtest.h"

#include <fstream>
#include <vector>
#include "tensorflow_model.h"

TEST(TensorflowModel, SmokeTest) {
    std::ifstream ifs("testmodel.pb", std::ios::binary | std::ios::ate);
    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    EXPECT_TRUE(ifs.read(buffer.data(), size));
    rnnoise::TensorflowModel model(buffer.data(), buffer.size());
    
    std::vector<float> input_buffer(16 * 42);
    std::vector<float> output_denoise_buffer(22);
    std::vector<float> output_val_buffer(1);
    rnnoise::TensorflowModel::Input input;
    input.data = input_buffer.data();
    input.dims.push_back(0);
    input.dims.push_back(16);
    input.dims.push_back(42);
    input.name = "main_input";
    std::vector<rnnoise::TensorflowModel::Output> outputs;
    {
        rnnoise::TensorflowModel::Output output;
        output.data = output_denoise_buffer.data();
        output.name = "denoise_output/Sigmoid";
        outputs.push_back(output);
    }
    {
        rnnoise::TensorflowModel::Output output;
        output.data = output_val_buffer.data();
        output.name = "vad_output/Sigmoid";
        outputs.push_back(output);
    }
    model.Predict(&input, 1, outputs.data(), 2);
}
