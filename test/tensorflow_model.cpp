#include "gtest/gtest.h"

#include <fstream>
#include <vector>
#include "tensorflow_model.h"
#include "rnn_data.h"
#include "common.h"

TEST(TensorflowModel, MatchKerasResult) {
    const auto eps = 1e-5;
    
    // cdn model
    std::ifstream ifs("testmodel.pb", std::ios::binary | std::ios::ate);
    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    EXPECT_TRUE(ifs.read(buffer.data(), size));
    rnnoise::TensorflowModel model(buffer.data(), buffer.size());
    
    std::vector<float> input_buffer(128 * NB_FEATURES);
    for (int i = 0; i < input_buffer.size(); i++) {
        input_buffer[i] = rnn_test_data[0].input[i];
    }
    std::vector<float> output_denoise_buffer(NB_BANDS);
    std::vector<float> output_val_buffer(1);
    rnnoise::TensorflowModel::Input input;
    input.data = input_buffer.data();
    input.dims.push_back(1);
    input.dims.push_back(128);
    input.dims.push_back(NB_FEATURES);
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
    
    const float *expected_gain = rnn_test_data[0].gain;
    const float *expected_vad = rnn_test_data[0].vad;
    
    for (int j = 0; j < NB_BANDS; j++) {
        EXPECT_NEAR(expected_gain[j], output_denoise_buffer[j], eps);
    }
    for (int j = 0; j < 1; j++) {
        EXPECT_NEAR(expected_vad[j], output_val_buffer[j], eps);
    }
}

TEST(TensorflowModel, TcnMatchKerasResult) {
    const auto eps = 1e-5;
    
    // tcn model
    std::ifstream ifs("testmodeltcn.pb", std::ios::binary | std::ios::ate);
    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    EXPECT_TRUE(ifs.read(buffer.data(), size));
    rnnoise::TensorflowModel model(buffer.data(), buffer.size());
    
    const int window_size = 128; // arbitrary but must match test data size
    std::vector<float> input_buffer(window_size * NB_FEATURES);
    for (int i = 0; i < input_buffer.size(); i++) {
        input_buffer[i] = rnn_test_data[0].input[i];
    }
    std::vector<float> output_denoise_buffer(window_size * NB_BANDS);
    std::vector<float> output_val_buffer(window_size * 1);
    rnnoise::TensorflowModel::Input input;
    input.data = input_buffer.data();
    input.dims.push_back(1);
    input.dims.push_back(window_size);
    input.dims.push_back(NB_FEATURES);
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
    
    const float *expected_gain = rnn_test_data[0].gain;
    const float *expected_vad = rnn_test_data[0].vad;
    
    for (int j = 0; j < window_size * NB_BANDS; j++) {
        EXPECT_NEAR(expected_gain[j], output_denoise_buffer[j], eps);
    }
    for (int j = 0; j < window_size * 1; j++) {
        EXPECT_NEAR(expected_vad[j], output_val_buffer[j], eps);
    }
}
