#include "gtest/gtest.h"

extern "C" {
#include "rnn.h"
#include "rnn_data.h"
}
    
TEST(Rnn, MatchKerasResult) {
    RNNState rnn = { 0 };
    const auto eps = 1e-5;
    for (int i = 0; i < 2000; i++) {
        const float *input = rnn_test_data[0].input + 42 * i;
        const float *expected_gain = rnn_test_data[0].gain + 22 * i;
        const float *expected_vad = rnn_test_data[0].vad + i;
        
        float actual_gain[22];
        float actual_vad[1];
        
        compute_rnn(&rnn, actual_gain, actual_vad, input);
        
        for (int j = 0; j < 22; j++) {
            EXPECT_NEAR(actual_gain[j], expected_gain[j], eps);
        }
        for (int j = 0; j < 1; j++) {
            EXPECT_NEAR(actual_vad[j], expected_vad[j], eps);
        }
    }
}
