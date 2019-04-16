#include "gtest/gtest.h"

extern "C" {
#include "rnn.h"
#include "common.h"
#include "rnn_data.h"
}
    
TEST(Rnn, MatchKerasResult) {
    RNNState rnn = { 0 };
    const auto eps = 1e-5;
    for (int i = 0; i < 2000; i++) {
        const float *input = rnn_test_data[0].input + NB_FEATURES * i;
        const float *expected_gain = rnn_test_data[0].gain + NB_BANDS * i;
        const float *expected_vad = rnn_test_data[0].vad + i;
        
        float actual_gain[NB_BANDS];
        float actual_vad[1];
        
        compute_rnn(&rnn, actual_gain, actual_vad, input);
        
        for (int j = 0; j < NB_BANDS; j++) {
            EXPECT_NEAR(expected_gain[j], actual_gain[j], eps);
        }
        for (int j = 0; j < 1; j++) {
            EXPECT_NEAR(expected_vad[j], actual_vad[j], eps);
        }
    }
}
