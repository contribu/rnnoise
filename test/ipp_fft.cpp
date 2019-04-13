#include "gtest/gtest.h"

extern "C" {
#include "kiss_fft.h"
    typedef struct {
        int init;
        kiss_fft_state *kfft;
        float half_window[480];
        float dct_table[22*22];
#ifdef RNNOISE_IPP
        void *ipp_dft_work;
#endif
    } CommonState;
    
    void forward_transform_reference(CommonState *common, kiss_fft_cpx *out, const float *in);
    void inverse_transform_reference(CommonState *common, float *out, const kiss_fft_cpx *in);
}

void forward_transform_ipp(CommonState *common, kiss_fft_cpx *out, const float *in);
void inverse_transform_ipp(CommonState *common, float *out, const kiss_fft_cpx *in);

TEST(IppFft, ForwardMatchReferenceFftResult) {
    const int window_size = 960;
    const auto eps = 1e-6;
    
    for (int j = 0; j < window_size; j++) {
        float input[window_size] = { 0 };
        float actual[window_size + 2] = { 0 };
        float expected[window_size + 2] = { 0 };
        
        for (int i = 0; i < window_size; i++) {
            input[i] = i == j ? 1 : 0;
        }
        
        {
            CommonState common = { 0 };
            forward_transform_reference(&common, (kiss_fft_cpx *)expected, input);
        }
        {
            CommonState common = { 0 };
            forward_transform_ipp(&common, (kiss_fft_cpx *)actual, input);
        }
        
        for (int i = 0; i < window_size; i++) {
            EXPECT_NEAR(expected[i], actual[i], eps) << j << " " << i;
        }
    }
}

TEST(IppFft, BackwardMatchReferenceFftResult) {
    const int window_size = 960;
    const auto eps = 1e-6;
    
    for (int j = 0; j < window_size + 2; j++) {
        float input[window_size + 2] = { 0 };
        float actual[window_size] = { 0 };
        float expected[window_size] = { 0 };
        
        for (int i = 0; i < window_size + 2; i++) {
            input[i] = i == j ? 1 : 0;
        }
        input[1] = 0;
        input[window_size + 1] = 0;
        
        {
            CommonState common = { 0 };
            inverse_transform_reference(&common, expected, (kiss_fft_cpx *)input);
        }
        {
            CommonState common = { 0 };
            inverse_transform_ipp(&common, actual, (kiss_fft_cpx *)input);
        }
        
        for (int i = 0; i < window_size; i++) {
            EXPECT_NEAR(expected[i], actual[i], eps) << j << " " << i;
        }
    }
}
