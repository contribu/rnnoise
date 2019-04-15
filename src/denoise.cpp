/* Copyright (c) 2017 Mozilla */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

extern "C" {

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "kiss_fft.h"
#include "common.h"
#include <math.h>
#include "rnnoise.h"
#include "pitch.h"
#include "arch.h"
#include "rnn.h"
#include "rnn_data.h"

#define FRAME_SIZE_SHIFT 2
#define FRAME_SIZE (120<<FRAME_SIZE_SHIFT)
#define WINDOW_SIZE (2*FRAME_SIZE)
#define FREQ_SIZE (FRAME_SIZE + 1)

#define PITCH_MIN_PERIOD 60
#define PITCH_MAX_PERIOD 768
#define PITCH_FRAME_SIZE 960
#define PITCH_BUF_SIZE (PITCH_MAX_PERIOD+PITCH_FRAME_SIZE)

#define SQUARE(x) ((x)*(x))

#define SMOOTH_BANDS 1

#if SMOOTH_BANDS
#define NB_BANDS 22
#else
#define NB_BANDS 21
#endif

#define CEPS_MEM 8
#define NB_DELTA_CEPS 6

#define NB_FEATURES (NB_BANDS+3*NB_DELTA_CEPS+2)


#ifndef TRAINING
#define TRAINING 0
#endif

static const opus_int16 eband5ms[] = {
/*0  200 400 600 800  1k 1.2 1.4 1.6  2k 2.4 2.8 3.2  4k 4.8 5.6 6.8  8k 9.6 12k 15.6 20k*/
  0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100
};


typedef struct {
  int init;
  kiss_fft_state *kfft;
  float half_window[FRAME_SIZE];
  float dct_table[NB_BANDS*NB_BANDS];
#ifdef RNNOISE_IPP
    void *ipp_dft_work;
#endif
} CommonState;

struct DenoiseState {
  float analysis_mem[FRAME_SIZE];
  float cepstral_mem[CEPS_MEM][NB_BANDS];
  int memid;
  float synthesis_mem[FRAME_SIZE];
  float pitch_buf[PITCH_BUF_SIZE];
  float pitch_enh_buf[PITCH_BUF_SIZE];
  float last_gain;
  int last_period;
  float mem_hp_x[2];
  float lastg[NB_BANDS];
  RNNState rnn;
  CommonState common;
    void *tensorflow_model;
    float past_features[128 * 42];
};

#if SMOOTH_BANDS
void compute_band_energy(float *bandE, const kiss_fft_cpx *X) {
  int i;
  float sum[NB_BANDS] = {0};
  for (i=0;i<NB_BANDS-1;i++)
  {
    int j;
    int band_size;
    band_size = (eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;
    for (j=0;j<band_size;j++) {
      float tmp;
      float frac = (float)j/band_size;
      tmp = SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].r);
      tmp += SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].i);
      sum[i] += (1-frac)*tmp;
      sum[i+1] += frac*tmp;
    }
  }
  sum[0] *= 2;
  sum[NB_BANDS-1] *= 2;
  for (i=0;i<NB_BANDS;i++)
  {
    bandE[i] = sum[i];
  }
}

void compute_band_corr(float *bandE, const kiss_fft_cpx *X, const kiss_fft_cpx *P) {
  int i;
  float sum[NB_BANDS] = {0};
  for (i=0;i<NB_BANDS-1;i++)
  {
    int j;
    int band_size;
    band_size = (eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;
    for (j=0;j<band_size;j++) {
      float tmp;
      float frac = (float)j/band_size;
      tmp = X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].r * P[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].r;
      tmp += X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].i * P[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].i;
      sum[i] += (1-frac)*tmp;
      sum[i+1] += frac*tmp;
    }
  }
  sum[0] *= 2;
  sum[NB_BANDS-1] *= 2;
  for (i=0;i<NB_BANDS;i++)
  {
    bandE[i] = sum[i];
  }
}

void interp_band_gain(float *g, const float *bandE) {
  int i;
  memset(g, 0, FREQ_SIZE);
  for (i=0;i<NB_BANDS-1;i++)
  {
    int j;
    int band_size;
    band_size = (eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;
    for (j=0;j<band_size;j++) {
      float frac = (float)j/band_size;
      g[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j] = (1-frac)*bandE[i] + frac*bandE[i+1];
    }
  }
}
#else
void compute_band_energy(float *bandE, const kiss_fft_cpx *X) {
  int i;
  for (i=0;i<NB_BANDS;i++)
  {
    int j;
    opus_val32 sum = 0;
    for (j=0;j<(eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;j++) {
      sum += SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].r);
      sum += SQUARE(X[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j].i);
    }
    bandE[i] = sum;
  }
}

void interp_band_gain(float *g, const float *bandE) {
  int i;
  memset(g, 0, FREQ_SIZE);
  for (i=0;i<NB_BANDS;i++)
  {
    int j;
    for (j=0;j<(eband5ms[i+1]-eband5ms[i])<<FRAME_SIZE_SHIFT;j++)
      g[(eband5ms[i]<<FRAME_SIZE_SHIFT) + j] = bandE[i];
  }
}
#endif
    
#ifdef RNNOISE_IPP
}

#include <cstring>
#include <assert.h>
#include "ipp.h"

struct MyIppR2CDft32 {
    typedef float Float;
    typedef int Size;
    
    MyIppR2CDft32(int len): spec_(nullptr), work_buffer_size_(0) {
        int spec_size = 0;
        int init_buffer_size = 0;
        CheckResult(ippsDFTGetSize_R_32f(len, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone, &spec_size, &init_buffer_size, &work_buffer_size_));
        
        spec_ = spec_size ? (IppsDFTSpec_R_32f *)ippsMalloc_8u(spec_size) : nullptr;
        
        Ipp8u *init_buffer = init_buffer_size ? ippsMalloc_8u(init_buffer_size) : nullptr;
        if (init_buffer) std::memset(init_buffer, 0, init_buffer_size);
        
        CheckResult(ippsDFTInit_R_32f(len, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone, spec_, init_buffer));
        
        if (init_buffer) ippsFree(init_buffer);
    }
    ~MyIppR2CDft32() {
        if (!spec_) {
            ippsFree(spec_);
        }
    }
    void Execute(const Float *src, Float *dest, void *work) const {
        CheckResult(ippsDFTFwd_RToCCS_32f(src, dest, spec_, work_buffer_size_ ? (Ipp8u *)work : nullptr));
    }
    void ExecuteInv(const Float *src, Float *dest, void *work) const {
        CheckResult(ippsDFTInv_CCSToR_32f(src, dest, spec_, work_buffer_size_ ? (Ipp8u *)work : nullptr));
    }
    
    static const MyIppR2CDft32 &GetDefaultSizeInstance() {
        static MyIppR2CDft32 dft(WINDOW_SIZE);
        return dft;
    }
    
    int work_buffer_size() const { return work_buffer_size_; }
private:
    void CheckResult(IppStatus result) const {
        assert(result == ippStsNoErr);
    }
    IppsDFTSpec_R_32f *spec_;
    int work_buffer_size_;
};

static void init_ipp(CommonState *common) {
    const auto &dft = MyIppR2CDft32::GetDefaultSizeInstance();
    common->ipp_dft_work = ippsMalloc_8u(dft.work_buffer_size());
}
extern "C" {
#endif

}
#include <fstream>
#include <vector>
#include <stdexcept>
#include "tensorflow_model.h"
static void *create_tensormodel() {
    std::ifstream ifs("testmodel.pb", std::ios::binary | std::ios::ate);
    std::streamsize size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!ifs.read(buffer.data(), size)) throw std::logic_error("failed to load testmodel.pb");
    return new rnnoise::TensorflowModel(buffer.data(), buffer.size());
}

static void free_tensormodel(void *model) {
    delete (rnnoise::TensorflowModel *)model;
}
static void compute_rnn_tensorflow(DenoiseState *st, float *gain_ptr, float *vad_ptr, const float *input_ptr) {
    const int window_size = 128;
    rnnoise::TensorflowModel::Input input;
    input.data = input_ptr;
    input.dims.push_back(0);
    input.dims.push_back(window_size);
    input.dims.push_back(42);
    input.name = "main_input";
    std::vector<rnnoise::TensorflowModel::Output> outputs;
    {
        rnnoise::TensorflowModel::Output output;
        output.data = gain_ptr;
        output.name = "denoise_output/Sigmoid";
        outputs.push_back(output);
    }
    {
        rnnoise::TensorflowModel::Output output;
        output.data = vad_ptr;
        output.name = "vad_output/Sigmoid";
        outputs.push_back(output);
    }
    const auto model = (rnnoise::TensorflowModel *)st->tensorflow_model;
    model->Predict(&input, 1, outputs.data(), 2);
}

    extern "C" {

static void check_init(CommonState *common) {
  int i;
  if (common->init) return;
  common->kfft = opus_fft_alloc_twiddles(2*FRAME_SIZE, NULL, NULL, NULL, 0);
  for (i=0;i<FRAME_SIZE;i++)
    common->half_window[i] = sin(.5*M_PI*sin(.5*M_PI*(i+.5)/FRAME_SIZE) * sin(.5*M_PI*(i+.5)/FRAME_SIZE));
  for (i=0;i<NB_BANDS;i++) {
    int j;
    for (j=0;j<NB_BANDS;j++) {
      common->dct_table[i*NB_BANDS + j] = cos((i+.5)*j*M_PI/NB_BANDS);
      if (j==0) common->dct_table[i*NB_BANDS + j] *= sqrt(.5);
    }
  }
#ifdef RNNOISE_IPP
    init_ipp(common);
#endif
  common->init = 1;
}

static void dct(CommonState *common, float *out, const float *in) {
  int i;
  check_init(common);
  for (i=0;i<NB_BANDS;i++) {
    int j;
    float sum = 0;
    for (j=0;j<NB_BANDS;j++) {
      sum += in[j] * common->dct_table[j*NB_BANDS + i];
    }
    out[i] = sum*sqrt(2./22);
  }
}

#if 0
static void idct(float *out, const float *in) {
  int i;
  check_init();
  for (i=0;i<NB_BANDS;i++) {
    int j;
    float sum = 0;
    for (j=0;j<NB_BANDS;j++) {
      sum += in[j] * common.dct_table[i*NB_BANDS + j];
    }
    out[i] = sum*sqrt(2./22);
  }
}
#endif

#ifdef RNNOISE_IPP
}

void forward_transform_ipp(CommonState *common, kiss_fft_cpx *out, const float *in) {
    const auto &dft = MyIppR2CDft32::GetDefaultSizeInstance();
    
    check_init(common);
    static_assert(sizeof(kiss_fft_cpx) == 8, "kiss_fft_cpx must be packed");
    dft.Execute(in, (float *)out, common->ipp_dft_work);
    ippsMulC_32f_I(1.0 / WINDOW_SIZE, (float *)out, WINDOW_SIZE + 2);
}

void inverse_transform_ipp(CommonState *common, float *out, const kiss_fft_cpx *in) {
    const auto &dft = MyIppR2CDft32::GetDefaultSizeInstance();
    
    check_init(common);
    static_assert(sizeof(kiss_fft_cpx) == 8, "kiss_fft_cpx must be packed");
    dft.ExecuteInv((float *)in, out, common->ipp_dft_work);
}

extern "C" {
#endif
    
void forward_transform_reference(CommonState *common, kiss_fft_cpx *out, const float *in) {
  int i;
  kiss_fft_cpx x[WINDOW_SIZE];
  kiss_fft_cpx y[WINDOW_SIZE];
  check_init(common);
  for (i=0;i<WINDOW_SIZE;i++) {
    x[i].r = in[i];
    x[i].i = 0;
  }
  opus_fft(common->kfft, x, y, 0);
  for (i=0;i<FREQ_SIZE;i++) {
    out[i] = y[i];
  }
}

void inverse_transform_reference(CommonState *common, float *out, const kiss_fft_cpx *in) {
  int i;
  kiss_fft_cpx x[WINDOW_SIZE];
  kiss_fft_cpx y[WINDOW_SIZE];
  check_init(common);
  for (i=0;i<FREQ_SIZE;i++) {
    x[i] = in[i];
  }
  for (;i<WINDOW_SIZE;i++) {
    x[i].r = x[WINDOW_SIZE - i].r;
    x[i].i = -x[WINDOW_SIZE - i].i;
  }
  opus_fft(common->kfft, x, y, 0);
  /* output in reverse order for IFFT. */
  out[0] = WINDOW_SIZE*y[0].r;
  for (i=1;i<WINDOW_SIZE;i++) {
    out[i] = WINDOW_SIZE*y[WINDOW_SIZE - i].r;
  }
}
    
    
    static void forward_transform(CommonState *common, kiss_fft_cpx *out, const float *in) {
#ifdef RNNOISE_IPP
        forward_transform_ipp(common, out, in);
#else
        forward_transform_reference(common, out, in);
#endif
    }
    
    static void inverse_transform(CommonState *common, float *out, const kiss_fft_cpx *in) {
#ifdef RNNOISE_IPP
        inverse_transform_ipp(common, out, in);
#else
        inverse_transform_reference(common, out, in);
#endif
    }

static void apply_window(CommonState *common, float *x) {
  int i;
  check_init(common);
  for (i=0;i<FRAME_SIZE;i++) {
    x[i] *= common->half_window[i];
    x[WINDOW_SIZE - 1 - i] *= common->half_window[i];
  }
}

int rnnoise_get_size() {
  return sizeof(DenoiseState);
}

int rnnoise_init(DenoiseState *st) {
  memset(st, 0, sizeof(*st));
    st->tensorflow_model = create_tensormodel();
  return 0;
}

DenoiseState *rnnoise_create() {
  DenoiseState *st;
  st = (DenoiseState *)malloc(rnnoise_get_size());
  rnnoise_init(st);
  return st;
}

void rnnoise_destroy(DenoiseState *st) {
#ifdef RNNOISE_IPP
    if (st->common.init) {
        ippsFree(st->common.ipp_dft_work);
    }
#endif
    free_tensormodel(st->tensorflow_model);
  free(st);
}

#if TRAINING
int lowpass = FREQ_SIZE;
int band_lp = NB_BANDS;
#endif

static void frame_analysis(DenoiseState *st, kiss_fft_cpx *X, float *Ex, const float *in) {
  int i;
  float x[WINDOW_SIZE];
  RNN_COPY(x, st->analysis_mem, FRAME_SIZE);
  for (i=0;i<FRAME_SIZE;i++) x[FRAME_SIZE + i] = in[i];
  RNN_COPY(st->analysis_mem, in, FRAME_SIZE);
  apply_window(&st->common, x);
  forward_transform(&st->common, X, x);
#if TRAINING
  for (i=lowpass;i<FREQ_SIZE;i++)
    X[i].r = X[i].i = 0;
#endif
  compute_band_energy(Ex, X);
}

static int compute_frame_features(DenoiseState *st, kiss_fft_cpx *X, kiss_fft_cpx *P,
                                  float *Ex, float *Ep, float *Exp, float *features, const float *in) {
  int i;
  float E = 0;
  float *ceps_0, *ceps_1, *ceps_2;
  float spec_variability = 0;
  float Ly[NB_BANDS];
  float p[WINDOW_SIZE];
  float pitch_buf[PITCH_BUF_SIZE>>1];
  int pitch_index;
  float gain;
  float *pre[1];
  float tmp[NB_BANDS];
  float follow, logMax;
  frame_analysis(st, X, Ex, in);
  RNN_MOVE(st->pitch_buf, &st->pitch_buf[FRAME_SIZE], PITCH_BUF_SIZE-FRAME_SIZE);
  RNN_COPY(&st->pitch_buf[PITCH_BUF_SIZE-FRAME_SIZE], in, FRAME_SIZE);
  pre[0] = &st->pitch_buf[0];
  pitch_downsample(pre, pitch_buf, PITCH_BUF_SIZE, 1);
  pitch_search(pitch_buf+(PITCH_MAX_PERIOD>>1), pitch_buf, PITCH_FRAME_SIZE,
               PITCH_MAX_PERIOD-3*PITCH_MIN_PERIOD, &pitch_index);
  pitch_index = PITCH_MAX_PERIOD-pitch_index;

  gain = remove_doubling(pitch_buf, PITCH_MAX_PERIOD, PITCH_MIN_PERIOD,
          PITCH_FRAME_SIZE, &pitch_index, st->last_period, st->last_gain);
  st->last_period = pitch_index;
  st->last_gain = gain;
  for (i=0;i<WINDOW_SIZE;i++)
    p[i] = st->pitch_buf[PITCH_BUF_SIZE-WINDOW_SIZE-pitch_index+i];
  apply_window(&st->common, p);
  forward_transform(&st->common, P, p);
  compute_band_energy(Ep, P);
  compute_band_corr(Exp, X, P);
  for (i=0;i<NB_BANDS;i++) Exp[i] = Exp[i]/sqrt(.001+Ex[i]*Ep[i]);
  dct(&st->common, tmp, Exp);
  for (i=0;i<NB_DELTA_CEPS;i++) features[NB_BANDS+2*NB_DELTA_CEPS+i] = tmp[i];
  features[NB_BANDS+2*NB_DELTA_CEPS] -= 1.3;
  features[NB_BANDS+2*NB_DELTA_CEPS+1] -= 0.9;
  features[NB_BANDS+3*NB_DELTA_CEPS] = .01*(pitch_index-300);
  logMax = -2;
  follow = -2;
  for (i=0;i<NB_BANDS;i++) {
    Ly[i] = log10(1e-2+Ex[i]);
    Ly[i] = MAX16(logMax-7, MAX16(follow-1.5, Ly[i]));
    logMax = MAX16(logMax, Ly[i]);
    follow = MAX16(follow-1.5, Ly[i]);
    E += Ex[i];
  }
  if (!TRAINING && E < 0.04) {
    /* If there's no audio, avoid messing up the state. */
    RNN_CLEAR(features, NB_FEATURES);
    return 1;
  }
  dct(&st->common, features, Ly);
  features[0] -= 12;
  features[1] -= 4;
  ceps_0 = st->cepstral_mem[st->memid];
  ceps_1 = (st->memid < 1) ? st->cepstral_mem[CEPS_MEM+st->memid-1] : st->cepstral_mem[st->memid-1];
  ceps_2 = (st->memid < 2) ? st->cepstral_mem[CEPS_MEM+st->memid-2] : st->cepstral_mem[st->memid-2];
  for (i=0;i<NB_BANDS;i++) ceps_0[i] = features[i];
  st->memid++;
  for (i=0;i<NB_DELTA_CEPS;i++) {
    features[i] = ceps_0[i] + ceps_1[i] + ceps_2[i];
    features[NB_BANDS+i] = ceps_0[i] - ceps_2[i];
    features[NB_BANDS+NB_DELTA_CEPS+i] =  ceps_0[i] - 2*ceps_1[i] + ceps_2[i];
  }
  /* Spectral variability features. */
  if (st->memid == CEPS_MEM) st->memid = 0;
  for (i=0;i<CEPS_MEM;i++)
  {
    int j;
    float mindist = 1e15f;
    for (j=0;j<CEPS_MEM;j++)
    {
      int k;
      float dist=0;
      for (k=0;k<NB_BANDS;k++)
      {
        float tmp;
        tmp = st->cepstral_mem[i][k] - st->cepstral_mem[j][k];
        dist += tmp*tmp;
      }
      if (j!=i)
        mindist = MIN32(mindist, dist);
    }
    spec_variability += mindist;
  }
  features[NB_BANDS+3*NB_DELTA_CEPS+1] = spec_variability/CEPS_MEM-2.1;
  return TRAINING && E < 0.1;
}

static void frame_synthesis(DenoiseState *st, float *out, const kiss_fft_cpx *y) {
  float x[WINDOW_SIZE];
  int i;
  inverse_transform(&st->common, x, y);
  apply_window(&st->common, x);
  for (i=0;i<FRAME_SIZE;i++) out[i] = x[i] + st->synthesis_mem[i];
  RNN_COPY(st->synthesis_mem, &x[FRAME_SIZE], FRAME_SIZE);
}

static void biquad(float *y, float mem[2], const float *x, const float *b, const float *a, int N) {
  int i;
  for (i=0;i<N;i++) {
    float xi, yi;
    xi = x[i];
    yi = x[i] + mem[0];
    mem[0] = mem[1] + (b[0]*(double)xi - a[0]*(double)yi);
    mem[1] = (b[1]*(double)xi - a[1]*(double)yi);
    y[i] = yi;
  }
}

void pitch_filter(kiss_fft_cpx *X, const kiss_fft_cpx *P, const float *Ex, const float *Ep,
                  const float *Exp, const float *g) {
  int i;
  float r[NB_BANDS];
  float rf[FREQ_SIZE] = {0};
  for (i=0;i<NB_BANDS;i++) {
#if 0
    if (Exp[i]>g[i]) r[i] = 1;
    else r[i] = Exp[i]*(1-g[i])/(.001 + g[i]*(1-Exp[i]));
    r[i] = MIN16(1, MAX16(0, r[i]));
#else
    if (Exp[i]>g[i]) r[i] = 1;
    else r[i] = SQUARE(Exp[i])*(1-SQUARE(g[i]))/(.001 + SQUARE(g[i])*(1-SQUARE(Exp[i])));
    r[i] = sqrt(MIN16(1, MAX16(0, r[i])));
#endif
    r[i] *= sqrt(Ex[i]/(1e-8+Ep[i]));
  }
  interp_band_gain(rf, r);
  for (i=0;i<FREQ_SIZE;i++) {
    X[i].r += rf[i]*P[i].r;
    X[i].i += rf[i]*P[i].i;
  }
  float newE[NB_BANDS];
  compute_band_energy(newE, X);
  float norm[NB_BANDS];
  float normf[FREQ_SIZE]={0};
  for (i=0;i<NB_BANDS;i++) {
    norm[i] = sqrt(Ex[i]/(1e-8+newE[i]));
  }
  interp_band_gain(normf, norm);
  for (i=0;i<FREQ_SIZE;i++) {
    X[i].r *= normf[i];
    X[i].i *= normf[i];
  }
}

float rnnoise_process_frame(DenoiseState *st, float *out, const float *in, int pitch_filter_enabled) {
  int i;
  kiss_fft_cpx X[FREQ_SIZE];
  kiss_fft_cpx P[WINDOW_SIZE];
  float x[FRAME_SIZE];
  float Ex[NB_BANDS], Ep[NB_BANDS];
  float Exp[NB_BANDS];
  float features[NB_FEATURES];
  float g[NB_BANDS];
  float gf[FREQ_SIZE]={1};
  float vad_prob = 0;
  int silence;
  static const float a_hp[2] = {-1.99599, 0.99600};
  static const float b_hp[2] = {-2, 1};
  biquad(x, st->mem_hp_x, in, b_hp, a_hp, FRAME_SIZE);
  silence = compute_frame_features(st, X, P, Ex, Ep, Exp, features, x);

  if (!silence) {
#if 1
      for (int i = 0; i < 127; i++) {
          for (int j = 0; j < 42; j++) {
              st->past_features[42 * i + j] = st->past_features[42 * (i + 1) + j];
          }
      }
      for (int j = 0; j < 42; j++) {
          st->past_features[42 * 127 + j] = features[j];
      }
      float transposed[128 * 42];
      for (int i = 0; i < 128; i++) {
          for (int j = 0; j < 42; j++) {
              transposed[128 * j + i] = st->past_features[42 * i + j];
          }
      }
      compute_rnn_tensorflow(st, g, &vad_prob, transposed);
#else
    compute_rnn(&st->rnn, g, &vad_prob, features);
#endif
    if (pitch_filter_enabled) {
      pitch_filter(X, P, Ex, Ep, Exp, g);
    }
    for (i=0;i<NB_BANDS;i++) {
      float alpha = .6f;
      g[i] = MAX16(g[i], alpha*st->lastg[i]);
      st->lastg[i] = g[i];
    }
    interp_band_gain(gf, g);
#if 1
    for (i=0;i<FREQ_SIZE;i++) {
      X[i].r *= gf[i];
      X[i].i *= gf[i];
    }
#endif
  }

  frame_synthesis(st, out, X);
  return vad_prob;
}

}

#if TRAINING

#include <stdexcept>
#include "gflags/gflags.h"

DEFINE_string(clean, "", "clean raw pcm path");
DEFINE_string(noise, "", "noise raw pcm path");
DEFINE_string(output, "", "output path (unused but must be specified)");
DEFINE_int64(output_count, 50000000, "output frame count");
DEFINE_int64(clean_initial_pos, 0, "clean initial pos in sample");
DEFINE_int64(noise_initial_pos, 0, "noise initial pos in sample");

namespace {
    static float uni_rand() {
        return rand()/(double)RAND_MAX-.5;
    }
    
    static void rand_resp(float *a, float *b) {
        a[0] = .75*uni_rand();
        a[1] = .75*uni_rand();
        b[0] = .75*uni_rand();
        b[1] = .75*uni_rand();
    }
}

int main(int argc, char **argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    
    int i;
    int count=0;
    static const float a_hp[2] = {-1.99599, 0.99600};
    static const float b_hp[2] = {-2, 1};
    float a_noise[2] = {0};
    float b_noise[2] = {0};
    float a_sig[2] = {0};
    float b_sig[2] = {0};
    float mem_hp_x[2]={0};
    float mem_hp_n[2]={0};
    float mem_resp_x[2]={0};
    float mem_resp_n[2]={0};
    float x[FRAME_SIZE];
    float n[FRAME_SIZE];
    float xn[FRAME_SIZE];
    int vad_cnt=0;
    int gain_change_count=0;
    float speech_gain = 1, noise_gain = 1;
    FILE *f1, *f2, *fout;
    DenoiseState *st;
    DenoiseState *noise_state;
    DenoiseState *noisy;
    st = rnnoise_create();
    noise_state = rnnoise_create();
    noisy = rnnoise_create();
    f1 = fopen(FLAGS_clean.c_str(), "r");
    if (!f1) throw std::logic_error("cannot open " + FLAGS_clean);
    if (fseek(f1, FLAGS_clean_initial_pos * 2, SEEK_SET)) throw std::logic_error("fseek failed");
    f2 = fopen(FLAGS_noise.c_str(), "r");
    if (!f2) throw std::logic_error("cannot open " + FLAGS_noise);
    if (fseek(f2, FLAGS_noise_initial_pos * 2, SEEK_SET)) throw std::logic_error("fseek failed");
    fout = fopen(FLAGS_output.c_str(), "w");
    if (!fout) throw std::logic_error("cannot open " + FLAGS_output);
    for(i=0;i<150;i++) {
        short tmp[FRAME_SIZE];
        fread(tmp, sizeof(short), FRAME_SIZE, f2);
    }
    while (1) {
        kiss_fft_cpx X[FREQ_SIZE], Y[FREQ_SIZE], N[FREQ_SIZE], P[WINDOW_SIZE];
        float Ex[NB_BANDS], Ey[NB_BANDS], En[NB_BANDS], Ep[NB_BANDS];
        float Exp[NB_BANDS];
        float Ln[NB_BANDS];
        float features[NB_FEATURES];
        float g[NB_BANDS];
        float gf[FREQ_SIZE]={1};
        short tmp[FRAME_SIZE];
        float vad=0;
        float vad_prob;
        float E=0;
        if (count==FLAGS_output_count) break;
        if (++gain_change_count > 2821) {
            speech_gain = pow(10., (-60+(rand()%100))/20.);
            noise_gain = pow(10., (-30+(rand()%50))/20.);
            if (rand()%3==0) noise_gain = 0;
            noise_gain *= speech_gain;
            if (rand()%3==0) speech_gain = 0;
            gain_change_count = 0;
            rand_resp(a_noise, b_noise);
            rand_resp(a_sig, b_sig);
            lowpass = FREQ_SIZE * 3000./24000. * pow(50., rand()/(double)RAND_MAX);
            for (i=0;i<NB_BANDS;i++) {
                if (eband5ms[i]<<FRAME_SIZE_SHIFT > lowpass) {
                    band_lp = i;
                    break;
                }
            }
        }
        if (speech_gain != 0) {
            fread(tmp, sizeof(short), FRAME_SIZE, f1);
            if (feof(f1)) {
                rewind(f1);
                fread(tmp, sizeof(short), FRAME_SIZE, f1);
            }
            for (i=0;i<FRAME_SIZE;i++) x[i] = speech_gain*tmp[i];
            for (i=0;i<FRAME_SIZE;i++) E += tmp[i]*(float)tmp[i];
        } else {
            for (i=0;i<FRAME_SIZE;i++) x[i] = 0;
            E = 0;
        }
        if (noise_gain!=0) {
            fread(tmp, sizeof(short), FRAME_SIZE, f2);
            if (feof(f2)) {
                rewind(f2);
                fread(tmp, sizeof(short), FRAME_SIZE, f2);
            }
            for (i=0;i<FRAME_SIZE;i++) n[i] = noise_gain*tmp[i];
        } else {
            for (i=0;i<FRAME_SIZE;i++) n[i] = 0;
        }
        biquad(x, mem_hp_x, x, b_hp, a_hp, FRAME_SIZE);
        biquad(x, mem_resp_x, x, b_sig, a_sig, FRAME_SIZE);
        biquad(n, mem_hp_n, n, b_hp, a_hp, FRAME_SIZE);
        biquad(n, mem_resp_n, n, b_noise, a_noise, FRAME_SIZE);
        for (i=0;i<FRAME_SIZE;i++) xn[i] = x[i] + n[i];
        if (E > 1e9f) {
            vad_cnt=0;
        } else if (E > 1e8f) {
            vad_cnt -= 5;
        } else if (E > 1e7f) {
            vad_cnt++;
        } else {
            vad_cnt+=2;
        }
        if (vad_cnt < 0) vad_cnt = 0;
        if (vad_cnt > 15) vad_cnt = 15;
        
        if (vad_cnt >= 10) vad = 0;
        else if (vad_cnt > 0) vad = 0.5f;
        else vad = 1.f;
        
        frame_analysis(st, Y, Ey, x);
        frame_analysis(noise_state, N, En, n);
        for (i=0;i<NB_BANDS;i++) Ln[i] = log10(1e-2+En[i]);
        int silence = compute_frame_features(noisy, X, P, Ex, Ep, Exp, features, xn);
        pitch_filter(X, P, Ex, Ep, Exp, g);
        //printf("%f %d\n", noisy->last_gain, noisy->last_period);
        for (i=0;i<NB_BANDS;i++) {
            g[i] = sqrt((Ey[i]+1e-3)/(Ex[i]+1e-3));
            if (g[i] > 1) g[i] = 1;
            // if (silence || i > band_lp) g[i] = -1;
            // if (Ey[i] < 5e-2 && Ex[i] < 5e-2) g[i] = -1;
            // if (vad==0 && noise_gain==0) g[i] = -1;
        }
        count++;
        if (count % 1000 == 0)
            fprintf(stderr, "%d\n", count);
#if 0
        for (i=0;i<NB_FEATURES;i++) printf("%f ", features[i]);
        for (i=0;i<NB_BANDS;i++) printf("%f ", g[i]);
        for (i=0;i<NB_BANDS;i++) printf("%f ", Ln[i]);
        printf("%f\n", vad);
#endif
#if 1
        fwrite(features, sizeof(float), NB_FEATURES, stdout);
        fwrite(g, sizeof(float), NB_BANDS, stdout);
        fwrite(Ln, sizeof(float), NB_BANDS, stdout);
        fwrite(&vad, sizeof(float), 1, stdout);
#endif
#if 0
        compute_rnn(&noisy->rnn, g, &vad_prob, features);
        interp_band_gain(gf, g);
#if 1
        for (i=0;i<FREQ_SIZE;i++) {
            X[i].r *= gf[i];
            X[i].i *= gf[i];
        }
#endif
        frame_synthesis(noisy, xn, X);
        
        for (i=0;i<FRAME_SIZE;i++) tmp[i] = xn[i];
        fwrite(tmp, sizeof(short), FRAME_SIZE, fout);
#endif
    }
    fprintf(stderr, "matrix size: %d x %d\n", count, NB_FEATURES + 2*NB_BANDS + 1);
    fclose(f1);
    fclose(f2);
    fclose(fout);
    return 0;
}

#endif
