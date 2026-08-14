
/**
  ******************************************************************************
  * @file    app_x-cube-ai.c
  * @author  X-CUBE-AI C code generator
  * @brief   AI program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

 /*
  * Description
  *   v1.0 - Minimum template to show how to use the Embedded Client API
  *          model. Only one input and one output is supported. All
  *          memory resources are allocated statically (AI_NETWORK_XX, defines
  *          are used).
  *          Re-target of the printf function is out-of-scope.
  *   v2.0 - add multiple IO and/or multiple heap support
  *
  *   For more information, see the embeded documentation:
  *
  *       [1] %X_CUBE_AI_DIR%/Documentation/index.html
  *
  *   X_CUBE_AI_DIR indicates the location where the X-CUBE-AI pack is installed
  *   typical : C:\Users\[user_name]\STM32Cube\Repository\STMicroelectronics\X-CUBE-AI\7.1.0
  */

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/

#if defined ( __ICCARM__ )
#elif defined ( __CC_ARM ) || ( __GNUC__ )
#endif

/* System headers */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>

#include "app_x-cube-ai.h"
#include "main.h"
#include "ai_datatypes_defines.h"
#include "network_1.h"
#include "network_1_data.h"
#include "network_2.h"
#include "network_2_data.h"
#include "network_ee.h"
#include "network_ee_data.h"
#include "test_data.h"
#include <math.h>

/* USER CODE BEGIN includes */

#define POOL0_SIZE ( (AI_NETWORK_1_DATA_ACTIVATION_1_SIZE > AI_NETWORK_2_DATA_ACTIVATION_1_SIZE) \
                       ? AI_NETWORK_1_DATA_ACTIVATION_1_SIZE \
                       : AI_NETWORK_2_DATA_ACTIVATION_1_SIZE )
#define EE_NUM_CLASSES   AI_NETWORK_EE_OUT_1_SIZE   /* should be 12 */
#define EE_CONF_THRESHOLD 0.90f
static volatile uint8_t run_network_2_flag = 0;
uint32_t t_init;
uint32_t t_out;
uint32_t duration_us;
uint32_t duration_dwt;
float clock_Hz;
uint32_t cpuclk;

typedef struct {
  float mae_ee_sum, cosine_ee_sum;
  float mae_net2_sum, cosine_net2_sum;
  int   net2_compare_count;
  int   ee_class_match, early_exit_match, cascade_correct;
  uint16_t confusion[EE_NUM_CLASSES][EE_NUM_CLASSES];  /* [true][predicted] */
} validation_stats_t;

#if AI_NETWORK_1_IN_1_SIZE != MFCC_VECTOR_SIZE
#error "Model input size and MFCC test vector size do not match"
#endif
#if AI_NETWORK_EE_OUT_1_SIZE != EE1_OUTPUT_SIZE
#error "network_ee output size and ee1 reference size do not match"
#endif
#if AI_NETWORK_2_OUT_1_SIZE != NETWORK2_OUTPUT_SIZE
#error "network_2 output size and reference size do not match"
#endif
/* USER CODE END includes */

/* IO buffers ----------------------------------------------------------------*/

#if !defined(AI_NETWORK_1_INPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_in_1[AI_NETWORK_1_IN_1_SIZE_BYTES];
ai_i8* data_ins[AI_NETWORK_1_IN_NUM] = {
data_in_1
};
#else
ai_i8* data_ins[AI_NETWORK_1_IN_NUM] = {
NULL
};
#endif

#if !defined(AI_NETWORK_1_OUTPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_out_1[AI_NETWORK_1_OUT_1_SIZE_BYTES];
ai_i8* data_outs[AI_NETWORK_1_OUT_NUM] = {
data_out_1
};
#else
ai_i8* data_outs[AI_NETWORK_1_OUT_NUM] = {
NULL
};
#endif

// Network 2 //
#if !defined(AI_NETWORK_2_INPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_in_2[AI_NETWORK_2_IN_1_SIZE_BYTES];
ai_i8* data_ins_2[AI_NETWORK_2_IN_NUM] = {
data_in_2
};
#else
ai_i8* data_ins_2[AI_NETWORK_2_IN_NUM] = {
NULL
};
#endif

#if !defined(AI_NETWORK_2_OUTPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_out_2[AI_NETWORK_2_OUT_1_SIZE_BYTES];
ai_i8* data_outs_2[AI_NETWORK_2_OUT_NUM] = {
data_out_2
};
#else
ai_i8* data_outs_2[AI_NETWORK_2_OUT_NUM] = {
NULL
};
#endif

// Network ee //
#if !defined(AI_NETWORK_EE_INPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_in_ee[AI_NETWORK_EE_IN_1_SIZE_BYTES];
ai_i8* data_ins_ee[AI_NETWORK_EE_IN_NUM] = {
data_in_ee
};
#else
ai_i8* data_ins_ee[AI_NETWORK_EE_IN_NUM] = {
NULL
};
#endif

#if !defined(AI_NETWORK_EE_OUTPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_out_ee[AI_NETWORK_EE_OUT_1_SIZE_BYTES];
ai_i8* data_outs_ee[AI_NETWORK_EE_OUT_NUM] = {
data_out_ee
};
#else
ai_i8* data_outs_ee[AI_NETWORK_EE_OUT_NUM] = {
NULL
};
#endif


/* Activations buffers -------------------------------------------------------*/

AI_ALIGNED(32)
static uint8_t pool0[POOL0_SIZE];
AI_ALIGNED(32)
static uint8_t pool2[AI_NETWORK_EE_DATA_ACTIVATION_1_SIZE];

ai_handle data_activations0[] = {pool0};
ai_handle data_activations2[] = {pool2};

/* AI objects ----------------------------------------------------------------*/

static ai_handle network_1 = AI_HANDLE_NULL;
static ai_handle network_2 = AI_HANDLE_NULL;
static ai_handle network_ee = AI_HANDLE_NULL;

static ai_buffer* ai_input;
static ai_buffer* ai_output;

static ai_buffer* ai_input_2;
static ai_buffer* ai_output_2;

static ai_buffer* ai_input_ee;
static ai_buffer* ai_output_ee;

static void ai_log_err(const ai_error err, const char *fct)
{
  /* USER CODE BEGIN log */
  if (fct)
    printf("TEMPLATE - Error (%s) - type=0x%02x code=0x%02x\r\n", fct,
        err.type, err.code);
  else
    printf("TEMPLATE - Error - type=0x%02x code=0x%02x\r\n", err.type, err.code);

  do {} while (1);
  /* USER CODE END log */
}

static int ai_boostrap(ai_handle *act_addr)
{
  ai_error err;

  /* Create and initialize an instance of the model */
  err = ai_network_1_create_and_init(&network_1, act_addr, NULL);
  if (err.type != AI_ERROR_NONE) {
    ai_log_err(err, "ai_network_1_create_and_init");
    return -1;
  }

  ai_input = ai_network_1_inputs_get(network_1, NULL);
  ai_output = ai_network_1_outputs_get(network_1, NULL);

#if defined(AI_NETWORK_1_INPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-inputs" option is used, memory buffer can be
   *  used from the activations buffer. This is not mandatory.
   */
  for (int idx=0; idx < AI_NETWORK_1_IN_NUM; idx++) {
	data_ins[idx] = ai_input[idx].data;
  }
#else
  for (int idx=0; idx < AI_NETWORK_1_IN_NUM; idx++) {
	  ai_input[idx].data = data_ins[idx];
  }
#endif

#if defined(AI_NETWORK_1_OUTPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-outputs" option is used, memory buffer can be
   *  used from the activations buffer. This is no mandatory.
   */
  for (int idx=0; idx < AI_NETWORK_1_OUT_NUM; idx++) {
	data_outs[idx] = ai_output[idx].data;
  }
#else
  for (int idx=0; idx < AI_NETWORK_1_OUT_NUM; idx++) {
	ai_output[idx].data = data_outs[idx];
  }
#endif

  return 0;
}

static int ai_run(void)
{
  ai_i32 batch;

  batch = ai_network_1_run(network_1, ai_input, ai_output);
  if (batch != 1) {
    ai_log_err(ai_network_1_get_error(network_1),
        "ai_network_1_run");
    return -1;
  }

  return 0;
}

/* USER CODE BEGIN 2 */
static int softmax_and_argmax(const float* logits, float* probs, int n, float* confidence)
{
  /* 1. Find max logit for numerical stability (avoids exp() overflow) */
  float max_logit = logits[0];
  for (int i = 1; i < n; i++) {
    if (logits[i] > max_logit) {
      max_logit = logits[i];
    }
  }

  /* 2. Compute exp(logit - max) and accumulate the sum */
  float sum_exp = 0.0f;
  for (int i = 0; i < n; i++) {
    probs[i] = expf(logits[i] - max_logit);
    sum_exp += probs[i];
  }

  /* 3. Normalize to get probabilities, tracking argmax along the way */
  int predicted_class = 0;
  float max_prob = 0.0f;

  for (int i = 0; i < n; i++) {
    probs[i] /= sum_exp;
    if (probs[i] > max_prob) {
      max_prob = probs[i];
      predicted_class = i;
    }
  }

  *confidence = max_prob;
  return predicted_class;
}
static int ai_boostrap_2(ai_handle *act_addr)
{
  ai_error err;

  /* Create and initialize an instance of the model */
  err = ai_network_2_create_and_init(&network_2, act_addr, NULL);
  if (err.type != AI_ERROR_NONE) {
    ai_log_err(err, "ai_network_2_create_and_init");
    return -1;
  }

  ai_input_2 = ai_network_2_inputs_get(network_2, NULL);
  ai_output_2 = ai_network_2_outputs_get(network_2, NULL);

#if defined(AI_NETWORK_2_INPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-inputs" option is used, memory buffer can be
   *  used from the activations buffer. This is not mandatory.
   */
  for (int idx=0; idx < AI_NETWORK_2_IN_NUM; idx++) {
	data_ins_2[idx] = ai_input_2[idx].data;
  }
#else
  for (int idx=0; idx < AI_NETWORK_2_IN_NUM; idx++) {
	  ai_input_2[idx].data = data_ins_2[idx];
  }
#endif

#if defined(AI_NETWORK_2_OUTPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-outputs" option is used, memory buffer can be
   *  used from the activations buffer. This is no mandatory.
   */
  for (int idx=0; idx < AI_NETWORK_2_OUT_NUM; idx++) {
	data_outs_2[idx] = ai_output_2[idx].data;
  }
#else
  for (int idx=0; idx < AI_NETWORK_2_OUT_NUM; idx++) {
	ai_output_2[idx].data = data_outs_2[idx];
  }
#endif

  return 0;
}

static int ai_boostrap_ee(ai_handle *act_addr)
{
  ai_error err;

  /* Create and initialize an instance of the model */
  err = ai_network_ee_create_and_init(&network_ee, act_addr, NULL);
  if (err.type != AI_ERROR_NONE) {
    ai_log_err(err, "ai_network_ee_create_and_init");
    return -1;
  }

  ai_input_ee = ai_network_ee_inputs_get(network_ee, NULL);
  ai_output_ee = ai_network_ee_outputs_get(network_ee, NULL);

#if defined(AI_NETWORK_EE_INPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-inputs" option is used, memory buffer can be
   *  used from the activations buffer. This is not mandatory.
   */
  for (int idx=0; idx < AI_NETWORK_EE_IN_NUM; idx++) {
	data_ins_ee[idx] = ai_input_ee[idx].data;
  }
#else
  for (int idx=0; idx < AI_NETWORK_EE_IN_NUM; idx++) {
	  ai_input_ee[idx].data = data_ins_ee[idx];
  }
#endif

#if defined(AI_NETWORK_EE_OUTPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-outputs" option is used, memory buffer can be
   *  used from the activations buffer. This is no mandatory.
   */
  for (int idx=0; idx < AI_NETWORK_EE_OUT_NUM; idx++) {
	data_outs_ee[idx] = ai_output_ee[idx].data;
  }
#else
  for (int idx=0; idx < AI_NETWORK_EE_OUT_NUM; idx++) {
	ai_output_ee[idx].data = data_outs_ee[idx];
  }
#endif

  return 0;
}

static int ai_run_2(void)
{
  ai_i32 batch;

  batch = ai_network_2_run(network_2, ai_input_2, ai_output_2);
  if (batch != 1) {
    ai_log_err(ai_network_2_get_error(network_2),
        "ai_network_2_run");
    return -1;
  }

  return 0;
}

static int ai_run_ee(void)
{
  ai_i32 batch;

  batch = ai_network_ee_run(network_ee, ai_input_ee, ai_output_ee);
  if (batch != 1) {
    ai_log_err(ai_network_ee_get_error(network_ee),
        "ai_network_ee_run");
    return -1;
  }

  return 0;
}

float network_1_out_buff[AI_NETWORK_1_OUT_1_SIZE];
float network_ee_out_buff[AI_NETWORK_EE_OUT_1_SIZE];
float network_2_out_buff[AI_NETWORK_2_OUT_1_SIZE];
static float ee_probs[EE_NUM_CLASSES];

/* Set before calling network_1_run() to select which test vector feeds in.
 * NULL falls back to normal production behavior (unchanged from before). */
static const float* current_test_input = NULL;

int acquire_and_process_data(ai_i8* data[])
{
	float* in_buf = (float*)data[0];
	const float* src = current_test_input ? current_test_input : mfcc_samples[0];
	for(int i = 0; i < AI_NETWORK_1_IN_1_SIZE; i++){
		in_buf[i] = src[i];
	}
	return 0;
}

int post_process(ai_i8* data[])
{
	float* out_buf = (float*)data[0];
	for(int i = 0; i < AI_NETWORK_1_OUT_1_SIZE; i++){
		network_1_out_buff[i] = out_buf[i];
	}
	return 1;
}

int acquire_and_process_data_ee(ai_i8* data[])
{
  float* in_buf_ee = (float*)data[0];
  /* Confirm AI_NETWORK_EE_IN_1_SIZE_BYTES == AI_NETWORK_1_OUT_1_SIZE_BYTES */
  memcpy(in_buf_ee, network_1_out_buff, AI_NETWORK_EE_IN_1_SIZE_BYTES);
  return 0;
}

//int post_process_ee(ai_i8* data[])
//{
//  float* out_buf_ee = (float*)data[0];
//  float confidence;
//  int predicted_class = softmax_and_argmax(out_buf_ee, ee_probs, EE_NUM_CLASSES, &confidence);
//  for (int i = 0; i < AI_NETWORK_EE_OUT_1_SIZE; i++) {
//    network_ee_out_buff[i] = out_buf_ee[i];
//  }
//  run_network_2_flag = (confidence > EE_CONF_THRESHOLD) ? 0 : 1;
//  return 1;
//}

int post_process_ee(ai_i8* data[])
{
  float* out_buf_ee = (float*)data[0];

  /* out_buf_ee is ALREADY post-softmax — the network_ee model's final
   * layer includes softmax. Do NOT apply softmax again; just take argmax
   * and read the confidence directly. */
  int predicted_class = 0;
  float max_prob = out_buf_ee[0];
  for (int i = 1; i < AI_NETWORK_EE_OUT_1_SIZE; i++) {
    if (out_buf_ee[i] > max_prob) {
      max_prob = out_buf_ee[i];
      predicted_class = i;
    }
  }
  float confidence = max_prob;

  for (int i = 0; i < AI_NETWORK_EE_OUT_1_SIZE; i++) {
    network_ee_out_buff[i] = out_buf_ee[i];
    ee_probs[i] = out_buf_ee[i];   /* already probabilities — copy as-is */
  }

  run_network_2_flag = (confidence > EE_CONF_THRESHOLD) ? 0 : 1;
  return 1;
}

int acquire_and_process_data_2(ai_i8* data[])
{
  float* in_buf_2 = (float*)data[0];
  memcpy(in_buf_2, network_1_out_buff, AI_NETWORK_2_IN_1_SIZE_BYTES);
  return 0;
}

int post_process_2(ai_i8* data[])
{
  float* out_buf_2 = (float*)data[0];
  for (int i = 0; i < AI_NETWORK_2_OUT_1_SIZE; i++) {
      network_2_out_buff[i] = out_buf_2[i];
   }
  return 1;
}

static void network_1_run(void){
	int res = 1;
	do {
	      /* 1 - acquire and pre-process input data */
	      res = acquire_and_process_data(data_ins);
	      /* 2 - process the data - call inference engine */
	      if (res == 0)
	        res = ai_run();
	      /* 3- post-process the predictions */
	      if (res == 0)
	        res = post_process(data_outs);
	    } while (res==0);
}

static void network_2_run(void){
	int res = 1;
	do {
	      /* 1 - acquire and pre-process input data */
	      res = acquire_and_process_data_2(data_ins_2);
	      /* 2 - process the data - call inference engine */
	      if (res == 0)
	        res = ai_run_2();
	      /* 3- post-process the predictions */
	      if (res == 0)
	        res = post_process_2(data_outs_2);
	    } while (res==0);
}

static void network_ee_run(void){
	int res = 1;
	do {
	      /* 1 - acquire and pre-process input data */
	      res = acquire_and_process_data_ee(data_ins_ee);
	      /* 2 - process the data - call inference engine */
	      if (res == 0)
	        res = ai_run_ee();
	      /* 3- post-process the predictions */
	      if (res == 0)
	        res = post_process_ee(data_outs_ee);
	    } while (res==0);
}

static int ai_deinit_network_1(void)
{
  network_1 = ai_network_1_destroy(network_1);
  if (network_1 != AI_HANDLE_NULL) {
    return -1;   /* destroy did not fully release */
  }
  return 0;
}

static float compute_mae(const float* a, const float* b, int n)
{
  float sum_abs_diff = 0.0f;
  for (int i = 0; i < n; i++) sum_abs_diff += fabsf(a[i] - b[i]);
  return sum_abs_diff / (float)n;
}

static float compute_cosine_similarity(const float* a, const float* b, int n)
{
  float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
  for (int i = 0; i < n; i++) {
    dot    += a[i] * b[i];
    norm_a += a[i] * a[i];
    norm_b += b[i] * b[i];
  }
  float denom = sqrtf(norm_a) * sqrtf(norm_b);
  if (denom < 1e-12f) return 0.0f;
  return dot / denom;
}

static int argmax_float(const float* arr, int n)
{
  int best = 0;
  for (int i = 1; i < n; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

void init_dwt(void){
	CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;  /* enable trace/debug block */
	DWT->CYCCNT = 0;                                  /* reset cycle counter */
	DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;               /* enable cycle counter */
}

void time_in(void){
	DWT->CYCCNT = 0;
	t_init = DWT->CYCCNT;
}

uint32_t time_out(void){
	t_out = DWT->CYCCNT;
	return (t_out - t_init);   /* unsigned subtraction — safe across wraparound */
}

static validation_stats_t vstats;
static uint32_t sample_duration_us[NUM_TEST_SAMPLES];

static void run_validation_sample(int idx)
{
  const test_metadata_t* md = &test_metadata[idx];

  time_in();

  /* Restore network_1 into pool0 if network_2 occupied it last cycle
   * — same guard your production Process() uses. */
  if (network_2 != AI_HANDLE_NULL) {
    network_2 = ai_network_2_destroy(network_2);
    ai_boostrap(data_activations0);
  }

  current_test_input = mfcc_samples[idx];
  network_1_run();     /* -> network_1_out_buff */
  network_ee_run();    /* -> network_ee_out_buff, ee_probs, run_network_2_flag */

  /* --- network_ee vs Python reference --- */
  float mae_ee = compute_mae(network_ee_out_buff, ee1_ref_outputs[idx], EE_NUM_CLASSES);
  float cos_ee = compute_cosine_similarity(network_ee_out_buff, ee1_ref_outputs[idx], EE_NUM_CLASSES);
  vstats.mae_ee_sum += mae_ee;
  vstats.cosine_ee_sum += cos_ee;

  int device_ee_class = argmax_float(ee_probs, EE_NUM_CLASSES);
  float device_confidence = ee_probs[device_ee_class];
  if (device_ee_class == md->ee1_prediction) vstats.ee_class_match++;

  int device_early_exit = !run_network_2_flag;
  if (device_early_exit == md->early_exit) vstats.early_exit_match++;

  int cascade_prediction = device_ee_class;

  /* --- network_2, only if escalation triggered --- */
  if (run_network_2_flag) {
    ai_deinit_network_1();
    if (ai_boostrap_2(data_activations0) == 0) {
      network_2_run();   /* -> network_2_out_buff */

      int device_net2_class = argmax_float(network_2_out_buff, AI_NETWORK_2_OUT_1_SIZE);
      cascade_prediction = device_net2_class;

      if (md->network2_prediction != -1) {
        vstats.mae_net2_sum += compute_mae(network_2_out_buff, network2_ref_outputs[idx], AI_NETWORK_2_OUT_1_SIZE);
        vstats.cosine_net2_sum += compute_cosine_similarity(network_2_out_buff, network2_ref_outputs[idx], AI_NETWORK_2_OUT_1_SIZE);
        vstats.net2_compare_count++;
      }
    }
    run_network_2_flag = 0;
  }

  if (cascade_prediction == md->cascade_prediction) vstats.cascade_correct++;
  if (md->true_label >= 0 && md->true_label < EE_NUM_CLASSES &&
      cascade_prediction >= 0 && cascade_prediction < EE_NUM_CLASSES) {
    vstats.confusion[md->true_label][cascade_prediction]++;
  }

  sample_duration_us[idx] = time_out();
  sample_duration_us[idx] = (uint32_t)(((float)sample_duration_us[idx] * 1000000.0f) / clock_Hz);

  printf("[%d] true=%d  ee_pred(dev/py)=%d/%d  conf(dev/py)=%.4f/%.4f  "
         "early_exit(dev/py)=%d/%d  cascade(dev/py)=%d/%d  "
         "MAE_ee=%.6f  cos_ee=%.6f  time=%luus\r\n",
         idx, md->true_label,
         device_ee_class, md->ee1_prediction,
         device_confidence, md->ee1_confidence,
         device_early_exit, md->early_exit,
         cascade_prediction, md->cascade_prediction,
         mae_ee, cos_ee, sample_duration_us[idx]);
}

void run_full_validation_suite(void)
{
  memset(&vstats, 0, sizeof(vstats));

  printf("\r\n=== Running validation suite (%d samples) ===\r\n", NUM_TEST_SAMPLES);

  for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
    run_validation_sample(i);
  }

  uint32_t total_us = 0;
  for (int i = 0; i < NUM_TEST_SAMPLES; i++) total_us += sample_duration_us[i];

  printf("\r\n=== Summary ===\r\n");
  printf("network_ee vs Python: avg MAE=%.6f  avg cosine=%.6f\r\n",
         vstats.mae_ee_sum / NUM_TEST_SAMPLES, vstats.cosine_ee_sum / NUM_TEST_SAMPLES);
  printf("ee1 predicted-class match: %d/%d\r\n", vstats.ee_class_match, NUM_TEST_SAMPLES);
  printf("early-exit decision match: %d/%d\r\n", vstats.early_exit_match, NUM_TEST_SAMPLES);
  printf("cascade prediction match (vs Python cascade_prediction): %d/%d\r\n",
         vstats.cascade_correct, NUM_TEST_SAMPLES);

  if (vstats.net2_compare_count > 0) {
    printf("network_2 vs Python (%d escalated samples): avg MAE=%.6f  avg cosine=%.6f\r\n",
           vstats.net2_compare_count,
           vstats.mae_net2_sum / vstats.net2_compare_count,
           vstats.cosine_net2_sum / vstats.net2_compare_count);
  } else {
    printf("network_2 never ran on-device\r\n");
  }

  printf("avg inference time: %lu us/sample\r\n", total_us / NUM_TEST_SAMPLES);

  printf("\r\nConfusion matrix [true][predicted]:\r\n");
  for (int t = 0; t < EE_NUM_CLASSES; t++) {
    printf("  %d: ", t);
    for (int p = 0; p < EE_NUM_CLASSES; p++) printf("%3u ", vstats.confusion[t][p]);
    printf("\r\n");
  }

  /* Restore normal production state for MX_X_CUBE_AI_Process() afterward */
  current_test_input = NULL;
  if (network_2 != AI_HANDLE_NULL) {
    network_2 = ai_network_2_destroy(network_2);
    ai_boostrap(data_activations0);
  }
}



/* USER CODE END 2 */

/* Entry points --------------------------------------------------------------*/

void MX_X_CUBE_AI_Init(void)
{
    /* USER CODE BEGIN 5 */
  ai_boostrap(data_activations0);
  ai_boostrap_ee(data_activations2);
  cpuclk = HAL_RCC_GetHCLKFreq();
  clock_Hz = (float)cpuclk;
  init_dwt();
    /* USER CODE END 5 */
}

void MX_X_CUBE_AI_Process(void)
{
    /* USER CODE BEGIN 6 */
	run_full_validation_suite();
}
    /* USER CODE END 6 */

