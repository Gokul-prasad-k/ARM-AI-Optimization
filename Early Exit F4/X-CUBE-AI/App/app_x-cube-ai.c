
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

//AI_ALIGNED(32)
//static uint8_t pool0[AI_NETWORK_1_DATA_ACTIVATION_1_SIZE];
//AI_ALIGNED(32)
//static uint8_t pool1[AI_NETWORK_2_DATA_ACTIVATION_1_SIZE];

AI_ALIGNED(32)
static uint8_t pool0[POOL0_SIZE];
AI_ALIGNED(32)
static uint8_t pool2[AI_NETWORK_EE_DATA_ACTIVATION_1_SIZE];

ai_handle data_activations0[] = {pool0};
//ai_handle data_activations1[] = {pool1};
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
static float ee_probs[EE_NUM_CLASSES];

int acquire_and_process_data(ai_i8* data[])
{
	float* in_buf = (float*)data[0];
	for(int i = 0; i < AI_NETWORK_1_IN_1_SIZE; i++){
		in_buf[i] = 1.0;
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

int post_process_ee(ai_i8* data[])
{
  float* out_buf_ee = (float*)data[0];
  float confidence;
  int predicted_class = softmax_and_argmax(out_buf_ee, ee_probs, EE_NUM_CLASSES, &confidence);
  for (int i = 0; i < AI_NETWORK_EE_OUT_1_SIZE; i++) {
    network_ee_out_buff[i] = out_buf_ee[i];
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

/* USER CODE END 2 */

/* Entry points --------------------------------------------------------------*/

void MX_X_CUBE_AI_Init(void)
{
    /* USER CODE BEGIN 5 */
  printf("\r\nTEMPLATE - initialization\r\n");

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
	time_in();
	if (network_2 != AI_HANDLE_NULL) {
		network_2 = ai_network_2_destroy(network_2);
		ai_boostrap(data_activations0);
	}
	network_1_run();
	network_ee_run();

	if (run_network_2_flag) {
		ai_deinit_network_1();
		if (ai_boostrap_2(data_activations0) == 0) {
			network_2_run();
		}
	}
	run_network_2_flag = 0;
	duration_dwt = time_out();
	duration_us = (uint32_t)(((float)duration_dwt * 1000000.0)/clock_Hz);
}
    /* USER CODE END 6 */

/* Multiple network support --------------------------------------------------*/

//#include <string.h>
//#include "ai_datatypes_defines.h"
//
//static const ai_network_entry_t networks[AI_MNETWORK_NUMBER] = {
//    {
//        .name = (const char *)AI_NETWORK_1_MODEL_NAME,
//        .config = AI_NETWORK_1_DATA_CONFIG,
//        .ai_get_report = ai_network_1_get_report,
//        .ai_create = ai_network_1_create,
//        .ai_destroy = ai_network_1_destroy,
//        .ai_get_error = ai_network_1_get_error,
//        .ai_init = ai_network_1_init,
//        .ai_run = ai_network_1_run,
//        .ai_forward = ai_network_1_forward,
//        .ai_data_params_get = ai_network_1_data_params_get,
//        .activations = data_activations0
//    },
//    {
//        .name = (const char *)AI_NETWORK_2_MODEL_NAME,
//        .config = AI_NETWORK_2_DATA_CONFIG,
//        .ai_get_report = ai_network_2_get_report,
//        .ai_create = ai_network_2_create,
//        .ai_destroy = ai_network_2_destroy,
//        .ai_get_error = ai_network_2_get_error,
//        .ai_init = ai_network_2_init,
//        .ai_run = ai_network_2_run,
//        .ai_forward = ai_network_2_forward,
//        .ai_data_params_get = ai_network_2_data_params_get,
//        .activations = data_activations1
//    },
//    {
//        .name = (const char *)AI_NETWORK_EE_MODEL_NAME,
//        .config = AI_NETWORK_EE_DATA_CONFIG,
//        .ai_get_report = ai_network_ee_get_report,
//        .ai_create = ai_network_ee_create,
//        .ai_destroy = ai_network_ee_destroy,
//        .ai_get_error = ai_network_ee_get_error,
//        .ai_init = ai_network_ee_init,
//        .ai_run = ai_network_ee_run,
//        .ai_forward = ai_network_ee_forward,
//        .ai_data_params_get = ai_network_ee_data_params_get,
//        .activations = data_activations2
//    },
//};
//
//struct network_instance {
//     const ai_network_entry_t *entry;
//     ai_handle handle;
//     ai_network_params params;
//};
//
///* Number of instance is aligned on the number of network */
//AI_STATIC struct network_instance gnetworks[AI_MNETWORK_NUMBER] = {0};
//
//AI_DECLARE_STATIC
//ai_bool ai_mnetwork_is_valid(const char* name,
//        const ai_network_entry_t *entry)
//{
//    if (name && (strlen(entry->name) == strlen(name)) &&
//            (strncmp(entry->name, name, strlen(entry->name)) == 0))
//        return true;
//    return false;
//}
//
//AI_DECLARE_STATIC
//struct network_instance *ai_mnetwork_handle(struct network_instance *inst)
//{
//    for (int i=0; i<AI_MNETWORK_NUMBER; i++) {
//        if ((inst) && (&gnetworks[i] == inst))
//            return inst;
//        else if ((!inst) && (gnetworks[i].entry == NULL))
//            return &gnetworks[i];
//    }
//    return NULL;
//}
//
//AI_DECLARE_STATIC
//void ai_mnetwork_release_handle(struct network_instance *inst)
//{
//    for (int i=0; i<AI_MNETWORK_NUMBER; i++) {
//        if ((inst) && (&gnetworks[i] == inst)) {
//            gnetworks[i].entry = NULL;
//            return;
//        }
//    }
//}
//
//AI_API_ENTRY
//const char* ai_mnetwork_find(const char *name, ai_int idx)
//{
//    const ai_network_entry_t *entry;
//
//    for (int i=0; i<AI_MNETWORK_NUMBER; i++) {
//        entry = &networks[i];
//        if (ai_mnetwork_is_valid(name, entry))
//            return entry->name;
//        else {
//            if (!idx--)
//                return entry->name;
//        }
//    }
//    return NULL;
//}
//
//AI_API_ENTRY
//ai_error ai_mnetwork_create(const char *name, ai_handle* network,
//        const ai_buffer* network_config)
//{
//    const ai_network_entry_t *entry;
//    const ai_network_entry_t *found = NULL;
//    ai_error err;
//    struct network_instance *inst = ai_mnetwork_handle(NULL);
//
//    if (!inst) {
//        err.type = AI_ERROR_ALLOCATION_FAILED;
//        err.code = AI_ERROR_CODE_NETWORK;
//        return err;
//    }
//
//    for (int i=0; i<AI_MNETWORK_NUMBER; i++) {
//        entry = &networks[i];
//        if (ai_mnetwork_is_valid(name, entry)) {
//            found = entry;
//            break;
//        }
//    }
//
//    if (!found) {
//        err.type = AI_ERROR_INVALID_PARAM;
//        err.code = AI_ERROR_CODE_NETWORK;
//        return err;
//    }
//
//    if (network_config == NULL)
//        err = found->ai_create(network, found->config);
//    else
//        err = found->ai_create(network, network_config);
//    if ((err.code == AI_ERROR_CODE_NONE) && (err.type == AI_ERROR_NONE)) {
//        inst->entry = found;
//        inst->handle = *network;
//        *network = (ai_handle*)inst;
//    }
//
//    return err;
//}
//
//AI_API_ENTRY
//ai_handle ai_mnetwork_destroy(ai_handle network)
//{
//    struct network_instance *inn;
//    inn =  ai_mnetwork_handle((struct network_instance *)network);
//    if (inn) {
//        ai_handle hdl = inn->entry->ai_destroy(inn->handle);
//        if (hdl != inn->handle) {
//            ai_mnetwork_release_handle(inn);
//            network = AI_HANDLE_NULL;
//        }
//    }
//    return network;
//}
//
//AI_API_ENTRY
//ai_bool ai_mnetwork_get_report(ai_handle network, ai_network_report* report)
//{
//    struct network_instance *inn;
//    inn =  ai_mnetwork_handle((struct network_instance *)network);
//    if (inn)
//        return inn->entry->ai_get_report(inn->handle, report);
//    else
//        return false;
//}
//
//AI_API_ENTRY
//ai_error ai_mnetwork_get_error(ai_handle network)
//{
//    struct network_instance *inn;
//    ai_error err;
//    err.type = AI_ERROR_INVALID_PARAM;
//    err.code = AI_ERROR_CODE_NETWORK;
//
//    inn =  ai_mnetwork_handle((struct network_instance *)network);
//    if (inn)
//        return inn->entry->ai_get_error(inn->handle);
//    else
//        return err;
//}
//
//AI_API_ENTRY
//ai_bool ai_mnetwork_init(ai_handle network)
//{
//    struct network_instance *inn;
//    ai_network_params par;
//
//    inn =  ai_mnetwork_handle((struct network_instance *)network);
//    if (inn) {
//        inn->entry->ai_data_params_get(&par);
//        for (int idx=0; idx < par.map_activations.size; idx++)
//          AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&par.map_activations, idx, inn->entry->activations[idx]);
//        return inn->entry->ai_init(inn->handle, &par);
//    }
//    else
//        return false;
//}
//
//AI_API_ENTRY
//ai_i32 ai_mnetwork_run(ai_handle network, const ai_buffer* input,
//        ai_buffer* output)
//{
//    struct network_instance* inn;
//    inn =  ai_mnetwork_handle((struct network_instance *)network);
//    if (inn)
//        return inn->entry->ai_run(inn->handle, input, output);
//    else
//        return 0;
//}
//
//AI_API_ENTRY
//ai_i32 ai_mnetwork_forward(ai_handle network, const ai_buffer* input)
//{
//    struct network_instance *inn;
//    inn =  ai_mnetwork_handle((struct network_instance *)network);
//    if (inn)
//        return inn->entry->ai_forward(inn->handle, input);
//    else
//        return 0;
//}
//
//AI_API_ENTRY
// int ai_mnetwork_get_private_handle(ai_handle network,
//         ai_handle *phandle,
//         ai_network_params *pparams)
// {
//     struct network_instance* inn;
//     inn =  ai_mnetwork_handle((struct network_instance *)network);
//     if (inn && phandle && pparams) {
//         *phandle = inn->handle;
//         *pparams = inn->params;
//         return 0;
//     }
//     else
//         return -1;
// }
//
//#ifdef __cplusplus
//}
//#endif
