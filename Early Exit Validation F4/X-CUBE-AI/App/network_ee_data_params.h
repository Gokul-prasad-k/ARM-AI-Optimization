/**
  ******************************************************************************
  * @file    network_ee_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-08-14T23:49:04+0530
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef NETWORK_EE_DATA_PARAMS_H
#define NETWORK_EE_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_NETWORK_EE_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_network_ee_data_weights_params[1]))
*/

#define AI_NETWORK_EE_DATA_CONFIG               (NULL)


#define AI_NETWORK_EE_DATA_ACTIVATIONS_SIZES \
  { 32000, }
#define AI_NETWORK_EE_DATA_ACTIVATIONS_SIZE     (32000)
#define AI_NETWORK_EE_DATA_ACTIVATIONS_COUNT    (1)
#define AI_NETWORK_EE_DATA_ACTIVATION_1_SIZE    (32000)



#define AI_NETWORK_EE_DATA_WEIGHTS_SIZES \
  { 3120, }
#define AI_NETWORK_EE_DATA_WEIGHTS_SIZE         (3120)
#define AI_NETWORK_EE_DATA_WEIGHTS_COUNT        (1)
#define AI_NETWORK_EE_DATA_WEIGHT_1_SIZE        (3120)



#define AI_NETWORK_EE_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_network_ee_activations_table[1])

extern ai_handle g_network_ee_activations_table[1 + 2];



#define AI_NETWORK_EE_DATA_WEIGHTS_TABLE_GET() \
  (&g_network_ee_weights_table[1])

extern ai_handle g_network_ee_weights_table[1 + 2];


#endif    /* NETWORK_EE_DATA_PARAMS_H */
