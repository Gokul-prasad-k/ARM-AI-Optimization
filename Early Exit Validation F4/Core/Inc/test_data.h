#ifndef TEST_DATA_H
#define TEST_DATA_H

#include <stdint.h>

#define NUM_TEST_SAMPLES  10
#define MFCC_VECTOR_SIZE  490
#define EE1_OUTPUT_SIZE   12
#define NETWORK2_OUTPUT_SIZE  12

typedef struct {
    int   sample_index;
    int   true_label;
    int   ee1_prediction;
    float ee1_confidence;
    float threshold;
    uint8_t early_exit;
    int   network2_prediction;   /* -1 if not applicable (early exit taken) */
    int   cascade_prediction;
} test_metadata_t;

extern const float* mfcc_samples[NUM_TEST_SAMPLES];
extern const float* ee1_ref_outputs[NUM_TEST_SAMPLES];
extern const float* network2_ref_outputs[NUM_TEST_SAMPLES];
extern const test_metadata_t test_metadata[NUM_TEST_SAMPLES];


#endif /* TEST_DATA_H */
