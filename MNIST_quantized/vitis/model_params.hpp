#ifndef MLP_HPP
#define MLP_HPP

#include "hls_stream.h"
#include "ap_axi_sdata.h"

#define INPUT_SIZE 784
#define HIDDEN_SIZE_0 512
#define HIDDEN_SIZE_1 256
#define OUTPUT_SIZE 10
#define WEIGHTS_BIAS_0 (INPUT_SIZE * HIDDEN_SIZE_0 + HIDDEN_SIZE_0)
#define WEIGHTS_BIAS_1 (HIDDEN_SIZE_0 * HIDDEN_SIZE_1 + HIDDEN_SIZE_1)
#define WEIGHTS_BIAS_2 (OUTPUT_SIZE * HIDDEN_SIZE_1 + OUTPUT_SIZE)
#define TOTAL_WEIGHTS_BIAS (WEIGHTS_BIAS_0 + WEIGHTS_BIAS_1 + WEIGHTS_BIAS_2)
/*
 * Define ap_axis data type
 * data: 32 bit data width
 * id: 2 bit used to differentiate streams
 * user: 5 bit custom metadata
 * dest: 6 bit destination of the data
 */
typedef ap_axis<32, 2, 5, 6> axi_stream;

/*
 * Union to represent different data type in same memory location.
 * Used to intepret int as float or vice versa.
 */
union data {
	int intVal;
	float floatVal;
};
void read_weights_biases(hls::stream<axi_stream> &in_stream);
void init_layers();
void write_weights_biases(hls::stream<axi_stream> &out_stream);
void read_inputs(hls::stream<axi_stream> &in_stream, int8_t input[INPUT_SIZE]);
float tanh_approx(float x);
int8_t quantize_tanh(int32_t sum, float scale);
void inference(int8_t input[INPUT_SIZE],
               int8_t hidden_1[HIDDEN_SIZE_0],
               int8_t hidden_2[HIDDEN_SIZE_1],
               int32_t output[OUTPUT_SIZE],
               int8_t weights_0[HIDDEN_SIZE_0][INPUT_SIZE],
               int8_t weights_1[HIDDEN_SIZE_1][HIDDEN_SIZE_0],
               int8_t weights_2[OUTPUT_SIZE][HIDDEN_SIZE_1],
               int32_t bias_0[HIDDEN_SIZE_0],
               int32_t bias_1[HIDDEN_SIZE_1],
               int32_t bias_2[OUTPUT_SIZE],
               float scale_0,
               float scale_1,
               float class_predictions[OUTPUT_SIZE]);
void compute_layer0(int8_t input[INPUT_SIZE],
                    int8_t hidden1[HIDDEN_SIZE_0],
                    int8_t w0[HIDDEN_SIZE_0][INPUT_SIZE],
                    int32_t b0[HIDDEN_SIZE_0],
                    float scale_0);
void compute_layer1(int8_t hidden1[HIDDEN_SIZE_0],
                    int8_t hidden2[HIDDEN_SIZE_1],
                    int8_t w1[HIDDEN_SIZE_1][HIDDEN_SIZE_0],
                    int32_t b1[HIDDEN_SIZE_1],
                    float scale_1);
void compute_layer2(int8_t hidden2[HIDDEN_SIZE_1],
                    int32_t output[OUTPUT_SIZE],
                    int8_t w2[OUTPUT_SIZE][HIDDEN_SIZE_1],
                    int32_t b2[OUTPUT_SIZE]);
void compute_softmax(int32_t output[OUTPUT_SIZE], float class_predictions[OUTPUT_SIZE]);
void send_outputs(hls::stream<axi_stream> &out_stream, float class_predictions[OUTPUT_SIZE]);
void mnist_quantized(hls::stream<axi_stream> &in_stream, hls::stream<axi_stream> &out_stream);

#endif
