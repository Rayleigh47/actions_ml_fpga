#ifndef MLP_HPP
#define MLP_HPP

#include "hls_stream.h"
#include "ap_axi_sdata.h"

#define INPUT_SIZE 72
#define HIDDEN_SIZE 64
#define OUTPUT_SIZE 10

/*
 * Define ap_axis data type
 * data: 32 bit data width
 * id: 2 bit used to differentiate streams
 * user: 5 bit custom metadata
 * dest: 6 bit destination of the data
 */
typedef ap_axis<32, 2, 5, 6> axi_stream;
typedef ap_fixed<8,1> weights_t;
typedef ap_fixed<32,8> fixed_t;

/*
 * Union to represent different data type in same memory location.
 * Used to intepret int as float or vice versa.
 */
union data {
	int intVal;
	float floatVal;
};

// Functions
void read_stream(hls::stream<axi_stream> &in_stream, fixed_t input[INPUT_SIZE]);
void write_stream(hls::stream<axi_stream> &out_stream, int label);
fixed_t tanh_activation(fixed_t x);
void compute_layer0(fixed_t input[INPUT_SIZE],
                    fixed_t hidden1[HIDDEN_SIZE],
                    const weights_t w0[HIDDEN_SIZE][INPUT_SIZE],
                    const fixed_t b0[HIDDEN_SIZE]);
void compute_layer1(const fixed_t hidden1[HIDDEN_SIZE],
                    fixed_t hidden2[HIDDEN_SIZE],
                    const weights_t w1[HIDDEN_SIZE][HIDDEN_SIZE],
                    const fixed_t b1[HIDDEN_SIZE]);
void compute_layer2(const fixed_t hidden2[HIDDEN_SIZE],
                    fixed_t output[OUTPUT_SIZE],
                    const weights_t w2[OUTPUT_SIZE][HIDDEN_SIZE],
                    const fixed_t b2[OUTPUT_SIZE]);
void compute_softmax(fixed_t output[OUTPUT_SIZE], float class_predictions[OUTPUT_SIZE]);
void mlp_tanh_forward(hls::stream<axi_stream> &in_stream, hls::stream<axi_stream> &out_stream);

#endif
