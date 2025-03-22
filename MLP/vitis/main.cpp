#include "model_params.hpp"
#include <ap_fixed.h>
#include <hls_stream.h>

// Read AXI Stream and store into input array
void read_stream(hls::stream<AXI_VAL> &in_stream, input_fixed_t input[INPUT_SIZE]) {
    data x;
    for (int i = 0; i < INPUT_SIZE; i++) {
        AXI_VAL inVal = in_stream.read();
        x.intVal = inVal.data;
        input[i] = x.floatVal; // Convert float to fixed_t explicitly
    }
}

// Write the predicted label to the AXI stream
void write_stream(hls::stream<AXI_VAL> &out_stream, int label) {
    AXI_VAL outVal;
    outVal.data = label;
    outVal.keep = -1;
    outVal.strb = -1;
    outVal.last = 1;
    out_stream.write(outVal);
}

// Compute first hidden layer with ReLU activation
void compute_layer0(input_fixed_t input[INPUT_SIZE], fixed_t hidden1[HIDDEN_SIZE],
                    const fixed_t w0[HIDDEN_SIZE][INPUT_SIZE], const fixed_t b0[HIDDEN_SIZE]) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
		#pragma HLS PIPELINE II=4 // more relaxation dealing with float multiplied by fixed_t
        prod_fixed_t sum = b0[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
        	prod_fixed_t product = w0[i][j] * input[j];
        	sum += product;
        }
        hidden1[i] = (sum > 0) ? (fixed_t) sum : (fixed_t) 0;
//        hidden1[i] = (sum > 0) ? sum : 0;
    }
}

// Compute second hidden layer with ReLU activation
void compute_layer1(fixed_t hidden1[HIDDEN_SIZE], fixed_t hidden2[HIDDEN_SIZE],
                    const fixed_t w1[HIDDEN_SIZE][HIDDEN_SIZE], const fixed_t b1[HIDDEN_SIZE]) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
		#pragma HLS PIPELINE II=2
        fixed_t sum = b1[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {

            sum += w1[i][j] * hidden1[j];
        }
        hidden2[i] = (sum > 0) ? sum : (fixed_t) 0;
//        hidden2[i] = (sum > 0) ? sum : 0;
    }
}

// Compute output layer
void compute_layer2(fixed_t hidden2[HIDDEN_SIZE], fixed_t output[OUTPUT_SIZE],
                    const fixed_t w2[OUTPUT_SIZE][HIDDEN_SIZE], const fixed_t b2[OUTPUT_SIZE]) {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
		#pragma HLS PIPELINE II=2
        fixed_t sum = b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += w2[i][j] * hidden2[j];
        }
        output[i] = sum;
    }
}

// Top-level function
void mlp_forward(hls::stream<AXI_VAL> &in_stream, hls::stream<AXI_VAL> &out_stream) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=return

    // Force constant weights/biases to be stored in BRAM
    #pragma HLS BIND_STORAGE variable=layer_0_weight type=ROM_2P impl=BRAM
    #pragma HLS BIND_STORAGE variable=layer_1_weight type=ROM_2P impl=BRAM
    #pragma HLS BIND_STORAGE variable=layer_2_weight type=ROM_2P impl=BRAM
    #pragma HLS BIND_STORAGE variable=layer_0_bias   type=ROM_1P impl=BRAM
    #pragma HLS BIND_STORAGE variable=layer_1_bias   type=ROM_1P impl=BRAM
    #pragma HLS BIND_STORAGE variable=layer_2_bias   type=ROM_1P impl=BRAM

	input_fixed_t input[INPUT_SIZE];
    fixed_t hidden1[HIDDEN_SIZE];
    fixed_t hidden2[HIDDEN_SIZE];
    fixed_t output[OUTPUT_SIZE];

    read_stream(in_stream, input);
    compute_layer0(input, hidden1, layer_0_weight, layer_0_bias);
    compute_layer1(hidden1, hidden2, layer_1_weight, layer_1_bias);
    compute_layer2(hidden2, output, layer_2_weight, layer_2_bias);

    fixed_t max_val = -1e9;
//    float max_val = -1e9;
    int label = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        if (output[i] > max_val) {
            max_val = output[i];
            label = i;
        }
    }
    write_stream(out_stream, label);
}
