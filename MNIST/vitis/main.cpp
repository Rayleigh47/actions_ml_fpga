#include "model_params.hpp"
#include <hls_stream.h>
#include <hls_math.h>
#include <ap_fixed.h>

// Fixed-point ReLU: simply zero out negative values.
fixed_t fixed_relu(fixed_t x) {
    return (x < (fixed_t) 0) ? (fixed_t) 0 : x;
}

// Read AXI Stream and store into input array as fixed_t.
// We assume the incoming float is in [0,1] and we directly cast it to fixed_t.
void read_stream(hls::stream<axi_stream> &in_stream, fixed_t input[INPUT_SIZE]) {
    data x;
    axi_stream inVal;
    for (int i = 0; i < INPUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        inVal = in_stream.read();
        x.intVal = inVal.data;  // reinterpret the 32-bit data as a float
        float value = x.floatVal; // value in [0,1]
        input[i] = (fixed_t)value; // direct conversion; no extra scaling needed
    }
}

// Write the predicted label to the AXI stream
void write_stream(hls::stream<axi_stream> &out_stream, int label) {
    axi_stream outVal;
    outVal.data = label;
    outVal.keep = -1;
    outVal.strb = -1;
    outVal.last = 1;
    out_stream.write(outVal);
}

// Compute first hidden layer with fixed-point ReLU activation.
void compute_layer0(fixed_t input[INPUT_SIZE],
                    fixed_t hidden1[HIDDEN_SIZE_0],
                    const weight_t w0[HIDDEN_SIZE_0][INPUT_SIZE],
                    const fixed_t b0[HIDDEN_SIZE_0]) {
    for (int i = 0; i < HIDDEN_SIZE_0; i++) {
        #pragma HLS PIPELINE II=1
        fixed_t sum = b0[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            #pragma HLS UNROLL factor=4
            fixed_t product = input[j] * w0[i][j];
            sum += product;
        }
        hidden1[i] = fixed_relu(sum);
    }
}

// Compute second hidden layer with fixed-point ReLU activation.
void compute_layer1(fixed_t input[HIDDEN_SIZE_0],
                    fixed_t hidden2[HIDDEN_SIZE_1],
                    const weight_t w1[HIDDEN_SIZE_1][HIDDEN_SIZE_0],
                    const fixed_t b1[HIDDEN_SIZE_1]) {
    for (int i = 0; i < HIDDEN_SIZE_1; i++) {
        #pragma HLS PIPELINE II=1
        fixed_t sum = b1[i];
        for (int j = 0; j < HIDDEN_SIZE_0; j++) {
            #pragma HLS UNROLL factor=4
            fixed_t product = input[j] * w1[i][j];
            sum += product;
        }
        hidden2[i] = fixed_relu(sum);
    }
}

// Compute output layer (without activation quantization)
void compute_layer2(fixed_t hidden2[HIDDEN_SIZE_1],
                    fixed_t output[OUTPUT_SIZE],
                    const weight_t w2[OUTPUT_SIZE][HIDDEN_SIZE_1],
                    const fixed_t b2[OUTPUT_SIZE]) {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        fixed_t sum = b2[i];
        for (int j = 0; j < HIDDEN_SIZE_1; j++) {
            #pragma HLS UNROLL factor=4
            fixed_t product = hidden2[j] * w2[i][j];
            sum += product;
        }
        output[i] = sum;
    }
}

// Compute softmax using floating-point operations.
// Convert fixed-point outputs to float before processing.
void compute_softmax(fixed_t output[OUTPUT_SIZE], float class_predictions[OUTPUT_SIZE]) {
    fixed_t max_val = output[0];
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        if (output[i] > max_val) {
            max_val = output[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        float temp = (float)(output[i] - max_val);
        float exp_val = expf(temp);
        class_predictions[i] = exp_val;
        sum += exp_val;
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        class_predictions[i] /= sum;
    }
}

// Top-level function
void mlp_fixed_forward(hls::stream<axi_stream> &in_stream, hls::stream<axi_stream> &out_stream) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=return

    // Bind weights and biases to BRAM/LUTRAM (make sure model_params.hpp defines these arrays with proper types)
    #pragma HLS BIND_STORAGE variable=layer_0_weight type=ROM_2P impl=LUTRAM
    #pragma HLS BIND_STORAGE variable=layer_1_weight type=ROM_2P impl=LUTRAM
    #pragma HLS BIND_STORAGE variable=layer_2_weight type=ROM_2P impl=LUTRAM
    #pragma HLS BIND_STORAGE variable=layer_0_bias   type=ROM_1P impl=BRAM
    #pragma HLS BIND_STORAGE variable=layer_1_bias   type=ROM_1P impl=BRAM
    #pragma HLS BIND_STORAGE variable=layer_2_bias   type=ROM_1P impl=BRAM

    fixed_t input[INPUT_SIZE];
    fixed_t hidden1[HIDDEN_SIZE_0];
    fixed_t hidden2[HIDDEN_SIZE_1];
    fixed_t output[OUTPUT_SIZE];
    float class_predictions[OUTPUT_SIZE];

    read_stream(in_stream, input);
    compute_layer0(input, hidden1, layer_0_weight, layer_0_bias);
    compute_layer1(hidden1, hidden2, layer_1_weight, layer_1_bias);
    compute_layer2(hidden2, output, layer_2_weight, layer_2_bias);
    compute_softmax(output, class_predictions);

    float tmp = class_predictions[0];
    int label = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        if (class_predictions[i] > tmp) {
            tmp = class_predictions[i];
            label = i;
        }
    }
    write_stream(out_stream, label);
}
