#include "model_params.hpp"
#include <hls_stream.h>
#include <math.h>
#include <ap_int.h>

// Define a helper function for 8-bit quantized ReLU.
// It scales the 32-bit accumulator, rounds, and saturates to the range [0, 127].
int8_t quantize_relu(int32_t sum, float scale) {
    float result = sum * scale;
    int result_int = (int)(result + 0.5f);
    if (result_int > 127) result_int = 127;
    if (result_int < 0) result_int = 0;
    return (int8_t)result_int;
}

// Read AXI Stream and store into input array as int8_t
// Here, we assume the incoming float is in [0,1] and we scale it to int8_t using a factor of 127.
void read_stream(hls::stream<axi_stream> &in_stream, int8_t input[INPUT_SIZE]) {
    data x;
    axi_stream inVal;
    for (int i = 0; i < INPUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        inVal = in_stream.read();
        x.intVal = inVal.data;  // reinterpret the 32-bit data as a float
        float scaled = x.floatVal * 127.0f;
        if (scaled > 127.0f) scaled = 127.0f;
        if (scaled < -128.0f) scaled = -128.0f;
        input[i] = (int8_t)(scaled + 0.5f); // round by adding 0.5
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

// Compute first hidden layer with quantized ReLU activation.
void compute_layer0(int8_t input[INPUT_SIZE],
                    int8_t hidden1[HIDDEN_SIZE],
                    const int8_t w0[HIDDEN_SIZE][INPUT_SIZE],
                    const int32_t b0[HIDDEN_SIZE],
                    const float scale0) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        int32_t sum = b0[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            #pragma HLS UNROLL
            int32_t product = input[j] * w0[i][j];
            sum += product;
        }
        hidden1[i] = quantize_relu(sum, scale0);
    }
}

// Compute second hidden layer with quantized ReLU activation.
void compute_layer1(const int8_t hidden1[HIDDEN_SIZE],
                    int8_t hidden2[HIDDEN_SIZE],
                    const int8_t w1[HIDDEN_SIZE][HIDDEN_SIZE],
                    const int32_t b1[HIDDEN_SIZE],
                    const float scale1) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        int32_t sum = b1[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            #pragma HLS UNROLL
            int32_t product = hidden1[j] * w1[i][j];
            sum += product;
        }
        hidden2[i] = quantize_relu(sum, scale1);
    }
}

// Compute output layer (no activation quantization applied here)
void compute_layer2(const int8_t hidden2[HIDDEN_SIZE],
                    int32_t output[OUTPUT_SIZE],
                    const int8_t w2[OUTPUT_SIZE][HIDDEN_SIZE],
                    const int32_t b2[OUTPUT_SIZE]) {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        int32_t sum = b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            #pragma HLS UNROLL
            int32_t product = ((int32_t)hidden2[j]) * w2[i][j];
            sum += product;
        }
        output[i] = sum;
    }
}

void compute_softmax(int32_t output[10], float class_predictions[10]) {
    int32_t max_val = output[0];
    for (int i = 1; i < 10; i++) {
        #pragma HLS PIPELINE II=1
        if (output[i] > max_val) {
            max_val = output[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < 10; i++) {
        #pragma HLS PIPELINE II=1
        float temp = (float)(output[i] - max_val);
        float exp_val = expf(temp);
        class_predictions[i] = exp_val;
        sum += exp_val;
    }

    for (int i = 0; i < 10; i++) {
        #pragma HLS PIPELINE II=1
        class_predictions[i] /= sum;
    }
}

// Top-level function
void mlp_forward(hls::stream<axi_stream> &in_stream, hls::stream<axi_stream> &out_stream) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=return

    // Bind weights and biases to BRAM
    #pragma HLS BIND_STORAGE variable=weight_layer_0 type=ROM_2P impl=BRAM
    #pragma HLS BIND_STORAGE variable=weight_layer_1 type=ROM_2P impl=BRAM
    #pragma HLS BIND_STORAGE variable=weight_layer_2 type=ROM_2P impl=BRAM
    #pragma HLS BIND_STORAGE variable=bias_layer_0   type=ROM_1P impl=BRAM
    #pragma HLS BIND_STORAGE variable=bias_layer_1   type=ROM_1P impl=BRAM
    #pragma HLS BIND_STORAGE variable=bias_layer_2   type=ROM_1P impl=BRAM

    int8_t input[INPUT_SIZE];
    int8_t hidden1[HIDDEN_SIZE];
    int8_t hidden2[HIDDEN_SIZE];
    int32_t output[OUTPUT_SIZE];
    float class_predictions[OUTPUT_SIZE];

    // Scaling factors for quantization (tune these based on calibration)
    const float scale0 = 0.001f; // Example for layer 0
    const float scale1 = 0.001f; // Example for layer 1

    read_stream(in_stream, input);
    compute_layer0(input, hidden1, weight_layer_0, bias_layer_0, scale0);
    compute_layer1(hidden1, hidden2, weight_layer_1, bias_layer_1, scale1);
    compute_layer2(hidden2, output, weight_layer_2, bias_layer_2);
    compute_softmax(output, class_predictions);

    float tmp = 0.7f;
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
