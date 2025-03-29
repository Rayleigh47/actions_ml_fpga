#include "model_params.hpp"
#include <hls_stream.h>
#include <math.h>
#include <hls_math.h>
#include <ap_int.h>

void init_layers(hls::stream<axi_stream> &in_stream) {
    axi_stream inVal;
    // Load first layer weights
    for (int i = 0; i < HIDDEN_SIZE_0; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            inVal = in_stream.read();
            int32_t val = inVal.data;
            weights_0[i][j] = int8_t (val);
        }
    }
    // Load first layer biases
    for (int i = 0; i < HIDDEN_SIZE_0; i++) {
        inVal = in_stream.read();
        int32_t val = inVal.data;
        bias_0[i] = int32_t (val);
    }
    // Load second layer weights
    for (int i = 0; i < HIDDEN_SIZE_1; i++) {
        for (int j = 0; j < HIDDEN_SIZE_0; j++) {
            inVal = in_stream.read();
            int32_t val = inVal.data;
            weights_1[i][j] = int8_t (val);
        }
    }
    // Load second layer biases
    for (int i = 0; i < HIDDEN_SIZE_1; i++) {
        inVal = in_stream.read();
        int32_t val = inVal.data;
        bias_1[i] = int32_t (val);
    }
    // Load third layer weights
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE_1; j++) {
            inVal = in_stream.read();
            int32_t val = inVal.data;
            weights_2[i][j] = int8_t (val);
        }
    }
    // Load third layer biases
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        inVal = in_stream.read();
        int32_t val = inVal.data;
        bias_2[i] = int32_t (val);
    }
}

void read_inputs(hls::stream<axi_stream> &in_stream, int8_t input[INPUT_SIZE]) {
    data x;
    axi_stream inVal;
    for (int i = 0; i < INPUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        inVal = in_stream.read();
        x.intVal = inVal.data;
        float scaled = x.floatVal * 127.0f;
        if (scaled > 127.0f) {
            scaled = 127.0f;
        }
        if (scaled < -128.0f) {
            scaled = -128.0f;
        }
        input[i] = (int8_t) (scaled + 0.5f);
    }
}

// Tanh approximation function for x in [-1, 1]. Outside this range, we saturate.
float tanh_approx(float x) {
    if (x < -1.0f) return -1.0f;
    else if (x > 1.0f) return 1.0f;
    else return x * (27.0f + x * x) / (27.0f + 9.0f * x * x);
}

int8_t quantize_tanh(int32_t sum, float scale) {
    // Scale the accumulated sum
    float x = sum * scale;
    // Use the tanh approximation
    float y = tanh_approx(x);
    // Map the output from [-1, 1] to [-128, 127]
    int result_int = (int)(y * 127.0f + (y >= 0 ? 0.5f : -0.5f));
    // Clamp to int8_t range
    if (result_int > 127) result_int = 127;
    if (result_int < -128) result_int = -128;
    return (int8_t)result_int;
}

int32_t inference(int8_t input[INPUT_SIZE],
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
               float scale_1) {
    compute_layer0(input, hidden_1, weights_0, bias_0, scale_0);
    compute_layer1(hidden_1, hidden_2, weights_1, bias_1, scale_1);
    compute_layer2(hidden_2, output, weights_2, bias_2);
    return compute_argmax(output);

}

void compute_layer0(int8_t input[INPUT_SIZE],
                    int8_t hidden1[HIDDEN_SIZE_0],
                    int8_t w0[HIDDEN_SIZE_0][INPUT_SIZE],
                    int32_t b0[HIDDEN_SIZE_0],
                    float scale_0) {
    for (int i = 0; i < HIDDEN_SIZE_0; i++) {
        #pragma HLS PIPELINE II=8
        int32_t sum = b0[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            #pragma HLS UNROLL
            int32_t product = input[j] * w0[i][j];
            sum += product;
        }
        hidden1[i] = quantize_tanh(sum, scale_0);
    }
}

void compute_layer1(int8_t hidden1[HIDDEN_SIZE_0],
                    int8_t hidden2[HIDDEN_SIZE_1],
                    int8_t w1[HIDDEN_SIZE_1][HIDDEN_SIZE_0],
                    int32_t b1[HIDDEN_SIZE_1],
                    float scale_1) {
    for (int i = 0; i < HIDDEN_SIZE_1; i++) {
        #pragma HLS PIPELINE II=7
        int32_t sum = b1[i];
        for (int j = 0; j < HIDDEN_SIZE_0; j++) {
            #pragma HLS UNROLL
            int32_t product = hidden1[j] * w1[i][j];
            sum += product;
        }
        hidden2[i] = quantize_tanh(sum, scale_1);
    }
}

void compute_layer2(int8_t hidden2[HIDDEN_SIZE_1],
                    int32_t output[OUTPUT_SIZE],
                    int8_t w2[OUTPUT_SIZE][HIDDEN_SIZE_1],
                    int32_t b2[OUTPUT_SIZE]) {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        int32_t sum = b2[i];
        for (int j = 0; j < HIDDEN_SIZE_1; j++) {
            #pragma HLS UNROLL
            int32_t product = ((int32_t)hidden2[j]) * w2[i][j];
            sum += product;
        }
        output[i] = sum;
    }
}

int32_t compute_argmax(int32_t output[OUTPUT_SIZE]) {
    int32_t current_max = output[0];
    int32_t max_index = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        if (output[i] > current_max) {
            current_max = output[i];
            max_index = i;
        }
    }
    return max_index;
}

void send_outputs(hls::stream<axi_stream> &out_stream, int32_t max_index) {
    axi_stream outVal;
    outVal.data = max_index;
    outVal.keep = -1;
    outVal.strb = -1;
    outVal.last = 1;
    out_stream.write(outVal);
}

// Top-level function
void mnist_quantized(hls::stream<axi_stream> &in_stream, hls::stream<axi_stream> &out_stream) {
    // Instantiate top level interface
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    int8_t input[INPUT_SIZE];
    int8_t hidden_1[HIDDEN_SIZE_0];
    int8_t hidden_2[HIDDEN_SIZE_1];
    int32_t output[OUTPUT_SIZE];
    static int i = 0;

    // Uncomment for actual implementation (latency will look super big because it measures over 10000 inferences)
    // init_layers(in_stream);
    // for (int i = 0; i < 10000; i ++) {
    //     read_inputs(in_stream, input);
    //     int32_t max_index = inference(input, hidden_1, hidden_2, output, weights_0, weights_1, weights_2, bias_0, bias_1, bias_2, scale_0, scale_1);
    //     send_outputs(out_stream, max_index);
    // }

    // Uncomment for running testbench with test.cpp/ inference speed results
    // read_inputs(in_stream, input);
    // int32_t max_index = inference(input, hidden_1, hidden_2, output, weights_0, weights_1, weights_2, bias_0, bias_1, bias_2, scale_0, scale_1);
    // send_outputs(out_stream, max_index);

    if (i == 0) {
        init_layers(in_stream);
        i = 1;
    } else {
        read_inputs(in_stream, input);
        int32_t max_index = inference(input, hidden_1, hidden_2, output, weights_0, weights_1, weights_2, bias_0, bias_1, bias_2, scale_0, scale_1);
        send_outputs(out_stream, max_index);
    }

}