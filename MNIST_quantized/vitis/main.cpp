#include "model_params.hpp"
#include <hls_stream.h>
#include <math.h>
#include <hls_math.h>
#include <ap_int.h>

// Contiguous RAM to store DMA transfer
// static int32_t total_weights_bias[TOTAL_WEIGHTS_BIAS] = {0};
// static int8_t weights_0[HIDDEN_SIZE_0][INPUT_SIZE] = {0};
// static int8_t weights_1[HIDDEN_SIZE_1][HIDDEN_SIZE_0] = {0};
// static int8_t weights_2[OUTPUT_SIZE][HIDDEN_SIZE_1] = {0};
// static int32_t bias_0[HIDDEN_SIZE_0] = {0};
// static int32_t bias_1[HIDDEN_SIZE_1] = {0};
// static int32_t bias_2[OUTPUT_SIZE] = {0};

// void read_weights_biases(hls::stream<axi_stream> &in_stream) {
//     axi_stream inVal;
//     for (int i = 0; i < TOTAL_WEIGHTS_BIAS; i++) {
//         #pragma HLS PIPELINE off
//         inVal = in_stream.read();
//         int32_t val = inVal.data;
//         total_weights_bias[i] = val;
//     }
// }

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

void write_weights_biases(hls::stream<axi_stream> &out_stream) {
    axi_stream outVal;
    // Write first layer weights
    for (int i = 0; i < HIDDEN_SIZE_0; i++) {
        #pragma HLS PIPELINE OFF
        for (int j = 0; j < INPUT_SIZE; j++) {
            #pragma HLS PIPELINE OFF
            outVal.data = weights_0[i][j];
            outVal.keep = -1;
            outVal.strb = -1;
            outVal.last = 0;
            out_stream.write(outVal);
        }
    }
    // Write first layer biases
    for (int i = 0; i < HIDDEN_SIZE_0; i++) {
        #pragma HLS PIPELINE OFF
        outVal.data = bias_0[i];
        outVal.keep = -1;
        outVal.strb = -1;
        outVal.last = 0;
        out_stream.write(outVal);
    }
    // Write second layer weights
    for (int i = 0; i < HIDDEN_SIZE_1; i++) {
        #pragma HLS PIPELINE OFF
        for (int j = 0; j < HIDDEN_SIZE_0; j++) {
            #pragma HLS PIPELINE OFF
            outVal.data = weights_1[i][j];
            outVal.keep = -1;
            outVal.strb = -1;
            outVal.last = 0;
            out_stream.write(outVal);
        }
    }
    // Write second layer biases
    for (int i = 0; i < HIDDEN_SIZE_1; i++) {
        #pragma HLS PIPELINE OFF
        outVal.data = bias_1[i];
        outVal.keep = -1;
        outVal.strb = -1;
        outVal.last = 0;
        out_stream.write(outVal);
    }
    // Write third layer weights
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        #pragma HLS PIPELINE OFF
        for (int j = 0; j < HIDDEN_SIZE_1; j++) {
            #pragma HLS PIPELINE OFF
            outVal.data = weights_2[i][j];
            outVal.keep = -1;
            outVal.strb = -1;
            outVal.last = 0;
            out_stream.write(outVal);
        }
    }
    // Write third layer biases
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        #pragma HLS PIPELINE OFF
        outVal.data = bias_2[i];
        outVal.keep = -1;
        outVal.strb = -1;
        outVal.last = (i == OUTPUT_SIZE - 1 ? 1 : 0);
        out_stream.write(outVal);
    }
}

void read_inputs(hls::stream<axi_stream> &in_stream, int8_t input[INPUT_SIZE]) {
    data x;
    axi_stream inVal;
    for (int i = 0; i < INPUT_SIZE; i++) {
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
               float class_predictions[OUTPUT_SIZE]) {
    compute_layer0(input, hidden_1, weights_0, bias_0, scale_0);
    compute_layer1(hidden_1, hidden_2, weights_1, bias_1, scale_1);
    compute_layer2(hidden_2, output, weights_2, bias_2);
    compute_softmax(output, class_predictions);
}

void compute_layer0(int8_t input[INPUT_SIZE],
                    int8_t hidden1[HIDDEN_SIZE_0],
                    int8_t w0[HIDDEN_SIZE_0][INPUT_SIZE],
                    int32_t b0[HIDDEN_SIZE_0],
                    float scale_0) {
    for (int i = 0; i < HIDDEN_SIZE_0; i++) {
        #pragma HLS PIPELINE OFF
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
        #pragma HLS PIPELINE OFF
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
        #pragma HLS PIPELINE OFF
        int32_t sum = b2[i];
        for (int j = 0; j < HIDDEN_SIZE_1; j++) {
            #pragma HLS UNROLL
            int32_t product = ((int32_t)hidden2[j]) * w2[i][j];
            sum += product;
        }
        output[i] = sum;
    }
}

void compute_softmax(int32_t output[OUTPUT_SIZE], float class_predictions[OUTPUT_SIZE]) {
    int32_t max_val = output[0];
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

void send_outputs(hls::stream<axi_stream> &out_stream, float class_predictions[OUTPUT_SIZE]) {
    float tmp = 0.8f; // 80% threshold
    int label = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        if (class_predictions[i] > tmp) {
            tmp = class_predictions[i];
            label = i;
        }
    }
    axi_stream outVal;
    outVal.data = label;
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
    float class_predictions[OUTPUT_SIZE];

    // Scaling factors for quantization (tune these based on calibration)
    // Tanh
    const float scale_0 = 0.0002452194457873702f / 0.0078125f; // From ONNX file
    const float scale_1 = 0.000244140625f / 0.0078125f; // From ONNX file

    
    // // Uncomment for actual implementation (latency will look super big because it measures over 10000 inferences)
    // // read_weights_biases(in_stream);
    // init_layers(in_stream);
    // // write_weights_biases(out_stream);
    // for (int i = 0; i < 10000; i ++) {
    //     read_inputs(in_stream, input);
    //     inference(input, hidden_1, hidden_2, output, weights_0, weights_1, weights_2, bias_0, bias_1, bias_2, scale_0, scale_1, class_predictions);
    //     send_outputs(out_stream, class_predictions);
    // }

    // Uncomment for running testbench with test.cpp/ inference speed results
    read_inputs(in_stream, input);
    inference(input, hidden_1, hidden_2, output, weights_0, weights_1, weights_2, bias_0, bias_1, bias_2, scale_0, scale_1, class_predictions);
    send_outputs(out_stream, class_predictions);

}