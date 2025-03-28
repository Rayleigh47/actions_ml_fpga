#include "model_params.hpp"
#include <hls_stream.h>
#include <math.h>
#include <hls_math.h>
#include <ap_int.h>

// Contiguous RAM to store DMA transfer
static int32_t total_weights_bias[TOTAL_WEIGHTS_BIAS] = {0};
// Weights and biases
static int8_t weights_0[HIDDEN_SIZE][INPUT_SIZE] = {0};
static int32_t bias_0[HIDDEN_SIZE] = {0};
static int8_t weights_1[HIDDEN_SIZE][HIDDEN_SIZE] = {0};
static int32_t bias_1[HIDDEN_SIZE] = {0};
static int8_t weights_2[OUTPUT_SIZE][HIDDEN_SIZE] = {0};
static int32_t bias_2[OUTPUT_SIZE] = {0};

void read_weights_biases(hls::stream<axi_stream> &in_stream) {
    axi_stream inVal;
    for (int i = 0; i < TOTAL_WEIGHTS_BIAS; i++) {
        #pragma HLS PIPELINE off
        inVal = in_stream.read();
        int32_t val = inVal.data;
        total_weights_bias[i] = val;
    }
}

void init_layers() {
    int index = 0;
    // Load first layer weights
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            weights_0[i][j] = total_weights_bias[index++];
        }
    }
    // Load first layer biases
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        bias_0[i] = total_weights_bias[index++];
    }
    // Load second layer weights
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            weights_1[i][j] = total_weights_bias[index++];
        }
    }
    // Load second layer biases
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        bias_1[i] = total_weights_bias[index++];
    }
    // Load third layer weights
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            weights_2[i][j] = total_weights_bias[index++];
        }
    }
    // Load third layer biases
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        bias_2[i] = total_weights_bias[index++];
    }
}

void write_weights_biases(hls::stream<axi_stream> &out_stream) {
    axi_stream outVal;
    // Write first layer weights
    for (int i = 0; i < HIDDEN_SIZE; i++) {
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
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        #pragma HLS PIPELINE OFF
        outVal.data = bias_0[i];
        outVal.keep = -1;
        outVal.strb = -1;
        outVal.last = 0;
        out_stream.write(outVal);
    }
    // Write second layer weights
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        #pragma HLS PIPELINE OFF
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            #pragma HLS PIPELINE OFF
            outVal.data = weights_1[i][j];
            outVal.keep = -1;
            outVal.strb = -1;
            outVal.last = 0;
            out_stream.write(outVal);
        }
    }
    // Write second layer biases
    for (int i = 0; i < HIDDEN_SIZE; i++) {
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
        for (int j = 0; j < HIDDEN_SIZE; j++) {
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

int8_t quantize_relu(int32_t sum, float scale) {
    float result = sum * scale;
    int result_int = (int)(result + 0.5f);
    if (result_int > 127) result_int = 127;
    if (result_int < 0) result_int = 0;
    return (int8_t)result_int;
}

void inference(int8_t input[INPUT_SIZE],
               int8_t hidden_1[HIDDEN_SIZE],
               int8_t hidden_2[HIDDEN_SIZE],
               int32_t output[OUTPUT_SIZE],
               int8_t weights_0[HIDDEN_SIZE][INPUT_SIZE],
               int8_t weights_1[HIDDEN_SIZE][HIDDEN_SIZE],
               int8_t weights_2[OUTPUT_SIZE][HIDDEN_SIZE],
               int32_t bias_0[HIDDEN_SIZE],
               int32_t bias_1[HIDDEN_SIZE],
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
                    int8_t hidden1[HIDDEN_SIZE],
                    int8_t w0[HIDDEN_SIZE][INPUT_SIZE],
                    int32_t b0[HIDDEN_SIZE],
                    float scale_0) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        #pragma HLS PIPELINE OFF
        int32_t sum = b0[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            #pragma HLS UNROLL
            int32_t product = input[j] * w0[i][j];
            sum += product;
        }
        hidden1[i] = quantize_relu(sum, scale_0);
    }
}

void compute_layer1(int8_t hidden1[HIDDEN_SIZE],
                    int8_t hidden2[HIDDEN_SIZE],
                    int8_t w1[HIDDEN_SIZE][HIDDEN_SIZE],
                    int32_t b1[HIDDEN_SIZE],
                    float scale_1) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        #pragma HLS PIPELINE OFF
        int32_t sum = b1[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            #pragma HLS UNROLL
            int32_t product = hidden1[j] * w1[i][j];
            sum += product;
        }
        hidden2[i] = quantize_relu(sum, scale_1);
    }
}

void compute_layer2(int8_t hidden2[HIDDEN_SIZE],
                    int32_t output[OUTPUT_SIZE],
                    int8_t w2[OUTPUT_SIZE][HIDDEN_SIZE],
                    int32_t b2[OUTPUT_SIZE]) {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        #pragma HLS PIPELINE OFF
        int32_t sum = b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
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
void mlp_quantized(hls::stream<axi_stream> &in_stream, hls::stream<axi_stream> &out_stream) {
    // Instantiate top level interface
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    int8_t input[INPUT_SIZE];
    int8_t hidden_1[HIDDEN_SIZE];
    int8_t hidden_2[HIDDEN_SIZE];
    int32_t output[OUTPUT_SIZE];
    float class_predictions[OUTPUT_SIZE];

    // Scaling factors for quantization (tune these based on calibration)
    const float scale_0 = 0.000030655f / 0.015625f; // From ONNX file
    const float scale_1 = 0.000061035f / 0.015625f; // From ONNX file

    
    read_weights_biases(in_stream);
    init_layers();
    // write_weights_biases(out_stream);
    for (int i = 0; i < 1000; i ++) {
        read_inputs(in_stream, input);
        inference(input, hidden_1, hidden_2, output, weights_0, weights_1, weights_2, bias_0, bias_1, bias_2, scale_0, scale_1, class_predictions);
        send_outputs(out_stream, class_predictions);
    }

}
