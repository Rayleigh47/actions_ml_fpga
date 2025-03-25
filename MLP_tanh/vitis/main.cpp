#include "model_params.hpp"
#include <hls_stream.h>
#include <math.h>
#include <hls_math.h>
#include <ap_int.h>
#include <ap_fixed.h>

fixed_t tanh_activation(fixed_t x) {
    // Convert the fixed-point input to float
    float x_float = (float) x;
    // Use hls_math's tanh function
    float tanhf_val = hls::tanh(x_float);
    // Convert the result back to fixed_t
    return (fixed_t) tanhf_val;
}

void read_stream(hls::stream<axi_stream> &in_stream, fixed_t input[INPUT_SIZE]) {
    data x;
    axi_stream inVal;
    for (int i = 0; i < INPUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        inVal = in_stream.read();
        x.intVal = inVal.data;  
        // Direct conversion from float to fixed_t without scaling
        fixed_t val = x.floatVal;
        input[i] = val;
    }
}

void read_weights_biases(hls::stream<axi_stream> &in_stream, 
                         weights_t w0[HIDDEN_SIZE][INPUT_SIZE],
                         weights_t w1[HIDDEN_SIZE][HIDDEN_SIZE], 
                         weights_t w2[OUTPUT_SIZE][HIDDEN_SIZE],
                         fixed_t b0[HIDDEN_SIZE], 
                         fixed_t b1[HIDDEN_SIZE], 
                         fixed_t b2[OUTPUT_SIZE]) {
    data x;
    axi_stream inVal;

    // Load first layer weights
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            #pragma HLS PIPELINE II=1
            inVal = in_stream.read();
            x.intVal = inVal.data;  
            // Direct conversion from float to fixed_t without scaling
            fixed_t val = x.floatVal;
            w0[i][j] = val;
        }
    }
    // Load first layer biases
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        inVal = in_stream.read();
        x.intVal = inVal.data;
        fixed_t val = x.floatVal;
        b0[i] = val;
    }

    // Load second layer weights
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            #pragma HLS PIPELINE II=1
            inVal = in_stream.read();
            x.intVal = inVal.data;
            fixed_t val = x.floatVal;
            w1[i][j] = val;
        }
    }
    // Load second layer biases
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        inVal = in_stream.read();
        x.intVal = inVal.data;
        fixed_t val = x.floatVal;
        b1[i] = val;
    }

    // Load third layer weights
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            #pragma HLS PIPELINE II=1
            inVal = in_stream.read();
            x.intVal = inVal.data;
            fixed_t val = x.floatVal;
            w2[i][j] = val;
        }
    }
    // Load third layer biases
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        inVal = in_stream.read();
        x.intVal = inVal.data;
        fixed_t val = x.floatVal;
        b2[i] = val;
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
void compute_layer0(fixed_t input[INPUT_SIZE],
                    fixed_t hidden1[HIDDEN_SIZE],
                    const weights_t w0[HIDDEN_SIZE][INPUT_SIZE],
                    const fixed_t b0[HIDDEN_SIZE]) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        fixed_t sum = b0[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            #pragma HLS UNROLL
            sum += input[j] * w0[i][j];
        }
        hidden1[i] = tanh_activation(sum);
    }
}

// Compute second hidden layer with quantized ReLU activation.
void compute_layer1(const fixed_t hidden1[HIDDEN_SIZE],
                    fixed_t hidden2[HIDDEN_SIZE],
                    const weights_t w1[HIDDEN_SIZE][HIDDEN_SIZE],
                    const fixed_t b1[HIDDEN_SIZE]) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        fixed_t sum = b1[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            #pragma HLS UNROLL
            sum += hidden1[j] * w1[i][j];
        }
        hidden2[i] = tanh_activation(sum);
    }
}

// Compute output layer (no activation quantization applied here)
void compute_layer2(const fixed_t hidden2[HIDDEN_SIZE],
                    fixed_t output[OUTPUT_SIZE],
                    const weights_t w2[OUTPUT_SIZE][HIDDEN_SIZE],
                    const fixed_t b2[OUTPUT_SIZE]) {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        fixed_t sum = b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            #pragma HLS UNROLL
            sum += hidden2[j] * w2[i][j];
        }
        output[i] = sum;
    }
}

void compute_softmax(fixed_t output[OUTPUT_SIZE], float class_predictions[OUTPUT_SIZE]) {
    // Rescale the output layer to float
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
        float exp_val = hls::expf(temp);
        class_predictions[i] = exp_val;
        sum += exp_val;
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        class_predictions[i] /= sum;
    }
}

// Top-level function
void mlp_tanh_forward(hls::stream<axi_stream> &in_stream, 
                       hls::stream<axi_stream> &out_stream,
                       volatile int mode) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=mode bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    fixed_t hidden1[HIDDEN_SIZE];
    fixed_t hidden2[HIDDEN_SIZE];
    fixed_t input[INPUT_SIZE];
    fixed_t output[OUTPUT_SIZE];
    float class_predictions[OUTPUT_SIZE];

    // Weights and biases
    // static weights_t network_0_weight[HIDDEN_SIZE][INPUT_SIZE];
    // static weights_t network_2_weight[HIDDEN_SIZE][HIDDEN_SIZE];
    // static weights_t network_4_weight[OUTPUT_SIZE][HIDDEN_SIZE];
    // static fixed_t network_0_bias[HIDDEN_SIZE];
    // static fixed_t network_2_bias[HIDDEN_SIZE];
    // static fixed_t network_4_bias[OUTPUT_SIZE];

    if (mode == 1) {
       // Store weights & biases
       read_weights_biases(in_stream, network_0_weight, network_2_weight, network_4_weight, network_0_bias, network_2_bias, network_4_bias);
    }
    else {
        // Perform inference
        read_stream(in_stream, input);
        compute_layer0(input, hidden1, network_0_weight, network_0_bias);
        compute_layer1(hidden1, hidden2, network_2_weight, network_2_bias);
        compute_layer2(hidden2, output, network_4_weight, network_4_bias);
        compute_softmax(output, class_predictions);

        // Determine the predicted label
        int label = 0;
        float max_val = class_predictions[0];
        for (int i = 1; i < OUTPUT_SIZE; i++) {
            #pragma HLS PIPELINE II=1
            if (class_predictions[i] > max_val) {
                max_val = class_predictions[i];
                label = i;
            }
        }
        write_stream(out_stream, label);
    }
}
