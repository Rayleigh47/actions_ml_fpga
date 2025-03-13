#include "model_params.hpp"

// Read AXI Stream and store into input array
void read_stream(hls::stream<AXI_VAL> &in_stream, float input[INPUT_SIZE]) {
    #pragma HLS INLINE
    data x;
    for (int i = 0; i < INPUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        AXI_VAL inVal = in_stream.read();
        x.intVal = inVal.data;
        input[i] = x.floatVal;
    }
}

// Write the predicted label to the AXI stream
void write_stream(hls::stream<AXI_VAL> &out_stream, int label) {
    #pragma HLS INLINE
    AXI_VAL outVal;
    outVal.data = label;
    outVal.keep = -1;  // All bytes kept
    outVal.strb = -1;  // Valid strobe
    outVal.last = 1;   // Last transaction
    out_stream.write(outVal);
}

// Compute first hidden layer with ReLU activation: input -> hidden1
void compute_layer0(float input[INPUT_SIZE], float hidden1[HIDDEN_SIZE],
                   const float w0[HIDDEN_SIZE][INPUT_SIZE], const float b0[HIDDEN_SIZE]) {
    #pragma HLS ARRAY_PARTITION variable=input complete dim=1
    #pragma HLS ARRAY_PARTITION variable=hidden1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=w0 complete dim=2

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        #pragma HLS PIPELINE II=4
        float sum = b0[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += w0[i][j] * input[j];
        }
        // Apply ReLU activation
        hidden1[i] = (sum > 0) ? sum : 0;
    }
}

// Compute second hidden layer with ReLU activation: hidden1 -> hidden2
void compute_layer1(float hidden1[HIDDEN_SIZE], float hidden2[HIDDEN_SIZE],
                   const float w1[HIDDEN_SIZE][HIDDEN_SIZE], const float b1[HIDDEN_SIZE]) {
    #pragma HLS ARRAY_PARTITION variable=hidden1 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=hidden2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=w1 complete dim=2

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        #pragma HLS PIPELINE II=4
        float sum = b1[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += w1[i][j] * hidden1[j];
        }
        // Apply ReLU activation
        hidden2[i] = (sum > 0) ? sum : 0;
    }
}

// Compute output layer: hidden2 -> output (no activation)
void compute_layer2(float hidden2[HIDDEN_SIZE], float output[OUTPUT_SIZE],
                   const float w2[OUTPUT_SIZE][HIDDEN_SIZE], const float b2[OUTPUT_SIZE]) {
    #pragma HLS ARRAY_PARTITION variable=hidden2 complete dim=1
    #pragma HLS ARRAY_PARTITION variable=output complete dim=1
    #pragma HLS ARRAY_PARTITION variable=w2 complete dim=2

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        #pragma HLS PIPELINE II=4
        float sum = b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += w2[i][j] * hidden2[j];
        }
        output[i] = sum;
    }
}

// Top-level function for the MLP forward pass
void mlp_forward(hls::stream<AXI_VAL> &in_stream, hls::stream<AXI_VAL> &out_stream) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=return
    #pragma HLS DATAFLOW

    // Local arrays for intermediate data
    float input[INPUT_SIZE];
    float hidden1[HIDDEN_SIZE];
    float hidden2[HIDDEN_SIZE];
    float output[OUTPUT_SIZE];

    // Read input data from the AXI stream
    read_stream(in_stream, input);

    // Forward pass: first hidden layer
    compute_layer0(input, hidden1, layer_0_weight, layer_0_bias);
    // Forward pass: second hidden layer
    compute_layer1(hidden1, hidden2, layer_1_weight, layer_1_bias);
    // Forward pass: output layer
    compute_layer2(hidden2, output, layer_2_weight, layer_2_bias);

    // Find the index of the maximum value (prediction)
    float max_val = -1e9; // Use a very small number for initialization
    int label = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        if (output[i] > max_val) {
            max_val = output[i];
            label = i;
        }
    }

    // Write the predicted label to the AXI stream
    write_stream(out_stream, label);
}
