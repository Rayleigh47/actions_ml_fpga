#include "mlp.hpp"

// Read AXI Stream
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

// Write AXI Stream
void write_stream(hls::stream<AXI_VAL> &out_stream, int label) {
    #pragma HLS INLINE
    AXI_VAL outVal;
    outVal.data = label;
    outVal.keep = -1;  // Ensure all bytes are kept
    outVal.strb = -1;  // Ensure valid strobe
    outVal.last = 1;   // Last transaction
    out_stream.write(outVal);
}

// Compute hidden layer with ReLU activation
void compute_layer1(float input[INPUT_SIZE], float hidden[HIDDEN_SIZE],
                   const float w1[HIDDEN_SIZE][INPUT_SIZE], const float b1[HIDDEN_SIZE]) {
    #pragma HLS ARRAY_PARTITION variable=input complete dim=1
    #pragma HLS ARRAY_PARTITION variable=hidden complete dim=1
    #pragma HLS ARRAY_PARTITION variable=w1 complete dim=2

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        float sum = b1[i];

        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += w1[i][j] * input[j];
        }
        hidden[i] = (sum > 0) ? sum : 0; // ReLU activation
    }
}

// Compute output layer
void compute_layer2(float hidden[HIDDEN_SIZE], float output[OUTPUT_SIZE],
                   const float w2[OUTPUT_SIZE][HIDDEN_SIZE], const float b2[OUTPUT_SIZE]) {
    #pragma HLS ARRAY_PARTITION variable=hidden complete dim=1
    #pragma HLS ARRAY_PARTITION variable=output complete dim=1
    #pragma HLS ARRAY_PARTITION variable=w2 complete dim=2

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        #pragma HLS PIPELINE II=1
        float sum = b2[i];

        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += w2[i][j] * hidden[j];
        }
        output[i] = sum;
    }
}

// Top-level function
void mlp_forward(hls::stream<AXI_VAL> &in_stream, hls::stream<AXI_VAL> &out_stream) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=return
    #pragma HLS DATAFLOW
    
    // Local arrays for internal computations
    float input[INPUT_SIZE];
    float hidden[HIDDEN_SIZE];
    float output[OUTPUT_SIZE];

    // Read input from AXI stream
    read_stream(in_stream, input);

    // Forward pass computation
    compute_layer1(input, hidden, weights1, bias1);
    compute_layer2(hidden, output, weights2, bias2);

    // Select action with highest probability.
    float threshold = -1e9; // Ensure we pick the largest value
    int label = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        if (output[i] > threshold) {
            threshold = output[i];
            label = i;
        }
    }

    // Send label to AXI stream
    write_stream(out_stream, label);
}
