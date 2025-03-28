#include <iostream>
#include <cmath>

#include "model_params.hpp"
#include "test_data.hpp"

bool is_equal(int a, int b) {
    return (a-b) == 0;
}

int main() {
    // // Load the weights and biases into the model
    hls::stream<axi_stream> in_stream;
    hls::stream<axi_stream> out_stream;
    // for (int i = 0; i < length; i++) {
    //     axi_stream val;
    //     val.data = weights_and_bias[i];
    //     val.last = (i == length - 1) ? 1 : 0;
    //     in_stream.write(val);
    // }
    // mlp_quantized(in_stream, out_stream);

    // // LOAD AND WRITE
    // mlp_quantized(in_stream, out_stream);

    // int weights_and_bias_out[length];
    // int c = 0;
    // for (int i = 0; i < length; i++) {
    //     axi_stream val = out_stream.read();
    //     weights_and_bias_out[i] = val.data;
    //     if (weights_and_bias_out[i] == weights_and_bias[i]) {
    //         c += 1;
    //     }
    // }
    // std::cout << "Count =" << c <<std::endl;


    // Results storage
    // float model_output[test_length][OUTPUT_SIZE];

    int model_output[test_length];
    bool all_tests_passed = true;
    int count = 0;
    // Process each test input
    for (int i = 0; i < test_length; i++) {
        // Prepare input data in AXI format
        for (int j = 0; j < INPUT_SIZE; j++) {
            axi_stream val;
            val.data = *((int*)&test_data[i][j]);
            val.last = (j == INPUT_SIZE - 1) ? 1 : 0;
            in_stream.write(val);
        }

        // Run the neural network
        mlp_quantized(in_stream, out_stream);

        axi_stream val = out_stream.read();
        model_output[i] = val.data;

        if (is_equal(test_labels[i], model_output[i])) {
        	count += 1;
        }
    }
    float accuracy = count / 1000.0 * 100;
    std::cout << "Accuracy =" << accuracy <<std::endl;
}
