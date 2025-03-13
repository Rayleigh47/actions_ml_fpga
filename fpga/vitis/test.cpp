#include <iostream>
#include <cmath>

#include "mlp.hpp"
#include "test_data.hpp"

typedef ap_axis<32, 2, 5, 6> AXI_VAL;

// Utility function to check if two floats are approximately equal
//bool is_approx_equal(float a, float b, float tolerance = 0.00001f) {
//    return std::fabs(a - b) <= tolerance;
//}

bool is_equal(int a, int b) {
    return (a-b) == 0;
}

int main() {
    // Results storage
    // float model_output[test_length][OUTPUT_SIZE];
    int model_output[test_length];
    bool all_tests_passed = true;

    // Process each test input
    for (int i = 0; i < test_length; i++) {
        // Create input and output streams
        hls::stream<AXI_VAL> in_stream;
        hls::stream<AXI_VAL> out_stream;

        // Prepare input data in AXI format
        for (int j = 0; j < INPUT_SIZE; j++) {
            AXI_VAL val;
            val.data = *((int*)&test_data[i][j]);
            val.last = (j == INPUT_SIZE - 1) ? 1 : 0;
            in_stream.write(val);
        }

        // Run the neural network
        mlp_forward(in_stream, out_stream);

        AXI_VAL val = out_stream.read();
        model_output[i] = val.data;

        if (!is_equal(test_labels[i], model_output[i])) {
        	all_tests_passed = false;
        }
        std::cout << "Expected " << test_labels[i] << ", got " << model_output[i] << std::endl;

//        // Get output data
//        for (int j = 0; j < OUTPUT_SIZE; j++) {
//            AXI_VAL val = out_stream.read();
//            model_output[i][j] = *((float*)&val.data);
//        }
//
//        // Verify results for this test case
//        bool test_passed = true;
//        for (int j = 0; j < OUTPUT_SIZE; j++) {
//            if (!is_approx_equal(test_output[i][j], model_output[i][j])) {
//                test_passed = false;
//                all_tests_passed = false;
//                std::cout << "Test " << i << " failed at output " << j << ": ";
//                std::cout << "Expected " << test_output[i][j] << ", got " << model_output[i][j] << std::endl;
//            }
//        }
//
//        if (test_passed) {
//            std::cout << "Test " << i << " passed." << std::endl;
//        }
    }
    
    // Final test result
    if (all_tests_passed) {
        std::cout << "All tests passed!" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests failed." << std::endl;
        return 1;
    }
}
