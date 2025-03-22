#ifndef MLP_HPP
#define MLP_HPP

#include "hls_stream.h"
#include "ap_axi_sdata.h"
#include "ap_fixed.h"

#define INPUT_SIZE 7
#define HIDDEN_SIZE 10
#define OUTPUT_SIZE 2

/*
 * Define ap_axis data type
 * data: 32 bit data width
 * id: 2 bit used to differentiate streams
 * user: 5 bit custom metadata
 * dest: 6 bit destination of the data
 */
typedef ap_axis<32, 2, 5, 6> AXI_VAL;

/*
 * Union to represent different data type in same memory location.
 * Used to intepret int as float or vice versa.
 */
union data {
	int intVal;
	float floatVal;
};

// Functions
void compute_layer1(float input[INPUT_SIZE], float hidden[HIDDEN_SIZE], const float w1[HIDDEN_SIZE][INPUT_SIZE], const float b1[HIDDEN_SIZE]);
void compute_layer2(float hidden[HIDDEN_SIZE], float output[OUTPUT_SIZE], const float w2[OUTPUT_SIZE][HIDDEN_SIZE], const float b2[OUTPUT_SIZE]);
void mlp_forward(hls::stream<AXI_VAL> &in_stream, hls::stream<AXI_VAL> &out_stream);
int getArgMax(const float output[OUTPUT_SIZE]);

// Define constants for the weights and biases
const float weights1[10][7] = {
    {0.16251044, 1.1831999, -0.626845, 0.43591866, 0.3022716, 0.43790257, -1.8799943},
    {-0.8555495, 0.29947925, -0.16439816, -0.062306784, 0.6782942, -0.050080393, -0.9461686},
    {0.40283397, -0.7647034, -0.33183578, -1.0438976, 0.5723187, -0.18979117, 0.39654627},
    {0.56663305, 0.25609604, -0.9047133, 0.6869576, -0.65168923, 0.54859626, -0.5302369},
    {-0.80193543, 1.1064588, 0.17399597, -0.9519692, 0.69940805, 0.05454964, -0.62333703},
    {1.4825789, 0.35610628, 0.055231318, 0.273459, -0.9687046, 0.17680521, -0.74570954},
    {0.015052237, -0.66221726, 0.35071638, 2.2531307, 0.3557224, 0.07123269, 0.20233655},
    {-0.92295605, -2.0898461, 0.6678868, -0.59614927, 0.036805272, -0.1637746, 0.07165193},
    {0.39577755, 0.8745704, 1.4146948, 1.316509, 0.5153451, 0.07958634, 0.039877355},
    {-0.27141938, 0.015207972, 0.029332682, -0.7423275, -0.48469085, 0.18848704, 1.2185943},
};

const float bias1[10] = {
    0.2856868,
    0.91037405,
    0.731866,
    1.0643086,
    -0.58022755,
    -1.626256,
    -0.7288438,
    0.6130599,
    -0.12780805,
    -0.8347394,
};

const float weights2[2][10] = {
    {-1.4511024, 1.5825605, 1.0410931, 1.4319519, -1.5868354, -3.6142776, -2.15811, -2.4363382, -1.4135438, -1.905008},
    {1.4831499, -1.2229822, -0.9404305, -1.4605352, 2.072559, 3.679158, 2.36289, 1.9350693, 1.8351182, 2.2359717},
};

const float bias2[2] = {
    0.847295,
    -0.7561734,
};

#endif
