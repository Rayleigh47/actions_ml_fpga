## Overview
Project to train various machine learning models on PyTorch to deploy on an FPGA accelerator using HLS to speed up the implementation and deployment of the model.

## Structure
1. actions: Raw data used for training and testing.
2. inference: Loads `model.pth`, launches bluepy bluetooth relay node to connect to hardware sensor, currently runs the MLP model for easy inference.
3. MLP: PyTorch MLP model. Generates `model_params.hpp` and also has the corresponding Vitis code for HLS IP.
4. MLP_quantized: PyTorch + Brevitas quantized MLP model. Generates `model_params.hpp` and also has the corresponding Vitis code for HLS IP.

## Install
Run in a virtual environment with dependencies installed from `requirements.txt`.
