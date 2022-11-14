#!/usr/bin/env python3 

import onnx 

# Load the saved model and will output a onnx.ModelProto structure 
# (A top-level file/container format for bundling a ML model)
onnx_model = onnx.load("conformer.onnx")
# Verify the model's structure and confirm that the model has a valid schema. 
# The validity of the ONNX graph is verified by checking the model's verison,
# the graph's structure, as well as the nodes and their inputs and outputs.
onnx.checker.check_model(onnx_model)