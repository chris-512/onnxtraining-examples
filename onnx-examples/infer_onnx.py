#!/usr/bin/env python3
import numpy as np
import onnxruntime

import torch
import torch.onnx

from conformer import Conformer

batch_size, sequence_length, dim = 1, 12345, 80

# Load pretrained model weights
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

inputs = torch.rand(batch_size, sequence_length, dim).to(device)
input_lengths = torch.LongTensor([12345, 12300, 12000])
targets = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                            [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                            [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]]).to(device)
target_lengths = torch.LongTensor([9, 8, 7])

model = Conformer(num_classes=10, input_dim=dim, encoder_dim=32, num_encoder_layers=3).to(device)

# set the model to inference mode
model.eval()

# Forward propagate
outputs, output_lengths = model(inputs, input_lengths)

# Export the model
torch.onnx.export(model, (inputs, input_lengths), "conformer.onnx", export_params=True, opset_version=10, do_constant_folding=True,
                    input_names=['input0', 'input1'],
                    output_names=['output0', 'output1'],
                    dynamic_axes={'input0' : {0 : 'batch_size'},    # variable length axes
                                'output0' : {0 : 'batch_size'}})

def infer(onnx_path, ins, outs):

    ort_session = onnxruntime.InferenceSession(onnx_path)

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if hasattr(tensor, 'requires_grad') and tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(ins[0]), ort_session.get_inputs()[1].name: to_numpy(ins[1])}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(outs[0]), ort_outs[0], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(to_numpy(outs[1]), ort_outs[1], rtol=1e-03, atol=1e-05)

if __name__ == '__main__':
    infer('conformer.onnx', ins=(inputs, input_lengths), outs=(outputs, output_lengths))
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")