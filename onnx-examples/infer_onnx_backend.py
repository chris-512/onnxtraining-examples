import numpy as np 
import onnx
import onnxruntime.backend as backend 

from onnxruntime import datasets, get_device
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument


batch_size, sequence_length, dim = 1, 12345, 80
x = np.random.randn(batch_size, sequence_length, dim).astype(np.float32)
x_lengths = np.array([12345, 12300, 12000], dtype=np.int64)
inputs = [x, x_lengths]

model = onnx.load('conformer.onnx')
prepared = backend.prepare(model, device=get_device())

try: 
    # If multiple inputs are required, inputs should be in list.
    outputs = prepared.run(inputs)
    print("out[0]={}".format(outputs[0]))
    print("out[1]={}".format(outputs[1]))
except (RuntimeError, InvalidArgument) as e:
    print(e)