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
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})