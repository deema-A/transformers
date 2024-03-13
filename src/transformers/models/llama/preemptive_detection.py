import torch
import torch.nn as nn


# for i in range(len(labels)):
#     tensor = features[i].squeeze(0)
#     # remove the prompt
#     # tensor= tensor[20:, :]

#     # 0th 
#     # tensor= tensor[:1, :]
#     # tensor= tensor[:21, :]

#     # # # mid
#     # rows = tensor.shape[0]//2
#     # print("rows", rows)
#     # tensor = tensor[:rows, ]
#     # print(tensor.shape, "shape")

#     # rows = tensor[20:, :].shape[0]//2
#     # print("rows", rows)
#     # tensor = tensor[:rows+20, ]
#     # print(tensor.shape, "shape")

#     # -3
#     # tensor = tensor[:-3, ]

#     tensor = tensor.float()
#     mean_pooled = torch.mean(tensor, dim=0)
#     features[i] = mean_pooled
#     print("shape  animals_hs[i]",  features[i].shape)

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)  # Input to Hidden Layer
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)  # Hidden to Output Layer
        self.sigmoid = nn.Sigmoid()  # Since it's a binary classification

    def forward(self, x):
        out = self.fc(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out


# model = SimpleMLP(4096, 512, 1)
# model_weights = torch.load('city_llama_7b_mode_0_layer_13.pth')
# model.load_state_dict(model_weights)
# model.eval() 

input_tensor = torch.randn(4096)  # Example tensor, replace with your actual data
input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimension if necessary


with torch.no_grad():  # No need to track gradients for inference
    output = model(input_tensor)
    print(output)
    print(torch.round(output))

###################################################################
import torch

def load_model(model_path):
    # Load the pre-trained model
    model = SimpleMLP(4096, 512, 1)
    model_weights = torch.load(model_path)
    model.load_state_dict(model_weights)
    model.eval()  # Set the model to inference mode
    return model

def prepare_input_tensor():
    # Prepare your input tensor
    input_tensor = torch.randn(4096)  # Example random tensor, replace with your actual data
    input_tensor = input_tensor.unsqueeze(0)  # Add a batch dimension
    return input_tensor

def predis(model, input_tensor):
    # Perform inference
    with torch.no_grad():  # No need to track gradients for inference
        output = model(input_tensor)
    return torch.round(output)

def predis():
    model_path = 'city_llama_7b_mode_0_layer_13.pth'  # Path to your model file
    model = load_model(model_path)
    
    input_tensor = prepare_input_tensor() ## need to check model size?
    return perform_inference(model, input_tensor)
    
if __name__ == '__main__':
    main()
