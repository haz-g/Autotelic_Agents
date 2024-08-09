import torch
import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=5):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # He initialization
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)  # Use softmax for action probabilities

    def get_parameters(self):
        return {name: param.data.clone() for name, param in self.named_parameters()}

    def set_parameters(self, parameters):
        with torch.no_grad():
            for name, param in self.named_parameters():
                param.copy_(parameters[name])

    def mutate_parameters(self, mutation_strength=0.3):
        with torch.no_grad():
            for param in self.parameters():
                noise = torch.randn_like(param) * mutation_strength
                param.add_(noise)