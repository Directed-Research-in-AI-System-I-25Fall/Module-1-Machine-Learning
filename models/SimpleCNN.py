import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    # TODO: Implement Simple CNN. You are allowed to use two conv layers (and pooling layers) and two linear layers.
    # NOTE: forward_map is used in Part 2.
    
    def __init__(self, conv_dim1, conv_dim2, fc_dim, dropout_rate):
        '''
        Args:
            conv_dim1 (int): Number of output channels for the first convolutional layer. 
            conv_dim2 (int): Number of output channels for the second convolutional layer.
            fc_dim (int): Number of channels in fully connected layer. 
            dropout_rate (float): Dropout rate to apply to prevent overfitting. 
        '''
        raise NotImplementedError()
        
    def name(self):
        return "SimpleCNN"

    def forward(self, x):
        '''
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32).

            Returns:
                torch.Tensor: Output tensor of shape (batch_size, 10), containing the logits for class predictions.
        '''
        raise NotImplementedError()
    
    def forward_map(self, x):
        '''
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, 3, 32, 32).
                
            Returns:
                torch.Tensor: feature maps of shape (batch_size, hidden_channel, hidden_h, hidden_w).
        '''
        raise NotImplementedError()
    
    
