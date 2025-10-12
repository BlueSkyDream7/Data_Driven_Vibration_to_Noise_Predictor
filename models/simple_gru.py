import torch.nn as nn
from torch.nn import functional as f


class SimpleGRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dr=0.0):
        super(SimpleGRU, self).__init__()                       
        # self.bn1 = nn.BatchNorm1d(input_size, affine=False)   # Batch Normalization
        self.linearIn = nn.Linear(input_size, hidden_size)      # Linear Fully Connected Input Layer
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            1,
            dropout=dr,
        )                                                       # GRU Layer
        self.linearOut = nn.Linear(hidden_size, output_size)    # Linear Fully Connected Output Layer

    def forward(self, x):                                       # Forward propagation function
        x0 = f.leaky_relu(self.linearIn(x))
        out_gru, hn = self.gru(x0)
        return self.linearOut(out_gru)                          # LinearIn -> ReLU -> GRU -> LinearOut


class SimpleGRUForecast(SimpleGRU):
    def __init__(self, input_size, output_size, hidden_size, forecast_length, dr=0.0):
        super(SimpleGRUForecast, self).__init__(input_size, output_size, hidden_size, dr)
        self.forecast_length = forecast_length
        self.output_size = output_size

    def forward(self, x):
        # Call the forward method of the parent class to obtain the complete output
        full_output = super(SimpleGRUForecast, self).forward(x)

        # Reshape output
        forecast_output = full_output[-self.forecast_length:, :, :]
        return forecast_output

