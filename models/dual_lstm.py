import torch.nn as nn
from torch.nn import functional as f


class DualLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dr=0.0):
        super(DualLSTM, self).__init__()                        
        # self.bn1 = nn.BatchNorm1d(input_size, affine=False)    # Batch Normalization                   
        self.linearIn = nn.Linear(input_size, hidden_size)       # Linear Fully Connected Input Layer
        self.lstm1 = nn.LSTM(
            hidden_size,
            hidden_size,
            1,
            dropout=dr,
        )                                                        # LSTM Layer 1
        self.lstm2 = nn.LSTM(
            hidden_size,
            hidden_size,
            1,
            dropout=dr,
        )                                                        # LSTM Layer 2
        self.linearOut = nn.Linear(hidden_size, output_size)     # Linear Fully Connected Output Layer

    def forward(self, x):                                        # Forward propagation function
        x0 = f.leaky_relu(self.linearIn(x))
        out_lstm1, (hn, cn) = self.lstm1(x0)
        out_lstm2, (hn, cn) = self.lstm2(out_lstm1)
        return self.linearOut(out_lstm2)                         # LinearIn -> ReLU -> LSTM1 -> LSTM2 -> LinearOut


class DualLSTMForecast(DualLSTM):
    def __init__(self, input_size, output_size, hidden_size, forecast_length, dr=0.0):
        super(DualLSTMForecast, self).__init__(input_size, output_size, hidden_size, dr)
        self.forecast_length = forecast_length
        self.output_size = output_size

    def forward(self, x):
        # Call the forward method of the parent class to obtain the complete output
        full_output = super(DualLSTMForecast, self).forward(x)

        # Reshape output
        forecast_output = full_output[-self.forecast_length:, :, :]
        return forecast_output

