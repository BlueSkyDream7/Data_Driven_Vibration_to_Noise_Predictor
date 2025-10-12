import torch
import sys
import os
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from models.simple_lstm import SimpleLSTMForecast
from models.simple_gru import SimpleGRUForecast
from models.simple_RNN import SimpleRNNForecast
from models.dual_lstm import DualLSTMForecast
from dataset.MATLAB_Dataset import MatDataset
from eval.eval import test_eval_model


base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)


def test_train_model():
    is_new: bool = True                      # True: Create a new NN model    False: Load pre-trained model 
    abla: bool = True                        # True: All vibration acceleration signals    False: Ablate some vibration acceleration signals
    train_data_in_memory: bool = False       # True: Load training dataset to memory (fast)    False: Do not load training dataset to memory (slow)
    eval_data_in_memory: bool = False        # True: Load evaluation dataset to memory (fast)    False: Do not load evaluation dataset to memory (slow)
    window_size: int = 2560                  # Ear perceives frequencies: 20Hz to 20480Hz (at least 20Hz)
    forecast_length: int = 2560              # Predict length: L
    hidden_size: int = 128                   # Hidden size for hidden layer
    batch_size: int = 32                     # 1 batch = 1.6s data
    input_size: int = 18 if abla else 36     # Input size (number of input signals) 
    dataset_id: int = 0                      # Dataset number
    dataset_path: str = "./data"             # Dataset directory
    mic: str = ''                            # Output sound pressure channel selection, 'near' = near-field only，'far' = far-field only，‘’ = all
    model_name: str = 'model_dual_lstm_abla.pth'                 # Name of saved .pth file (model)
    output_txt_name: str = 'output_dual_lstm_abla'               # Name of Loss history .txt file (loss profile)
    output_size: int = 5 if mic == 'near' or mic == 'far' else 10
    eps = torch.tensor(1e-4)                 # Prevent the denominator from approaching 0

    # Create a new model or load an existing model
    model = DualLSTMForecast(input_size=input_size, output_size=output_size, hidden_size=hidden_size,
                             forecast_length=forecast_length) if is_new else torch.load('model_dual_lstm.pth')

    # Load the training dataset (!!! Confirm whether the available memory is large enough to load the dataset !!!)
    dataset = MatDataset(train_mode=True, root_path=dataset_path, window_size=window_size,
                         forecast_length=forecast_length, memory=train_data_in_memory, dataset=dataset_id,
                         mic=mic, abla=abla)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    k = len(dataloader)

    # Set loss function & Optimizer
    criterion = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Number of epoch
    num_epochs = 90

    # Adopt cosine annealing lr_scheduler
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0001)

    # Check whether CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Start training
    print('Training started.')
    for epoch in range(num_epochs):
        t_start = time.time()   # Start timing
        model.train()
        running_loss = 0.0

        for inputs, targets in dataloader:
            inputs = inputs.transpose(1, 0)     # nn.LSTM: (T, B, C) 
            targets = targets.transpose(1, 0)   # nn.LSTM: (T, B, C) 

            # Clear gradient
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(inputs)

            # Loss calculation
            loss = criterion(outputs, targets)
            targets_square_mean = []
            loss_mean = []
            for i in range(output_size):
                targets_square_mean.append(torch.mean(targets[:, :, i] ** 2)+eps)
            for i in range(output_size):
                loss_mean.append(torch.mean(loss[:, :, i]))
            targets_square_mean = torch.stack(targets_square_mean)
            loss_mean = torch.stack(loss_mean)
            loss_final = loss_mean / targets_square_mean
            loss_final = torch.mean(loss_final)

            # Back propagation
            loss_final.backward()
            
            # Update model parameters
            optimizer.step()

            # Loss accumulation
            running_loss += loss_final.item()

        # Variable lr stepping
        scheduler.step()

        # Get training time consumption for each 1 epoch
        t_end = time.time()     # Stop timing
        train_time = t_end - t_start

        # Print average loss, lr and time consumption for 1 epoch of training
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/k:.7f}"
              f"  Learning Rate: {optimizer.param_groups[0]['lr']:.5f}"
              f"  Elapsed time(s): {train_time:.3f}")
        original_stdout = sys.stdout
        with open(output_txt_name+'.txt', 'a', encoding='utf-8') as file:
            sys.stdout = file
            print(f"Loss: {running_loss/k:.7f}")
            sys.stdout = original_stdout

        # Evaluate model performance for every 5 epochs of training
        if (epoch + 1) % 5 == 0:
            t_start = time.time()
            eval_loss = test_eval_model(model, window_size, forecast_length, output_size, mic, dataset_path,
                                        dataset_id, eval_data_in_memory, abla, eps)
            original_stdout = sys.stdout
            with open(output_txt_name+'_eval.txt', 'a', encoding='utf-8') as file:
                sys.stdout = file
                print(eval_loss)
                sys.stdout = original_stdout
            t_end = time.time()
            eval_time = t_end - t_start
            print(eval_loss + f"  Elapsed time(s): {eval_time:.3f}")

    torch.save(model, model_name)
    print('Training finished.')


if __name__ == '__main__':
    test_train_model()

