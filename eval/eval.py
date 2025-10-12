import torch
from dataset.MATLAB_Dataset import MatDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import pdb
import time


def r2_score(y_pred, y_true):
    ss_res = torch.sum(torch.pow(y_true - y_pred, 2))
    ss_tot = torch.sum(torch.pow(y_true - torch.mean(y_true), 2))
    return 1 - (ss_res / ss_tot)


def test_eval_model(model, window_size, forecast_length, output_size, mic, data_path, dataset_id, memory, abla, eps):
    model.eval()
    batch_size = 32
    dataset = MatDataset(train_mode=False, root_path=data_path, window_size=window_size,
                         forecast_length=forecast_length, memory=memory, dataset=dataset_id, mic=mic, abla=abla)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    k = len(dataloader)
    # device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    running_loss = 0.0
    # mae_loss_sum = 0.0
    # rmse_loss_sum = 0.0
    # r2_score_sum = 0.0
    criterion = nn.MSELoss(reduction='none')
    # mae_criterion = nn.L1Loss()
    # mse_criterion = nn.MSELoss()
    # print("Evaluation started.")

    # t_start = time.time()
    for inputs, targets in dataloader:
        inputs = inputs.transpose(1, 0)     # nn.LSTM default output: (seq, batch, feature)
        targets = targets.transpose(1, 0)   # nn.LSTM default output: (seq, batch, feature)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # mae_loss = mae_criterion(outputs, targets)
        # rmse_loss = torch.sqrt(mse_criterion(outputs, targets))
        # r2_loss = r2_score(outputs, targets)
        targets_square_mean = []
        loss_mean = []
        for i in range(output_size):
            targets_square_mean.append(torch.mean(targets[:, :, i] ** 2) + eps)
        for i in range(output_size):
            loss_mean.append(torch.mean(loss[:, :, i]))
        targets_square_mean = torch.stack(targets_square_mean)
        loss_mean = torch.stack(loss_mean)
        loss_final = loss_mean / targets_square_mean
        loss_final = torch.mean(loss_final)
        running_loss += loss_final.item()
        # mae_loss_sum += mae_loss.item()
        # rmse_loss_sum += rmse_loss.item()
        # r2_score_sum += r2_loss.item()
        # if r2_loss.item() < -1:
        #     pdb.set_trace()
    # t_end = time.time()
    # eval_time = t_end - t_start
    # print(eval_time)
    eval_loss = f"Eval Loss: {running_loss / k:.7f}"
    # mae_loss_avg = f"MAE Loss: {mae_loss_sum / k:.7f}"
    # rmse_loss_avg = f"RMSE Loss: {rmse_loss_sum / k:.7f}"
    # r2_score_avg = f"R2: {r2_score_sum / k:.7f}"
    return eval_loss


# Debug
if __name__ == '__main__':
    mdl = torch.load("../train/model_rnn_far_transient.pth")
    dataset_path = "./data"
    ws = 2560  # The human ear can perceive frequencies from 20Hz to 20,480Hz, therefore a lower limit of 20Hz was selected.
    fl = 2560  # seq2seq
    os = 10
    micp = 'far'
    dsid = 0
    loss, mae, rmse, r2 = test_eval_model(mdl, ws, fl, os, micp, dataset_path, dsid, memory=True, abla=False, eps=1e-4)
    print(loss)
    print(mae)
    print(rmse)
    print(r2)


