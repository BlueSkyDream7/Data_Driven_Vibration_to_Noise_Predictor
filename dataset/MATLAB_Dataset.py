import torch.utils.data.dataset as dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import os
import h5py
import time
import math


class MatDataset(dataset.Dataset):
    def __init__(self, train_mode, root_path, window_size, forecast_length, memory, dataset, mic, abla):
        self.train_mode = train_mode    # Read dataset for Training：train_mode = 1    Read dataset for Training：train_mode = 0
        self.window_size = window_size
        self.forecast_length = forecast_length
        files_path = root_path + '/Train' if train_mode else root_path + '/Test'    # The files_path directory must contain both a 'Train' folder and a 'Test' folder, 
                                                                                    # which will serve as the training dataset and test dataset respectively
        files_list = os.listdir(files_path)
        self.data_path = os.path.join(files_path, files_list[dataset])
        self.memory = memory
        self.abla = abla
        if mic == 'near':
            self.y_indices = [0, 2, 5, 7, 9]
        elif mic == 'far':
            self.y_indices = [1, 3, 4, 6, 8]
        else:
            self.y_indices = np.arange(0, 10)
        if abla:
            self.x_indices = np.concatenate((np.arange(12, 18), np.arange(24, 30), np.arange(33, 39)), axis=0)
        else:
            self.x_indices = np.concatenate((np.arange(6, 18), np.arange(21, 24), np.arange(27, 48)), axis=0)
        if self.memory:
            hdf = h5py.File(self.data_path)
            with hdf as file:
                # self.x0 = file['X'][:, self.x_indices]
                self.x0 = file['X'][:, :]
                self.y0 = file['Y'][:, :]

    def __getitem__(self, idx):
        if self.memory:
            if self.abla:
                x0 = np.concatenate((self.x0[idx * self.window_size:(idx + 1) * self.window_size, 12:18],
                                     self.x0[idx * self.window_size:(idx + 1) * self.window_size, 24:30],
                                     self.x0[idx * self.window_size:(idx + 1) * self.window_size, 33:39]), axis=1)
            else:
                x0 = np.concatenate((self.x0[idx * self.window_size:(idx + 1) * self.window_size, 6:18],
                                     self.x0[idx * self.window_size:(idx + 1) * self.window_size, 21:24],
                                     self.x0[idx * self.window_size:(idx + 1) * self.window_size, 27:48]), axis=1)
            y0 = self.y0[idx * self.window_size:(idx + 1) * self.window_size, self.y_indices]
        else:
            hdf = h5py.File(self.data_path)
            with hdf as file:
                # x0 = file['X'][idx * self.window_size:(idx + 1) * self.window_size, self.x_indices]
                if self.abla:
                    x0 = np.concatenate((file['X'][idx * self.window_size:(idx + 1) * self.window_size, 12:18],
                                         file['X'][idx * self.window_size:(idx + 1) * self.window_size, 24:30],
                                         file['X'][idx * self.window_size:(idx + 1) * self.window_size, 33:39]), axis=1)
                else:
                    x0 = np.concatenate((file['X'][idx * self.window_size:(idx + 1) * self.window_size, 6:18],
                                         file['X'][idx * self.window_size:(idx + 1) * self.window_size, 21:24],
                                         file['X'][idx * self.window_size:(idx + 1) * self.window_size, 27:48]), axis=1)
                y0 = file['Y'][idx * self.window_size:(idx + 1) * self.window_size, self.y_indices]
        x = torch.tensor(x0)
        y = torch.tensor(y0)
        if torch.cuda.is_available():
            x = x.to('cuda')
            y = y.to('cuda')
        return x, y

    def __len__(self):
        if self.memory:
            length = math.floor((self.y0.shape[0] - self.window_size + self.forecast_length) / self.window_size)
        else:
            hdf = h5py.File(self.data_path)
            with hdf as file:
                length = math.floor((file['Y'].shape[0]-self.window_size+self.forecast_length) / self.window_size)
        return length


# Debug
if __name__ == '__main__':
    ROOT_PATH = "../data"
    start_time = time.time()
    dataset = MatDataset(train_mode=False, root_path=ROOT_PATH, window_size=2560, forecast_length=2560, memory=True,
                         dataset=0, mic='far', abla=False)
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for data in train_dataloader:
        print(data)
        end_time = time.time()
        print(f'load time {end_time - start_time :.4f}')
        k = len(dataset)
        print('')

