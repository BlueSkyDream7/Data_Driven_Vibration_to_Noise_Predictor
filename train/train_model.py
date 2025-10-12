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
    is_new: bool = True                      # 是否创建新网络模型，不创建则加载现有模型
    abla: bool = True                        # 输入特征是否由12测点三向加速度消融至6测点三向加速度
    train_data_in_memory: bool = True        # 是否将训练数据加载至内存，内存足够前提下可以加速训练
    eval_data_in_memory: bool = False        # 是否将测试数据加载至内存
    window_size: int = 2560                  # 人耳感知频率20Hz~20480Hz，所以选最低20Hz，对应序列长度N
    forecast_length: int = 2560              # 预测序列长度L
    hidden_size: int = 128                   # LSTM 隐藏层神经元数
    batch_size: int = 32                     # 1个batch为1.6s的数据
    input_size: int = 18 if abla else 36     # 输入Features维度数
    dataset_id: int = 0                      # Dataset编号 0:All 1:Norm 2:Std 3:SteadyOnly 4:TransientOnly
    dataset_path: str = "F:\\VN_DL_Dataset"  # Dataset目录
    mic: str = ''                         # 预测麦克风点位， 'near':仅用近场，'far':仅用远场，‘’:所有麦克风
    model_name: str = 'model_dual_lstm_abla.pth'                 # 保存的网络模型.pth文件名
    output_txt_name: str = 'output_dual_lstm_abla'               # 损失曲线记录.txt文件名
    output_size: int = 5 if mic == 'near' or mic == 'far' else 10
    eps = torch.tensor(1e-4)                 # 防止权重系数分母接近于0

    # 新建或加载现有模型
    model = DualLSTMForecast(input_size=input_size, output_size=output_size, hidden_size=hidden_size,
                             forecast_length=forecast_length) if is_new else torch.load('model_dual_lstm.pth')

    # 加载训练数据集（可用内存至少为80GB时才可使用Memory=True加速训练过程中的数据读取，否则可能因内存不足导致死机）
    dataset = MatDataset(train_mode=True, root_path=dataset_path, window_size=window_size,
                         forecast_length=forecast_length, memory=train_data_in_memory, dataset=dataset_id,
                         mic=mic, abla=abla)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    k = len(dataloader)

    # 定义损失函数和优化器
    criterion = nn.MSELoss(reduction='none')  # MSE作为损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

    # 设置训练的轮数
    num_epochs = 90

    # 使用余弦式下降学习率调度器
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.0001)

    # 检测GPU是否可用，可用则将model转移
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 开始训练
    print('Training started.')
    for epoch in range(num_epochs):
        t_start = time.time()   # 训练开始计时
        model.train()  # 将模型设置为训练模式
        running_loss = 0.0

        for inputs, targets in dataloader:
            inputs = inputs.transpose(1, 0)     # 调整为 nn.LSTM 默认输入顺序(T, B, C) batch_first 默认为 false
            targets = targets.transpose(1, 0)   # 调整为 nn.LSTM 默认输出顺序(T, B, C) batch_first 默认为 false

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(inputs)

            # 计算损失
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

            # 反向传播
            loss_final.backward()
            
            # 参数更新
            optimizer.step()

            # 累计损失
            running_loss += loss_final.item()

        # 变学习率步进
        scheduler.step()

        # 训练时间
        t_end = time.time()     # 训练结束计时
        train_time = t_end - t_start

        # 打印每轮的平均损失、学习率与耗时
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/k:.7f}"
              f"  Learning Rate: {optimizer.param_groups[0]['lr']:.5f}"
              f"  Elapsed time(s): {train_time:.3f}")
        original_stdout = sys.stdout
        with open(output_txt_name+'.txt', 'a', encoding='utf-8') as file:
            sys.stdout = file
            print(f"Loss: {running_loss/k:.7f}")
            sys.stdout = original_stdout

        # 每5个Epoch进行一次测试
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
