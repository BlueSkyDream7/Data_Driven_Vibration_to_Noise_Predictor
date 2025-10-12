# Data_Driven_Vibration_to_Noise_Predictor
Vibration-Noise Predictor (VNP) is a data-driven modelling method for Electric Drive System (EDS) noise prediction. The model takes the vibration acceleration on the EDS housing as input and the noise sound pressure signal as output, which was trained by the weighted MSE as the loss function.
The results show that the prediction models based on LSTM and GRU perform better, among which the prediction network model based on LSTM has slightly better performance.

- ./data: The datasets for model training are supposed to be put here.

- ./dataset: The program for loading .mat(v7.3) dataset files, which should contains 2 variables: "X" and "Y".

- ./eval: The program for evaluating model performance during the training process.

- ./models: The folder which contains different kinds of VNP models, including neural network models based on RNN/LSTM/GRU, the structure of the models are configured here.

- ./train: The **main** training program of the project, hyperparameters such as lr, hidden size, batch size and window size for model training are set here. In addition, models and loss histories are saved here after the training process completed.

Install requirements in Terminal:
<code>
pip install -r requirements.txt
</code>
