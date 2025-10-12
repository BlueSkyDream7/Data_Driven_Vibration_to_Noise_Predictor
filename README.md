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

<img width="852" height="260" alt="image" src="https://github.com/user-attachments/assets/316625f9-bc15-4100-8e7b-95bc17694efa" />
<img width="862" height="613" alt="image" src="https://github.com/user-attachments/assets/0e7ad118-5ae4-4087-97fd-ee46ee53a416" />
<img width="862" height="613" alt="image" src="https://github.com/user-attachments/assets/e2480ba1-55f9-450a-a6cf-50715b4be3a4" />
