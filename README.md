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

<img width="868" height="616" alt="image" src="https://github.com/user-attachments/assets/b8534d3d-42e1-467b-bf44-a1b957f9ae50" />

<img width="868" height="618" alt="image" src="https://github.com/user-attachments/assets/45f7f846-2793-4a7b-9d66-9b172c1ed6a5" />

<img width="1205" height="499" alt="image" src="https://github.com/user-attachments/assets/d1bcc655-7b43-48c4-996c-a46e558b2879" />
