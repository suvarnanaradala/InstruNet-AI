# 🎼 Instrument Recognition System (CNN)

A Deep Learning based web application that classifies audio files into 11 musical instruments using a Convolutional Neural Network (CNN).

## 🎯 Supported Instruments

- Cello  
- Clarinet  
- Flute  
- Acoustic Guitar  
- Electric Guitar  
- Organ  
- Piano  
- Saxophone  
- Trumpet  
- Violin  
- Voice  

---

## 🧠 Model Details

- Model File: `instrument_cnn_final.h5`
- Input Shape: (130, 40, 1)
- Feature Extraction: MFCC (40 coefficients)
- Framework: TensorFlow / Keras
- Total Classes: 11
- Output Layer: Softmax

---

## 📁 Project Structure


InstruNet-AI/
│
├── Infosys ppt.pdf
├── LICENSE
├── Model_evaluation.py
├── README.md
├── app.py
├── audio_to_spectrogram.py
├── metadata.csv
├── requirements.txt
├── resize_spectrograms.py
├── train_final.py






---

## ⚙️ Installation

Install required libraries:

```bash
pip install streamlit tensorflow librosa numpy pandas plotly



▶️ Run the Application
streamlit run app.py



Open in browser:

http://localhost:8501



🔍 How It Works

Upload audio file (.wav / .mp3)

Audio loaded using Librosa (22050 Hz)

MFCC features extracted (40 coefficients)

Resized to (130, 40)

CNN model predicts probability distribution

Highest probability displayed as final instrument


⚠️ Important

Model file must be inside models/ folder

Class order must match training order

Input preprocessing must match training preprocessing



🛠 Technologies Used

Python

TensorFlow / Keras

Librosa

NumPy

Pandas

Plotly

Streamlit


License
This project is licensed under the MIT License.

