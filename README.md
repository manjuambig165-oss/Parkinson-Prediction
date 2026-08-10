# 🧠 Parkinson AI - Voice Measurements and Prediction System

An AI-powered web application for Parkinson's disease prediction using biomedical voice measurements.

## 📌 Project Overview

Parkinson AI is a Flask-based machine learning web application that analyzes 22 biomedical voice measurements and provides an AI-assisted Parkinson's disease risk prediction.

The application provides a simple dashboard where users can enter patient information, submit voice measurements, view prediction results, and maintain prediction history.

## ✨ Features

- 🔐 User registration and login
- 🧑‍⚕️ Patient information management
- 🎙️ Analysis of 22 biomedical voice features
- 🤖 Machine learning based prediction
- 📊 Prediction confidence and risk level
- 📈 Dashboard with prediction statistics
- 🕒 Prediction history
- 🗑️ Delete individual prediction records
- 🧹 Clear prediction history
- 📱 Responsive web interface
- ⚠️ Medical decision-support disclaimer

## 🧪 Voice Measurements

The system uses the following 22 voice-related features:

1. MDVP:Fo(Hz)
2. MDVP:Fhi(Hz)
3. MDVP:Flo(Hz)
4. MDVP:Jitter(%)
5. MDVP:Jitter(Abs)
6. MDVP:RAP
7. MDVP:PPQ
8. Jitter:DDP
9. MDVP:Shimmer
10. MDVP:Shimmer(dB)
11. Shimmer:APQ3
12. Shimmer:APQ5
13. MDVP:APQ
14. Shimmer:DDA
15. NHR
16. HNR
17. RPDE
18. DFA
19. spread1
20. spread2
21. D2
22. PPE

## 🛠️ Technologies Used

- Python
- Flask
- Pandas
- Scikit-learn
- Joblib
- SQLite
- HTML5
- CSS3
- JavaScript
- Font Awesome
- Git & GitHub

## 📂 Project Structure

```text
Parkinson-Prediction/
│
├── app.py
├── train_model.py
├── model.pkl
├── scaler.pkl
├── imputer.pkl
├── parkinsons.csv
├── requirements.txt
├── .gitignore
├── README.md
│
├── static/
│   └── css/
│       └── style.css
│
└── templates/
    ├── about.html
    ├── dashboard.html
    ├── history.html
    ├── index.html
    ├── login.html
    ├── patient.html
    ├── register.html
    └── result.html
