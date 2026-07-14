# 🔢 Handwritten Digit Recognition System using MediaPipe & XGBoost

A real-time AI-powered handwritten digit recognition system that allows users to draw digits (0–9) in the air using hand gestures. The application tracks the user's index finger with MediaPipe, converts the air drawing into a 28×28 grayscale image, and predicts the digit using a trained XGBoost machine learning model.

## 🚀 Live Demo

🔗 https://ai-based-digits-recognition-system-using-ml-mediapipegit-zftwi.streamlit.app/

---

## 📌 Features

- Real-time hand tracking using MediaPipe
- Air drawing with index finger gestures
- Gesture-based canvas clearing
- Automatic image preprocessing (28×28 grayscale)
- Handwritten digit prediction using XGBoost
- Prediction confidence score
- Interactive Streamlit web interface
- Responsive and user-friendly UI

---

## 🛠️ Tech Stack

- Python
- Streamlit
- OpenCV
- MediaPipe
- XGBoost
- Scikit-learn
- NumPy
- SciPy
- Joblib

---

## 📂 Project Structure

```
Digits Recognition System/
│── app.py
│── xgboost_digit_model.pkl
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/YourUsername/YourRepository.git
cd YourRepository
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 🎯 How It Works

1. Open the application.
2. Allow camera access.
3. Raise only your index finger to draw a digit in the air.
4. Hold an open palm to clear the drawing.
5. Click **Predict Air Drawing**.
6. The system preprocesses the drawing and predicts the handwritten digit.

---

## 📊 Machine Learning Pipeline

- Image Acquisition
- Hand Tracking using MediaPipe
- Air Drawing Capture
- Image Cropping
- Aspect Ratio Preservation
- Square Padding
- Resize to 28×28
- Normalization
- Feature Extraction
- XGBoost Prediction
- Confidence Score Generation

---

## 🚀 Future Improvements

- Support handwritten alphabet recognition.
- Upgrade to CNN-based deep learning models.
- Add voice feedback for predictions.
- Improve mobile responsiveness and cross-platform support.

---

## 👨‍💻 Developed By

**Shaik Khaja Mainuddin**

Artificial Intelligence & Data Science Graduate

GitHub:
https://github.com/Mainuddin123


---

## 📜 License

This project is developed for educational and portfolio purposes.
