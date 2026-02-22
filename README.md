# Integrated Medical Disease Detection

Integrated Medical Disease Detection is a multi-disease prediction system built using Machine Learning and Deep Learning. The application provides a unified platform where users can enter medical parameters or upload medical images to predict the risk or presence of multiple health conditions.

This project demonstrates an end-to-end healthcare AI system with multiple trained models deployed through a single Streamlit interface.

---

## Features

The system supports prediction for the following conditions:

- Heart Disease Prediction  
- Diabetes Prediction  
- Parkinson’s Disease Prediction  
- Maternal Health Risk Prediction  
- Fetal Health Classification  
- Brain Tumor Detection (Image-based CNN)

Users can select a disease from the sidebar, provide input values or upload an image, and receive instant predictions.

---

## Tech Stack

**Language & Framework**
- Python
- Streamlit

**Machine Learning / Deep Learning**
- Scikit-learn
- TensorFlow / Keras
- Pickle / Joblib model loading
- CNN for image classification

**Libraries**
- NumPy
- Pandas
- OpenCV
- Plotly
- Requests

---

## Project Structure
Integrated-Medical-Disease-Detection/
│
├── main.py
├── requirements.txt
│
├── Models/
│ ├── diabetes_model.sav
│ ├── parkinsons_model.sav
│ ├── finalized_maternal_model.sav
│ ├── fetal_health_classifier.sav
│ ├── heart_disease_model.h5
│ └── brain_tumor_model.h5
│
├── Datasets/
│ ├── diabetes.csv
│ ├── parkinsons.csv
│ ├── fetal_health.csv
│ ├── heart_disease_data.csv
│ └── Maternal Health Risk Data Set.csv


---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/Integrated-Medical-Disease-Detection.git
cd Integrated-Medical-Disease-Detection

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt

streamlit run main.py
