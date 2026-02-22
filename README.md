Integrated Medical Disease Detection

A multi-disease prediction system built using Machine Learning and Deep Learning models. This Streamlit-based web application allows users to input medical parameters or upload images to predict the risk or presence of multiple diseases in a single platform.

The system integrates multiple trained models to provide real-time predictions for various medical conditions, helping demonstrate practical healthcare AI applications.

Features

The application supports prediction for the following diseases:

Heart Disease Prediction

Diabetes Prediction

Parkinson’s Disease Prediction

Pregnancy Risk Prediction

Fetal Health Classification

Brain Tumor Detection (Image-based)

Users can interact with an easy-to-use web interface and receive instant results based on model predictions.

The system uses multiple ML and DL models loaded from saved files and deployed through a unified Streamlit interface. 


Libraries

NumPy

Pandas

OpenCV

Joblib

Plotly (for visualization support)

Dependencies are listed in requirements.txt. 

requirements

Project Structure
Integrated-Medical-Disease-Detection/
│
├── main.py
├── requirements.txt
│
├── Models/
│   ├── diabetes_model.sav
│   ├── parkinsons_model.sav
│   ├── finalized_maternal_model.sav
│   ├── fetal_health_classifier.sav
│   ├── heart_disease_prediction_model.h5
│   └── brain_model.h5
│
├── Datasets/
│   ├── diabetes.csv
│   ├── parkinsons.csv
│   ├── fetal_health.csv
│   ├── heart_disease_data.csv
│   └── Maternal Health Risk Data Set.csv
Installation
1. Clone the repository
git clone https://github.com/<your-username>/integrated-medical-disease-detection.git
cd integrated-medical-disease-detection
2. Create a virtual environment

Windows:

python -m venv venv
venv\Scripts\activate

Mac/Linux:

python3 -m venv venv
source venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
Run the Application
streamlit run main.py

The app will open in your browser at:

http://localhost:8501
How It Works

Users select a disease from the sidebar.

Enter medical parameters or upload an image.

The corresponding pre-trained model is loaded.

The system returns a prediction instantly.

The application integrates multiple prediction pipelines into a single interface for ease of use and demonstration purposes.

Model Details
Disease	Model Type
Heart Disease	Deep Learning (.h5)
Diabetes	Machine Learning (Pickle)
Parkinson’s	Machine Learning (Pickle)
Pregnancy Risk	Machine Learning (Pickle)
Fetal Health	Machine Learning (Pickle)
Brain Tumor	CNN Image Classifier
Use Case

Educational and research demonstration

Healthcare AI portfolio project

Multi-model deployment example

End-to-end ML system using Streamlit
