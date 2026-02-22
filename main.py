!pip install -r requirements.txt
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import warnings
import pandas as pd
import plotly.express as px
from io import StringIO
import requests
from keras.models import load_model
import streamlit as st
from keras.preprocessing import image
import numpy as np
import cv2
from joblib import load

maternal_model = pickle.load(open("C:\\Users\\kaushik\\Desktop\\Brain\\Models\\finalized_maternal_model.sav",'rb'))
fetal_model = pickle.load(open("C:\\Users\\kaushik\\Desktop\\Brain\\Models\\fetal_health_classifier.sav",'rb'))
heart_model = load_model("C:\\Users\\kaushik\\Desktop\\Brain\\Models\\heart_disease_prediction_model.h5")
brain_model = load_model('C:\\Users\\kaushik\\Desktop\\Brain\\model_VGG.h5')
diabetes_model = pickle.load(open('C:\\Users\\kaushik\\Desktop\\Brain\\Models\\diabetes_model.sav', 'rb'))
parkinsons_model = pickle.load(open('C:\\Users\\kaushik\\Desktop\\Brain\\Models\\parkinsons_model.sav', 'rb'))
# sidebar for navigation
with st.sidebar:
    st.title("Integrated Medical Disease Detection")
    

    selected = option_menu('Integrated Medical Disease Detection',
                          
                          ['Heart Disease Prediction',
                          'Diabetes Prediction',
                          'Parkinsons Disease Prediction',
                          'Pregnancy Risk Prediction',
                          'Fetal Health Prediction',
                          'Brain Tumour Detection'],
                          icons=['chat-square-text','hospital','capsule-pill','clipboard-data'],
                          default_index=0)


if (selected == 'Pregnancy Risk Prediction'):
    
    # page title
    st.title('Pregnancy Risk Prediction')
    content = "Predicting the risk in pregnancy involves analyzing several parameters, including age, blood sugar levels, blood pressure, and other relevant factors. By evaluating these parameters, we can assess potential risks and make informed predictions regarding the pregnancy's health"
    st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div></br>", unsafe_allow_html=True)
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input('Age of the Person', key = "age",min_value=10, max_value=70, step=1)
        
    with col2:
        SystolicBP = st.number_input('SystolicBP in mmHg',min_value=70, max_value=160, step=1)
    
    with col3:
        diastolicBP = st.number_input('DiastolicBP in mmHg',min_value=50, max_value=100, step=1)
        
    
    with col1:
        BS = st.number_input('Blood glucose in mmol/L',min_value=6.0, max_value=20.0, step=0.1)
        
    with col2:
        bodyTemp = st.number_input('Body Temperature in Celsius',min_value=98.0, max_value=103.0, step=0.1)
        
    with col3:
        heartRate = st.number_input('Heart rate in beats per minute',min_value=7, max_value=90, step=1)
    
    riskLevel=""
    #predicted_risk = [0] 
    # creating a button for Prediction
    with col1:
        if st.button('Predict Pregnancy Risk'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                predicted_risk = maternal_model.predict([[age, diastolicBP, BS, bodyTemp, heartRate]])
                print(predicted_risk)
            # st
            st.subheader("Risk Level:")
            if predicted_risk[0] == 0:
                st.markdown('<bold><p style="font-weight: bold; font-size: 20px; color: green;">Low Risk</p></bold>', unsafe_allow_html=True)
            elif predicted_risk[0] == 1:
                st.markdown('<bold><p style="font-weight: bold; font-size: 20px; color: orange;">Medium Risk</p></Bold>', unsafe_allow_html=True)
            else:
                st.markdown('<bold><p style="font-weight: bold; font-size: 20px; color: red;">High Risk</p><bold>', unsafe_allow_html=True)
    with col2:
        if st.button("Clear"): 
            st.rerun()

if (selected == 'Fetal Health Prediction'):
    
    # page title
    st.title('Fetal Health Prediction')
    
    content = "Cardiotocograms (CTGs) are a simple and cost accessible option to assess fetal health, allowing healthcare professionals to take action in order to prevent child and maternal mortality"
    st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div></br>", unsafe_allow_html=True)
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        BaselineValue = st.number_input('Baseline Value',min_value=105.0, max_value=160.0, step=0.1)
        
    with col2:
        Accelerations = st.text_input('Accelerations')
    
    with col3:
        fetal_movement = st.text_input('Fetal Movement')
    
    with col1:
        uterine_contractions = st.text_input('Uterine Contractions')

    with col2:
        light_decelerations = st.text_input('Light Decelerations')
    
    with col3:
        severe_decelerations = st.text_input('Severe Decelerations')

    with col1:
        prolongued_decelerations = st.text_input('Prolongued Decelerations')
        
    with col2:
        abnormal_short_term_variability = st.number_input('Abnormal Short Term Variability',min_value=12.0, max_value=87.0, step=0.1)
    
    with col3:
        mean_value_of_short_term_variability = st.number_input('Mean Value Of Short Term Variability',min_value=0.0, max_value=7.0, step=0.1)
    
    with col1:
        percentage_of_time_with_abnormal_long_term_variability = st.number_input('Percentage Of Time With ALTV',min_value=0.0, max_value=90.0, step=0.1)

    with col2:
        mean_value_of_long_term_variability = st.number_input('Mean Value Long Term Variability',min_value=0.0, max_value=50.0, step=0.1)
    
    with col3:
        histogram_width = st.number_input('Histogram Width',min_value=3.0, max_value=180.0, step=0.1)

    with col1:
        histogram_min = st.number_input('Histogram Min',min_value=50.0, max_value=160.0, step=0.1)
        
    with col2:
        histogram_max = st.number_input('Histogram Max',min_value=122.0, max_value=238.0, step=0.1)
    
    with col3:
        histogram_number_of_peaks = st.number_input('Histogram Number Of Peaks',min_value=0.0, max_value=18.0, step=0.1)
    
    with col1:
        histogram_number_of_zeroes = st.number_input('Histogram Number Of Zeroes',min_value=0.0, max_value=10.0, step=0.1)

    with col2:
        histogram_mode = st.number_input('Histogram Mode',min_value=60.0, max_value=187.0, step=0.1)
    
    with col3:
        histogram_mean = st.number_input('Histogram Mean',min_value=73.0, max_value=182.0, step=0.1)
    
    with col1:
        histogram_median = st.number_input('Histogram Median',min_value=77.0, max_value=186.0, step=0.1)

    with col2:
        histogram_variance = st.number_input('Histogram Variance',min_value=0.0, max_value=269.0, step=0.1)
    
    with col3:
        histogram_tendency = st.number_input('Histogram Tendency',min_value=-1.0, max_value=1.0, step=0.1)
    
    # creating a button for Prediction
    st.markdown('</br>', unsafe_allow_html=True)
    with col1:
        if st.button('Predict Fetal Health'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                predicted_risk = fetal_model.predict([[BaselineValue, Accelerations, fetal_movement,
       uterine_contractions, light_decelerations, severe_decelerations,
       prolongued_decelerations, abnormal_short_term_variability,
       mean_value_of_short_term_variability,
       percentage_of_time_with_abnormal_long_term_variability,
       mean_value_of_long_term_variability, histogram_width,
       histogram_min, histogram_max, histogram_number_of_peaks,
       histogram_number_of_zeroes, histogram_mode, histogram_mean,
       histogram_median, histogram_variance, histogram_tendency]])
            # st.subheader("Risk Level:")
            st.markdown('</br>', unsafe_allow_html=True)
            if predicted_risk[0] == 0:
                st.markdown('<bold><p style="font-weight: bold; font-size: 20px; color: green;">Result  Comes to be  Normal</p></bold>', unsafe_allow_html=True)
            elif predicted_risk[0] == 1:
                st.markdown('<bold><p style="font-weight: bold; font-size: 20px; color: orange;">Result  Comes to be  Suspect</p></Bold>', unsafe_allow_html=True)
            else:
                st.markdown('<bold><p style="font-weight: bold; font-size: 20px; color: red;">Result  Comes to be  Pathological</p><bold>', unsafe_allow_html=True)
    with col2:
        if st.button("Clear"): 
            st.rerun()

if (selected == 'Heart Disease Prediction'):

    st.title("Heart Disease Prediction")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Age',min_value=1, max_value=100, step=1)

    with col2:
        sex = st.number_input('Sex (0 for Male, 1 for Female)', min_value=0, max_value=1, step=1)

    with col3:
        cp = st.number_input('Chest Pain types: 0 = TA, 1 = ATA, 2 = NP, 3 = AS ', min_value=0, max_value=3, step=1)
        
        
    with col1:
        trestbps = st.number_input('Resting Blood Pressure', min_value=100, max_value=170, step=1)
        
       
    with col2:
        chol = st.number_input('Serum Cholestoral in mg/dl', min_value=130, max_value=450, step=1)
        
         
    with col3:
        fbs = st.number_input('Fbs > 120 mg/dl (0 for False, 1 for True)', min_value=0, max_value=1, step=1)
        
        
    with col1:
        restecg = st.number_input('Resting Electrocardiographic results', min_value=0, max_value=2, step=1)
        
        
    with col2:
        thalach = st.number_input('Maximum Heart Rate achieved', min_value=80, max_value=200, step=1)
        
        
    with col3:
        exang = st.number_input('Exang (0 for No, 1 for Yes)', min_value=0, max_value=1, step=1)
        
    with col1:
        oldpeak = st.number_input('ST depression induced by exercise',min_value=0.0,max_value=5.0,step=0.1)
        

    with col2:
        slope = st.number_input('Slope of the peak exercise ST segment', min_value=0, max_value=2, step=1)
   
    with col3:
        ca = st.number_input('Major vessels colored by flourosopy', min_value=0, max_value=4, step=1)
        

    with col1:
        thal = st.number_input('thal: 1 = normal; 2 = fixed defect; 3 = reversable defect', min_value=0, max_value=3, step=1)
        
    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        user_input = [float(x) for x in user_input]
        print("Input shape:", len(user_input))
        print("Input data:", user_input)
    
        user_input = np.array(user_input, dtype=np.float32)  # Convert to NumPy array with float32 dtype
        print("Input shape (after conversion to NumPy array):", user_input.shape)
    
        heart_prediction = heart_model.predict(user_input.reshape(1, -1))
        print("Prediction:", heart_prediction)
    
        if heart_prediction[0] == 1:
           heart_diagnosis = 'The person is having heart disease'
        else:
           heart_diagnosis = 'The person does not have any heart disease'

        st.success(heart_diagnosis)


if (selected == 'Diabetes Prediction'):

    st.title("Diabetes Prediction")

     # getting the input data from the user
    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input('Number of Pregnancies',min_value=0, max_value=20, step=1)

    with col2:
        Glucose = st.number_input('Glucose Level',min_value=0, max_value=200, step=1)

    with col3:
        BloodPressure = st.number_input('Blood Pressure value',min_value=0, max_value=120, step=1)

    with col1:
        SkinThickness = st.number_input('Skin Thickness value',min_value=0, max_value=100, step=1)

    with col2:
        Insulin = st.number_input('Insulin Level',min_value=0, max_value=800, step=1)

    with col3:
        BMI = st.number_input('BMI value',min_value=0.0, max_value=67.0, step=0.1)

    with col1:
        DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value',min_value=0.8, max_value=2.5, step=0.1)

    with col2:
        Age = st.number_input('Age of the Person',min_value=20, max_value=65, step=1)

    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):

        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]

        user_input = [float(x) for x in user_input]

        diab_prediction = diabetes_model.predict([user_input])

        if diab_prediction[0] == 1:
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)


if (selected == 'Parkinsons Disease Prediction'):

    st.title("Parkinsons Disease Prediction")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.number_input('MDVP Fo(Hz)',min_value=90.00, max_value=260.00, step=0.01)

    with col2:
        fhi = st.number_input('MDVP Fhi(Hz)',min_value=100.00, max_value=590.00, step=0.01)

    with col3:
        flo = st.number_input('MDVP Flo(Hz)',min_value=70.00, max_value=230.00, step=0.01)

    with col4:
        Jitter_percent = st.number_input('MDVP Jitter(%)',min_value=0.01, max_value=0.03, step=0.001)

    with col5:
        Jitter_Abs = st.text_input('MDVP Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP RAP')

    with col2:
        PPQ = st.text_input('MDVP PPQ')

    with col3:
        DDP = st.text_input('Jitter DDP')

    with col4:
        Shimmer = st.text_input('MDVP Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer APQ5')

    with col3:
        APQ = st.text_input('MDVP APQ')

    with col4:
        DDA = st.text_input('Shimmer DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('Spread1')

    with col5:
        spread2 = st.text_input('Spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction    
    if st.button("Parkinson's Test Result"):

        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]

        user_input = [float(x) for x in user_input]

        parkinsons_prediction = parkinsons_model.predict([user_input])

        if parkinsons_prediction[0] == 1:
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)

        

if (selected == 'Brain Tumour Detection'):
    st.title('Brain Tumour Detection')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
    # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
        img = image.load_img(uploaded_file, target_size=(128, 128))  # Resize to match model input shape
        img_array = image.img_to_array(img)
        img_array = img_array.astype('float32') / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
        prediction = brain_model.predict(img_array)
        print(prediction)

    # Check if prediction probability is greater than 0.5
        if prediction > 0.5:
            predicted_class = 1
            st.write('The person has brain tumour')
        else:
            predicted_class = 0
            st.write('The person does not have brain tumour')



