import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# Load custom CSS file
with open(".streamlit/custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Load datasets
df = pd.read_csv("C:/Users/LOQ/Documents/GUVI DS/Mini-Project/Multi Dieseas 04/Data Sets/indian_liver_patient.csv")
df1=pd.read_csv("C:/Users/LOQ/Documents/GUVI DS/Mini-Project/Multi Dieseas 04/Data Sets/kidney_disease.csv")
df2=pd.read_csv("C:/Users/LOQ/Documents/GUVI DS/Mini-Project/Multi Dieseas 04/Data Sets/parkinsons.csv")
  
#1.Load models and Scalers
with open("model_1.pkl", "rb") as f:
    parkinsons_model = pickle.load(f)
with open("scaler_1.pkl", "rb") as f:
    parkinsons_scaler = pickle.load(f)

#2.Load models and Scalers,encoder
with open("model_2.pkl", "rb") as f:
    kidney_model = pickle.load(f)
with open("scaler_2.pkl", "rb") as f:
    kidney_scaler = pickle.load(f)

#3.Load models and Scalers
with open("model_3.pkl", "rb") as f:
    liver_model = pickle.load(f)    
with open("scaler_3.pkl", "rb") as f:
    liver_scaler = pickle.load(f)



# Sidebar
with st.sidebar:
    st.markdown("<h1 class='sidebar-title'>⚕️ Prediction Menu</h1>", unsafe_allow_html=True)
    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)
    # Custom buttons
    if st.button("🏠 Home"):
        st.session_state["page"] = "Home"

    if st.button("🧠Parkinson Prediction"):
        st.session_state["page"] = "Parkinson"

    if st.button("🩸 Kidney Disease Prediction"):
        st.session_state["page"] = "Kidney"

    if st.button("❤️ Liver Disease Prediction"):
        st.session_state["page"] = "Liver"

    
    st.markdown("<hr class='sidebar-divider'>", unsafe_allow_html=True)

    st.markdown(
        "<p style='text-align:center; opacity:0.7;'>Made with ❤️ using Streamlit</p>",
        unsafe_allow_html=True
    )
# Default page
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# Render selected page
if st.session_state["page"] == "Home":
    st.markdown("<h1 class='sidebar-title'>⚕ Multiple Disease Prediction</h1>", unsafe_allow_html=True)
    
    st.set_page_config(layout="wide")
    st.write("### Objective")
    st.write("""
        **The objective of this web application is to provide users with a convenient platform to predict the likelihood of various diseases using machine learning models.
        By inputting relevant health parameters, users can receive predictions for diseases such as :rainbow[Kidney Disease, Liver Disease, and Parkinson's Disease].
        This tool aims to assist individuals in early detection and awareness of potential health issues, promoting proactive health management.**""")
    st.markdown("---")
    st.write("### Features")
    st.write("""
    - **:rainbow[User-Friendly Interface]:**  **Easy-to-navigate layout for seamless user experience.**
    - **:rainbow[Multiple Disease Predictions]:**  **Supports predictions for Kidney Disease, Liver Disease, and Parkinson's Disease.**
    - **:rainbow[Accurate Models]:**  **Utilizes pre-trained machine learning models for reliable predictions.**
    - **:rainbow[Real-Time Results]:**  **Instant feedback based on user inputs.**
    - **:rainbow[Educational Insights]:**  **Provides information about each disease and preventive measures.**
    """)
    st.markdown("---")
    st.write("### How To Use")
    st.write("""
    1. **:rainbow[Select Disease]:** **Use the sidebar to choose the disease you want to predict.**
    2. **:rainbow[Input Parameters]:** **Fill in the required health parameters in the provided fields.**
    3. **:rainbow[Get Prediction]:** **Click the 'Predict' button to receive your prediction results.**
    4. **:rainbow[Interpret Results]:** **Review the prediction outcome and any additional information provided.**
    """)
    st.markdown("---")
    st.write("### Disease Insights")

    tag1,tag2,tag3=st.columns(3)
    with tag1:
        st.markdown("#### :blue[Kidney Disease]")
        ploty_fig = px.histogram(df1, x='classification', title='Kidney Disease Distribution', color_discrete_sequence=['#636EFA'],
                                 labels={'classification': 'Kidney Disease Classification'})
        ploty_fig.update_xaxes(tickvals=[0, 1], ticktext=['No Kidney Disease', 'Kidney Disease'])
        st.plotly_chart(ploty_fig, use_container_width=True)
    with tag2:
        st.markdown("#### :green[Liver Disease]")
        ploty_fig = px.histogram(df, x='Dataset', title='Liver Disease Distribution', color_discrete_sequence=["#57C554"],
                                 labels={'Dataset': 'Liver Disease Classification'},nbins=3)
        ploty_fig.update_xaxes(tickvals=[1, 2], ticktext=['Liver Disease', 'No Liver Disease'])
        st.plotly_chart(ploty_fig, use_container_width=True)

    with tag3:
        st.markdown("#### :red[Parkinson's Disease]")
        ploty_fig = px.histogram(df2, x='status', title="Parkinson's Disease Distribution", color_discrete_sequence=["#C55454"],
                                 labels={'status': "Parkinson's Disease Classification"},nbins=3)
        ploty_fig.update_xaxes(tickvals=[0, 1], ticktext=['No Parkinsons', 'Parkinsons'])
        st.plotly_chart(ploty_fig, use_container_width=True)


elif st.session_state["page"] == "Parkinson":
    st.markdown("<h1 class='sidebar-title'>Parkinson Disease Prediction Page</h1>", unsafe_allow_html=True)
    
    st.markdown("""**A Parkinson Prediction Page is designed to help users understand and assess the likelihood of :rainbow[Parkinson’s disease using machine learning models and medical data.]
                It typically includes sections for inputting relevant health parameters, viewing prediction results, and accessing educational resources about Parkinson’s disease.
                To provide an easy-to-use interface where patients, doctors, or researchers can input relevant health parameters and receive a prediction about :red[Parkinson’s risk].**""")
    st.write("---")
    st.write("### :blue[Parkinson's Disease Prediction Form]")
    st.markdown("**Please fill in the following parameters to predict the likelihood of Parkinson's Disease:**")
    
    with st.form("parkinsons_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0, max_value=300.0)
            fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, max_value=300.0)
            flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0, max_value=300.0)
            Jitter_percent = st.number_input("Jitter (%)", min_value=0.0, max_value=1.0)
            Jitter_Abs = st.number_input("Jitter (Abs)", min_value=0.0, max_value=0.1)
            RAP = st.number_input("RAP", min_value=0.0, max_value=1.0)
            PPQ = st.number_input("PPQ", min_value=0.0, max_value=1.0)
            DDP = st.number_input("DDP", min_value=0.0, max_value=1.0)
        with col2:
            Shimmer = st.number_input("Shimmer", min_value=0.0, max_value=1.0)
            Shimmer_dB = st.number_input("Shimmer (dB)", min_value=0.0, max_value=5.0)
            APQ3 = st.number_input("APQ3", min_value=0.0, max_value=1.0)
            APQ5 = st.number_input("APQ5", min_value=0.0, max_value=1.0)
            APQ = st.number_input("APQ", min_value=0.0, max_value=1.0)
            DDA = st.number_input("DDA", min_value=0.0, max_value=1.0)
            NHR = st.number_input("NHR", min_value=0.0, max_value=1.0)
            HNR = st.number_input("HNR", min_value=0.0, max_value=50.0)
        with col3:
            RPDE = st.number_input("RPDE", min_value=0.0, max_value=1.0)
            DFA = st.number_input("DFA", min_value=0.0, max_value=2.0)
            spread1 = st.number_input("Spread1", min_value=-20.0, max_value=0.0)
            spread2 = st.number_input("Spread2", min_value=0.0, max_value=200.0)
            D2 = st.number_input("D2", min_value=0.0, max_value=5.0)
            PPE = st.number_input("PPE", min_value=0.0, max_value=1.0)

        prediction=st.form_submit_button("**:rainbow[Predict Parkinson's Disease]**")
        
    if prediction:

        input_data = [[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP,
                    Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR,
                    RPDE, DFA, spread1, spread2, D2, PPE]]
        # Scale the input data
        scaled_data = parkinsons_scaler.transform(input_data)
        # Make prediction
        prediction = parkinsons_model.predict(scaled_data)
        st.markdown("---")
        if prediction[0] == 1:
            st.markdown("#### **Results:**")
            st.markdown("### **The Model Predicts That The :red[Patient has Parkinson's Disease.]**")

            st.info("""
                    ### 🩺 Next Steps & Medical Advice  
                    **Please consider the following steps:**

                    - 📌 **Consult a Neurologist** for a complete diagnosis.  
                    - 🧪 **Confirm with clinical tests** such as DaTscan, MRI, or UPDRS evaluation.  
                    - 💊 **Early treatment** can slow down symptom progression.  
                    - 🏃‍♂️ **Regular exercise** (walking, cycling, stretching) improves balance & mobility.  
                    - 🥗 **Healthy diet** rich in antioxidants may help brain function.  
                    - 💬 **Discuss symptoms** like tremors, stiffness, voice changes, or movement difficulties.
                    - 👨‍⚕️ **Never rely only on AI predictions** — always follow professional medical guidance.
                    """)
            st.warning("⚠ Please consult a doctor for proper medical diagnosis and treatment. AI tools should not be used as a substitute for professional advice.")

        else:
            st.markdown("### **The Model Predicts That The :green[Patient Deos Not Have Parkinson's Disease.]**")

            st.success("""
                        ### 👍 Good News!
                        The model indicates **no signs of Parkinson’s Disease** based on the provided inputs.

                        ### 📝 Recommended Health Tips
                        Even though the results are positive, maintaining brain and body health is important:

                        - 🧠 Engage in **regular mental activities** (puzzles, reading, learning).
                        - 🚶‍♂️ Stay physically active (walking, yoga, light exercise).
                        - 🥗 Maintain a **balanced diet** rich in fruits & vegetables.
                        - 😴 Ensure proper **sleep routines**.
                        - 💧 Stay hydrated.
                        - 🔁 Go for periodic check-ups if you experience any new symptoms.
                        
                        > These suggestions support overall neurological well-being.
                        """)


elif st.session_state["page"] == "Kidney":
    st.markdown("<h1 class='sidebar-title'>Kidney Disease Prediction Page</h1>", unsafe_allow_html=True)

    st.markdown("""**The Kidney Disease Prediction Page is designed to help users assess their risk of :rainbow[Kidney Disease] using machine learning models and relevant health data.
                It typically includes sections for inputting health parameters, viewing prediction results, and accessing educational resources about kidney health.
                The goal is to provide an easy-to-use interface where patients, doctors, or researchers can input relevant health parameters and receive a prediction about :blue[Kidney Disease risk].**""")
    st.write("---")

    st.write("### :red[Kidney Disease Prediction Form]")
    st.markdown("**Please fill in the following parameters to predict the likelihood of Kidney Disease:**")
    with st.form("kidney_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120)
            bp = st.number_input("Blood Pressure", min_value=0, max_value=200)
            sg = st.number_input("Specific Gravity", min_value=1.005, max_value=1.030)
            al = st.number_input("Albumin", min_value=0, max_value=5)
            su = st.number_input("Sugar", min_value=0, max_value=5)
            rbc = st.selectbox("Red Blood Cells", options=["normal", "abnormal"])
            pc = st.selectbox("Pus Cell", options=["normal", "abnormal"])
        with col2:
            pcc = st.selectbox("Pus Cell Clumps", options=["present", "notpresent"])
            bgr = st.number_input("Blood Glucose Random", min_value=0, max_value=500)
            bu = st.number_input("Blood Urea", min_value=0, max_value=300)
            sc = st.number_input("Serum Creatinine", min_value=0.0, max_value=20.0)
            sod = st.number_input("Sodium", min_value=100, max_value=200)
            hemo = st.number_input("Hemoglobin", min_value=0.0, max_value=30.0)
            pcv = st.number_input("Packed Cell Volume", min_value=0, max_value=60)
        with col3:
            rc = st.number_input("Red Blood Cell Count", min_value=2.0, max_value=8.0)
            htn = st.selectbox("Hypertension", options=["yes", "no"])
            dm= st.selectbox("Diabetes Mellitus", options=["yes", "no"])
            cad= st.selectbox("Coronary Artery Disease", options=["yes", "no"])
            appet= st.selectbox("Appetite", options=["good", "poor"])
            pe= st.selectbox("Pedal Edema", options=["yes", "no"])
            ane= st.selectbox("Anemia", options=["yes", "no"])

        prediction = st.form_submit_button("**:rainbow[Predict Kidney Disease]**")
    if prediction:
        

        df1_1={'age':age, 
            'bp':bp,
            'sg':sg, 
            'al':al, 
            'su':su, 
            'rbc':rbc, 
            'pc':pc, 
            'pcc':pcc, 
            'bgr':bgr,
            'bu':bu, 
            'sc':sc, 
            'sod':sod, 
            'hemo':hemo, 
            'pcv':pcv, 
            'rc':rc, 
            'htn':htn, 
            'dm':dm,
            'cad':cad, 
            'appet':appet, 
            'pe':pe, 
            'ane':ane}

        # Encode categorical variables
        binary_map = {
                        "normal": 1,
                        "abnormal": 0,
                        "present": 1,
                        "notpresent": 0,
                        "yes": 1,
                        "no": 0,
                        "good": 0,
                        "poor": 1
                    }

        for i in ['rbc', 'pc', 'pcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']:
            df1_1[i] = binary_map[df1_1[i]]
        input=pd.DataFrame([df1_1])

        # Scale the input data
        scaled_df = kidney_scaler.transform(input)

        # Make prediction
        prediction = kidney_model.predict(scaled_df)
        st.markdown("---")
        
        if prediction[0] == 1:
            st.markdown("#### **Results:**")
            st.markdown("### **The Model Predicts That The :red[Patient has Kidney Disease.]**")
            st.info("""
                    ### 🩺 Next Steps & Medical Advice  
                    **Please consider the following steps:**

                    - 📌 **Consult a Nephrologist** for a detailed medical evaluation.  
                    - 🧪 **Perform kidney function tests**, such as eGFR, Serum Creatinine, BUN, Urine ACR, and Ultrasound.  
                    - 💧 **Stay hydrated**, unless a doctor has restricted fluids.  
                    - 🥗 **Follow a kidney-friendly diet** — reduce salt, processed food, red meat, and potassium-heavy foods if advised.  
                    - 💊 **Avoid self-medication**, especially painkillers (NSAIDs), which can worsen kidney function.  
                    - ❤️ **Maintain blood pressure & blood sugar**, as they are major causes of kidney issues.  
                    - 🚶 **Regular light exercise** helps overall health without stressing kidneys.  
                    - 👨‍⚕️ **Never rely only on AI predictions** — always consult a healthcare professional for accurate diagnosis.
                    """)

            st.warning("⚠ Please consult a doctor for proper medical diagnosis and treatment. AI tools should not be used as a substitute for professional advice.")

        else:
            st.markdown("#### **Results:**")
            st.markdown("### **The Model Predicts That The :green[Patient Does Not Have Kidney Disease.]**")
            st.info("""
            ### 🩺 Health Tips & Safety Advice  

            Even though the prediction is **negative**, maintaining healthy kidney function is important.

            #### ✔ Recommended Actions:
            - 💧 **Stay hydrated** — drink adequate water throughout the day.  
            - 🍎 **Maintain a balanced diet** low in salt and processed foods.  
            - 🩸 **Monitor blood pressure & sugar levels**, especially if you have diabetes or hypertension.  
            - 🧪 **Regular health checkups** (creatinine, urea, eGFR) to ensure stable kidney function.  
            - 🚶‍♂️ **Exercise regularly** to maintain overall health.  
            - 🚫 **Avoid excessive painkillers (NSAIDs)** as they can affect kidneys long-term.  
            - 💬 **Consult a doctor if you notice symptoms** like swelling, fatigue, changes in urine, or lower back pain.

            #### ⚠ Note:
            AI predictions are supportive tools — **always rely on a doctor for final diagnosis**.
            """)





elif st.session_state["page"] == "Liver":
    st.markdown("<h1 class='sidebar-title'>Liver Disease Prediction Page</h1>", unsafe_allow_html=True)

    st.markdown("""**The Liver Disease Prediction Page is designed to help users assess their risk of :rainbow[Liver Disease] using machine learning models and relevant health data.
                It typically includes sections for inputting health parameters, viewing prediction results, and accessing educational resources about liver health.     
                The goal is to provide an easy-to-use interface where patients, doctors, or researchers can input relevant health parameters and receive a prediction about :red[Liver Disease risk].**""")
    st.write("---") 
    st.write("### :green[Liver Disease Prediction Form]")
    st.markdown("**Please fill in the following parameters to predict the likelihood of Liver Disease:**")
    with st.form("liver_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=120)
            gender = st.selectbox("Gender", options=["Male", "Female"])
            tb = st.number_input("Total Bilirubin", min_value=0.0, max_value=50.0)
            db = st.number_input("Direct Bilirubin", min_value=0.0, max_value=30.0)
            alkphos = st.number_input("Alkaline Phosphotase", min_value=0, max_value=1000)
            
        with col2: 
            ala = st.number_input("Alamine Aminotransferase (SGPT)", min_value=0, max_value=1000)     
            asa = st.number_input("Aspartate Aminotransferase (SGOT)", min_value=0, max_value=1000)
            tp = st.number_input("Total Proteins", min_value=0.0, max_value=15.0)
            alb = st.number_input("Albumin", min_value=0.0, max_value=10.0)
            ag_ratio = st.number_input("Albumin and Globulin Ratio", min_value=0.0, max_value=5.0)
        prediction = st.form_submit_button("**:rainbow[Predict Liver Disease]**")

    if prediction:
        df2_2={'Age':age,   
            'Gender':gender, 
            'Total_Bilirubin':tb,    
            'Direct_Bilirubin':db,    
            'Alkaline_Phosphotase':alkphos,
            'Alamine_Aminotransferase':ala,    
            'Aspartate_Aminotransferase':asa,
            'Total_Protiens':tp,    
            'Albumin':alb,
            'Albumin_and_Globulin_Ratio':ag_ratio}
        # Encode categorical variables
        # gender encoding
        gender_map = {
                        "Male": 1,
                        "Female": 0}
        df2_2['Gender'] = gender_map[df2_2['Gender']]
        input=pd.DataFrame([df2_2]) 
        
        # Scale the input data
        scaled_df = liver_scaler.transform(input)
        
        # Make prediction
        prediction = liver_model.predict(scaled_df) 
        st.markdown("---")
        if prediction[0] == 1:
            st.markdown("#### **Results:**")
            st.markdown("### **The Model Predicts That The :red[Patient has Liver Disease.]**")
            st.warning("""
                            ### ⚠️ Medical Alert: Possible Liver Disease Detected

                            The model indicates **a high likelihood of liver disease** based on the provided inputs.  
                            Liver issues require timely medical attention to prevent complications.

                            #### 🚨 Recommended Next Steps
                            - 🩺 **Visit a hepatologist or gastroenterologist** for confirmation.
                            - 🧪 **Get diagnostic tests** such as LFT (Liver Function Test), Ultrasound, FibroScan, or MRI.
                            - 🚫 **Avoid alcohol**, as it can worsen inflammation or liver damage.
                            - 💊 **Do not take medication without medical advice** — some drugs strain the liver.
                            - 🥗 **Follow a liver-friendly diet**: low fat, no fried food, more fruits & vegetables.
                            - 🏃‍♂️ **Maintain healthy weight** to reduce fatty liver progression.
                            - 🔍 **Watch symptoms** like yellowing of eyes/skin, abdominal pain, fatigue, or nausea.

                            ### ⚠️ Important  
                            This is a machine-learning prediction.  
                            **A doctor’s consultation and lab tests are essential** for accurate diagnosis.
                            """)
   
        else:
            st.markdown("### **The Model Predicts That The :green[Patient Does Not Have Liver Disease.]**")
            st.success("""
            ### 🩺 Liver Health Status: Normal  
            Your inputs do **not show signs of Liver Disease.**  
            Maintaining healthy habits can keep your liver functioning well.

            #### 🟢 Recommended Liver Health Tips:
            - 🥗 **Eat fresh fruits & vegetables**; avoid junk and oily food.
            - 🚫 **Limit alcohol completely** for long-term liver safety.
            - 💧 **Stay hydrated** throughout the day.
            - 💊 **Use medications carefully** — avoid unnecessary drugs.
            - ⚖️ **Maintain healthy body weight** to prevent fatty liver.
            - 🏃‍♂️ **Regular exercise** supports liver metabolism.
            - 📅 **Annual health check-ups** if you have diabetes, obesity, or high cholesterol.

            > ✔ Even with a normal result, regular monitoring is important for long-term liver wellness.
            """)



    
   
