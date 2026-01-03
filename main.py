import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sksurv.datasets import get_x_y
from sksurv.ensemble import RandomSurvivalForest


st.title("Disease-free survival prediction after liver resection for hepatocellular carcinoma")

st.info(
    "⚠️ This tool provides **relative risk stratification** based on population-level models. "
    "Predictions should **not** be interpreted as absolute decision thresholds for individual patient management. "
    "Clinical decisions must be made within a multidisciplinary context."
)

st.write("### Input Data")
col1, col2 = st.columns(2)

Largest_nodule_diameter = col1.number_input("Size of the largest nodule in mm", min_value=1, value= 10, placeholder="Size in millimetes")
Satellite_nodules = col1.number_input("Satellite nodules", min_value=0, max_value=1, value=0, placeholder= "0/1")
VETC_subtype = col1.number_input("VETC subtype", min_value=0, max_value=1, value=0, placeholder= "0/1")
Surgical_margins = col2.number_input("Surgical margins", min_value=0, max_value=1, value=0, placeholder= "0/1")
Microvascular_invasion = col2.number_input("Microvascular invasion", min_value=0, max_value=1, value=0, placeholder= "0/1")
BCLC_before_intervention = col1.number_input("BCLC before surgery (0, A=1, B=2, C=3)", min_value=0,max_value=3, value=0, placeholder= "0-3")
Edmondson_Steiner_Grade = col1.number_input("Edmondson steiner grade (0=I, II=1, III=2, IV=3, V=4)", min_value=0, max_value=4, value=0, placeholder= "0-4")
Number_of_tumors_on_the_specimen = col2.number_input("Number of tumors on the specimen", min_value=1, max_value=50, value=1, placeholder= "1-50")
Preop_AFP = col2.number_input("Preoperative AFP level (ng/ml)", min_value=0.0, value=1.0, placeholder= "AFP level (ng/ml)")
Cirrhosis = col2.number_input("Cirrhosis", min_value=0, max_value=1, value=0, placeholder= "0/1")

df = pd.DataFrame(
    {
        "Satellite_nodules": [Satellite_nodules],
        "Largest_nodule_diameter": [Largest_nodule_diameter],
        "VETC_subtype": [VETC_subtype],
        "Microvascular_invasion": [Microvascular_invasion],
        "BCLC_before_intervention": [BCLC_before_intervention],
        "Preop_AFP": [Preop_AFP],
        "Surgical_margins": [Surgical_margins],
        "Edmondson_Steiner_Grade": [Edmondson_Steiner_Grade],
        "Number_of_tumors_on_the_specimen": [Number_of_tumors_on_the_specimen],
        "Cirrhosis": [Cirrhosis],
    }
)

train_data = pd.read_csv("hcc_pre_postoperative.csv", sep=";")
X, y = get_x_y(train_data, ["DFS", "DFS_Delay"], pos_label=1)
rsf = RandomSurvivalForest(n_estimators=250, max_depth=5, min_samples_leaf=1,min_samples_split=2, n_jobs=-1, random_state=20)
rsf.fit(X[df.columns], y)
# Predict survival function for the new data
survival_rsf = rsf.predict_survival_function(df)

# Define the time points to predict survival at
times_of_interest = np.linspace(1, 60, 60)  # From 0 to 40 with 100 points

d = pd.DataFrame(
    {
        "x_column": times_of_interest,
        "RSF": survival_rsf[0](times_of_interest),
        
    }
)
# Plot the survival curves
if st.button("Generate Plot"):
    st.subheader("Survival prediction using Random Survival Forest", divider=True)
    st.line_chart(
        data=d,
        x="x_column",
        y=["RSF"],
        x_label="Time in months",
        y_label="Survival Probability",
        color=["#FF0000"],
    )
    st.caption(
        "Interpretation note: Predicted curves represent **relative disease-free survival risk** "
        "and are intended for risk stratification and research purposes only. "
        "They should not be used as absolute decision thresholds for individual patient management."
    )







