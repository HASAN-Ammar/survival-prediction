import streamlit as st
import pandas as pd
import numpy as np
from sksurv.datasets import get_x_y
from sksurv.ensemble import RandomSurvivalForest


st.title("Disease-free survival prediction after liver resection for hepatocellular carcinoma")

st.write("### Input Data")
col1, col2 = st.columns(2)

Largest_nodule_diameter = col1.number_input("Size of the largest nodule in mm", min_value=1, value= 10, placeholder="Size in millimetes")
Satellite_nodules = col1.number_input("Satellite nodules", min_value=0, max_value=1, value=0, placeholder= "0/1")
VETC_subtype = col1.number_input("VETC subtype", min_value=0, max_value=1, value=0, placeholder= "0/1")
Macrotrabecular_massive_subtype = col2.number_input("Macrotrabecular massive subtype", min_value=0, max_value=1, value=0, placeholder= "0/1")
Microvascular_invasion = col2.number_input("Microvascular invasion", min_value=0, max_value=1, value=0, placeholder= "0/1")
BCLC_before_intervention = col1.number_input("BCLC before surgery (0, A=1, B=2, C=3)", min_value=0,max_value=3, value=0, placeholder="0-3")
Edmondson_Steiner_Grade = col1.number_input("Edmondson steiner grade (0=I, II=1, III=2, IV=3, V=4)", min_value=0, max_value=4, value=0)
Number_of_tumors_on_the_specimen = col2.number_input("Number of tumors on the specimen", min_value=0, max_value=4, value=0, placeholder= "0-4")
Preop_AFP = col2.number_input("Preoperative AFP level (ng/ml)", min_value=0.0, value=1.0, placeholder="AFP level (ng/ml)")
Gender = col2.number_input("Gender (F=0, M=1)", min_value=0, max_value=1, value=0, placeholder= "0/1")
# AFP level (ng/ml)

df = pd.DataFrame(
    {
        "Satellite_nodules": [Satellite_nodules],
        "Largest_nodule_diameter": [Largest_nodule_diameter],
        "VETC_subtype": [VETC_subtype],
        "Microvascular_invasion": [Microvascular_invasion],
        "BCLC_before_intervention": [BCLC_before_intervention],
        "Preop_AFP": [Preop_AFP],
        "Macrotrabecular_massive_subtype": [Macrotrabecular_massive_subtype],
        "Edmondson_Steiner_Grade": [Edmondson_Steiner_Grade],
        "Number_of_tumors_on_the_specimen": [Number_of_tumors_on_the_specimen],
        "Gender": [Gender],
    }
)

train_data = pd.read_csv("hcc_dataset.csv")
X, y = get_x_y(train_data, ["DFS", "DFS_Delay"], pos_label=1)

rsf = RandomSurvivalForest(n_estimators=250, max_depth=6, n_jobs=-1, random_state=20)
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
