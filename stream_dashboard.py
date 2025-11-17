import streamlit as st
import pandas as pd
import numpy as np
import joblib

#load model
model = joblib.load("xgb_model_ethical.pkl")


st.title("Employee Attrition Prediction")
st.write("Enter the candidate's information to predict if they will leave the company quickly.")

# Collect user input
total_working_years = st.slider("Total Working Years", 0, 40, 10)
hours_per_day = st.slider("Hours per Day", 4, 16, 8)
job_satisfaction = st.slider("Job Satisfaction (1-5)", 1, 5, 3)
business_travel = st.selectbox("Business Travel", ["Rarely", "Frequently", "No Travel"])
num_companies_worked = st.slider("Number of Companies Worked", 0, 10, 3)
work_life_balance = st.slider("Work-Life Balance (1-5)", 1, 5, 3)
environment_satisfaction = st.slider("Environment Satisfaction (1-5)", 1, 5, 3)
performance_rating = st.slider("Performance Rating (1-5)", 1, 5, 3)
training_times_last_year = st.slider("Training Times Last Year", 0, 10, 3)
job_involvement = st.slider("Job Involvement (1-5)", 1, 5, 3)
years_with_curr_manager = st.slider("Years With Current Manager", 0, 20, 5)
years_at_company = st.slider("Years at Company", 0, 40, 5)
department = st.selectbox("Department", ["Sales", "HR", "IT", "Finance", "Operations"])
years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 20, 3)
education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical", "Other"])
distance_from_home = st.slider("Distance From Home (km)", 0, 50, 10)
percent_salary_hike = st.slider("Percent Salary Hike", 0, 50, 10)
stock_option_level = st.slider("Stock Option Level", 0, 3, 1)
job_role = st.selectbox("Job Role", ["Manager", "Sales Executive", "Developer", "Research Scientist", "Other"])
monthly_income = st.number_input("Monthly Income ($)", min_value=500, max_value=20000, value=5000)
job_level = st.slider("Job Level", 1, 5, 2)
longest_absence = st.slider("Longest Absence (days)", 0, 30, 5)
education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
work_days = st.slider("Work Days per Week", 1, 7, 5)

input_data = pd.DataFrame({
    "TotalWorkingYears": [total_working_years],
    "HoursPerDay": [hours_per_day],
    "JobSatisfaction": [job_satisfaction],
    "BusinessTravel": [business_travel],
    "NumCompaniesWorked": [num_companies_worked],
    "WorkLifeBalance": [work_life_balance],
    "EnvironmentSatisfaction": [environment_satisfaction],
    "PerformanceRating": [performance_rating],
    "TrainingTimesLastYear": [training_times_last_year],
    "JobInvolvement": [job_involvement],
    "YearsWithCurrManager": [years_with_curr_manager],
    "YearsAtCompany": [years_at_company],
    "Department": [department],
    "YearsSinceLastPromotion": [years_since_last_promotion],
    "EducationField": [education_field],
    "DistanceFromHome": [distance_from_home],
    "PercentSalaryHike": [percent_salary_hike],
    "StockOptionLevel": [stock_option_level],
    "JobRole": [job_role],
    "MonthlyIncome": [monthly_income],
    "JobLevel": [job_level],
    "LongestAbsence": [longest_absence],
    "Education": [education],
    "WorkDays": [work_days]
})

#numerical features
numerical_features_ethic = [
    "Total Working Years", 
    "Hours per Day", 
    "Job Satisfaction (1-5)", 
    "Number of Companies Worked", 
    "Work-Life Balance (1-5)", 
    "Environment Satisfaction (1-5)", 
    "Performance Rating (1-5)", 
    "Training Times Last Year", 
    "Job Involvement (1-5)", 
    "Years With Current Manager", 
    "Years at Company", 
    "Years Since Last Promotion", 
    "Distance From Home (km)", 
    "Percent Salary Hike", 
    "Stock Option Level", 
    "Monthly Income ($)", 
    "Job Level", 
    "Longest Absence (days)", 
    "Work Days per Week"
]

categorical_features_ethic = [
    "Business Travel", 
    "Department", 
    "Education Field", 
    "Job Role", 
    "Education"
]

"""

input_data_prepared = full_pipeline.transform(input_data)

prediction = model.predict(input_data_prepared)[0]
prediction_proba = model.predict_proba(input_data_prepared)[0]"
""
""
"""

st.subheader("Prediction Result")
"""

if prediction == 1:
    st.error(f"This candidate is likely to leave the company. (Confidence: {prediction_proba[1]*100:.2f}%)")
else:
    st.success(f"This candidate is likely to stay. (Confidence: {prediction_proba[0]*100:.2f}%)")

st.write("*This prediction is based on past employee data and should be used as an additional insight.*")
"""