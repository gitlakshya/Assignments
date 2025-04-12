import streamlit as st
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle


@st.cache_resource
def load_model():
    with open("C:/Users/lakshya.vashisth/Documents/Assignments/ML/app/credit_risk_model.pkl", "rb") as file:
        loaded_model = pickle.load(file)
    return loaded_model

# Load the dataset
@st.cache_resource
def load_data():
    data = pd.read_csv('C:/Users/lakshya.vashisth/Documents/Assignments/ML/app/credit_risk.csv')
    return data



st.set_page_config(
    page_title="Credit Risk Analysis App",
    page_icon="📈",
    layout="centered"
)

# st.title("Credit Risk Analysis Application")
st.sidebar.header("Switch Tab")

# Sidebar
option = st.sidebar.selectbox("Choose Section", ["Home - Credit Risk Prediction", "Data Analysis"])


status_mapping = {"Yes": 1, "No": 0}
if option == "Home - Credit Risk Prediction":
    st.header("Welcome to Credit Risk Analysis App")
    st.write("Explore credit risk modeling and business insights interactively.")
    
    model = load_model()
    
    st.write("Input Data:")

# Input form
    def user_input():
        st.header("Enter Applicant Information")  
            
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        purpose = st.selectbox("Purpose", ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
        home_ownership = st.selectbox("Home Ownership status", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        income = st.number_input("Income", min_value=0, value=50000, max_value=1000000)
        emp_length = st.number_input("Employment Length (years)", min_value=0, value=5)
        amount = st.number_input("Loan Amount", min_value=0, value=10000)
            
        loan_int_rate = st.slider(
        "Loan Interest Rate (%)",
        min_value=1.0,
        max_value=30.0,
        value=10.0,
        step=0.1
        )

        cred_length = st.number_input("Credit Length (in years)", min_value=0.0, value=1.0, max_value=99.0)    
        loan_percent_income = amount / income
        status = st.selectbox("Loan Approval Status", options=list(status_mapping.keys()))

        data = {
            'Age': age,
            'Income': income,
            'Emp_length': emp_length,
            'Amount': amount,
            'Rate': loan_int_rate,
            'Cred_length': cred_length,
            'Intent': purpose,
            'Home': home_ownership,
            'Percent_income': loan_percent_income,
            'Status': status_mapping[status]   
        }

        features = pd.DataFrame(data, index=[0])
        return features

    input_df = user_input()

        # Make prediction
    if st.button("Predict"):
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        st.subheader("Probability of Credit Risk")
        if prediction[0] == 1:
            st.markdown('<span style="color:red">*High Credit Risk*</span>', unsafe_allow_html=True)
            st.write("**Probability of High Risk:**", round(prediction_proba[0][1],4))
        else:
            st.markdown('<span style="color:green">*Low Credit Risk*</span>', unsafe_allow_html=True)
            st.write("**Probability of High Risk:**", round(prediction_proba[0][1],4))

elif option == "Data Analysis":

#     st.title("Data Analysis")

# if 'prediction_data' in st.session_state and st.session_state.get('predicted', False):
#     data = st.session_state['prediction_data']
#     df = pd.DataFrame([data]) # Create a DataFrame for easier plotting

#     st.subheader("Input Data")
#     st.write(df)

#     st.subheader("Data Visualization")

#         # Example 1: Bar chart of numerical features
#     numerical_cols = ['Age', 'Income', 'Employment Length (years)']
#     if numerical_cols:
#         df_numerical = df[numerical_cols].melt(var_name='Feature', value_name='Value')
#         fig_bar, ax_bar = plt.subplots()
#         sns.barplot(x='Feature', y='Value', data=df_numerical, ax=ax_bar)
#         plt.title('Applicant Numerical Features')
#         st.pyplot(fig_bar)

#         # Example 2: Count plot of categorical features
#     categorical_cols = ['Purpose', 'Education', 'Home Ownership status']
#     for col in categorical_cols:
#         fig_count, ax_count = plt.subplots()
#         sns.countplot(x=col, data=df, ax=ax_count)
#         plt.title(f'Applicant {col} Distribution')
#         st.pyplot(fig_count)

#         # You can add more sophisticated analysis and visualizations here
#         # Potentially using the prediction result if you stored it in session state

# else:
#     st.info("No prediction data available yet. Please go back to the Home page and click Predict.")
        



# Preprocess the data
    # def preprocess_data(data):
    #     # Handle missing values, encode categorical variables, etc.
    #     data.fillna(data.mean(), inplace=True)
    #     data = pd.get_dummies(data, drop_first=True)
    #     return data

    # Feature Importance
    # def feature_importance(X, y):
    #     model = RandomForestClassifier()
    #     model.fit(X, y)
    #     importance = model.feature_importances_
    #     return importance

    
    st.title("Credit Risk Analysis")

        # Load and preprocess data
    data = load_data()
    st.write("Dataset Overview:")
    st.write(data.head())

        # Preprocess the data
    processed_data = data

        # Split the data into features and target
    X = processed_data.drop('Default', axis=1)  # Replace 'loan_default' with your target column
    y = processed_data['Default']


        # Pie Chart for Loan Intent Distribution
    st.subheader("Loan Intent Distribution")
    loan_intent_counts = data['Intent'].value_counts()  # Replace 'loan_intent' with your feature
    plt.figure(figsize=(8, 6))
    plt.pie(loan_intent_counts, labels=loan_intent_counts.index, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(plt)

        # Heatmap of correlations for numeric features only
    st.subheader("Correlation Heatmap for Numeric Features")
    numeric_features = processed_data.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_features.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    st.pyplot(plt)



    # Visualizations
    st.subheader("Distribution of Loan Amounts")
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Amount'], bins=30, kde=True)  # Replace 'loan_amount' with your feature
    st.pyplot(plt)

    st.subheader("Default Rate by Employment Length")
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Emp_length', hue='Default', data=data)  # Replace 'employment_status' with your feature
    st.pyplot(plt)


    st.subheader("Loan Default by Age Group")
    data['age_group'] = pd.cut(data['Age'], bins=[20, 30, 40, 50, 60, 70, 80], labels=['20-30', '30-40', '40-50', '50-60', '60-70', '70-80'])  # Replace 'age' with your feature
    plt.figure(figsize=(10, 6))
    sns.countplot(x='age_group', hue='Default', data=data)
    st.pyplot(plt)

    selected_features_df = data[['Income', 'Amount', 'Rate']]
    selected_features_df['Default'] = y # Adding the target variable 'Default' as a column

# Create a pair plot (scatter plot matrix)
    st.subheader("Features Affecting the Decision Mostly")
    sns.pairplot(selected_features_df, hue='Default', diag_kind='kde')  # Now using 'Default' as the hue
    plt.suptitle('Scatter Plot of Most Important Features (Income, Amount, Rate)', y=1.02)
    st.pyplot(plt)
