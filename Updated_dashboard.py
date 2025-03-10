import streamlit as st
import pandas as pd
import numpy as np
import cv2
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Synthetic Dataset Creation
def create_synthetic_data():
    data = {
        "Project ID": ["P001", "P002", "P003", "P004", "P005"],
        "Budget (INR)": [500000000, 750000000, 1000000000, 300000000, 600000000],
        "Planned Duration (Days)": [365, 450, 500, 200, 300],
        "Actual Duration (Days)": [400, 420, 550, 210, 320],
        "Workforce Size": [200, 300, 400, 150, 250],
        "Material Costs (INR)": [200000000, 300000000, 400000000, 100000000, 250000000],
        "Weather Conditions": ["Rainy", "Clear", "Stormy", "Sunny", "Cloudy"],
        "Risk Level": ["High", "Medium", "High", "Low", "Medium"],
        "Supply Chain Status": ["Disrupted", "Stable", "Partially Disrupted", "Stable", "Disrupted"],
        "Progress (%)": [75, 85, 60, 90, 70],
        "Cost Efficiency (%)": [85, 90, 80, 95, 88],
        "Workforce Utilization (%)": [90, 95, 85, 98, 88],
        "Carbon Emissions (kg)": [5000, 4000, 6000, 3000, 4500],
        "Material Waste (%)": [15, 10, 20, 5, 12]
    }
    return pd.DataFrame(data)

# Progress Estimation Using Edge Detection
def estimate_progress(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    progress = np.mean(edges) / 255 * 100  # Normalize to percentage
    return progress

# Streamlit App
def main():
    st.title("Construction Project Monitoring Dashboard")
    st.sidebar.subheader("Upload or Use Synthetic Data")

    # Option to upload custom dataset
    uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        df = create_synthetic_data()

    # Display Dataset
    st.subheader("Project Data Overview")
    st.dataframe(df)

    # Risk Analysis
    st.subheader("Risk Analysis")
    fig, ax = plt.subplots()
    risk_levels = df["Risk Level"].value_counts()
    ax.pie(risk_levels, labels=risk_levels.index, autopct='%1.1f%%', startangle=90)
    ax.set_title("Risk Level Distribution")
    st.pyplot(fig)

    # Supply Chain Resilience
    st.subheader("Supply Chain Resilience")
    fig, ax = plt.subplots()
    supply_chain_status = df["Supply Chain Status"].value_counts()
    sns.barplot(x=supply_chain_status.index, y=supply_chain_status.values, ax=ax)
    ax.set_title("Supply Chain Status")
    ax.set_ylabel("Number of Projects")
    st.pyplot(fig)

    # Workforce Optimization
    st.subheader("Workforce Optimization")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Workforce Size", y="Workforce Utilization (%)", hue="Risk Level", size="Progress (%)", sizes=(50, 200), ax=ax)
    ax.set_title("Workforce Utilization vs. Size")
    ax.set_xlabel("Workforce Size")
    ax.set_ylabel("Workforce Utilization (%)")
    st.pyplot(fig)

    # Cost Efficiency
    st.subheader("Cost Efficiency Analysis")
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="Project ID", y="Cost Efficiency (%)", hue="Risk Level", ax=ax)
    ax.set_title("Cost Efficiency by Project")
    ax.set_ylabel("Cost Efficiency (%)")
    st.pyplot(fig)

    # Carbon Emissions
    st.subheader("Carbon Emissions")
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="Project ID", y="Carbon Emissions (kg)", ax=ax)
    ax.set_title("Carbon Emissions by Project")
    ax.set_ylabel("Carbon Emissions (kg)")
    st.pyplot(fig)

    # Material Waste
    st.subheader("Material Waste")
    fig, ax = plt.subplots()
    sns.barplot(data=df, x="Project ID", y="Material Waste (%)", ax=ax)
    ax.set_title("Material Waste by Project")
    ax.set_ylabel("Material Waste (%)")
    st.pyplot(fig)

    # Real-Time Image Monitoring
    st.subheader("Real-Time Image Monitoring")
    image_file = st.file_uploader("Upload an Image of the Construction Site", type=["jpg", "png"])
    if image_file:
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
        progress = estimate_progress(image_file)
        st.image(image, channels="BGR", caption="Uploaded Image")
        st.write(f"Estimated Progress: {progress:.2f}%")

if __name__ == "__main__":
    main()