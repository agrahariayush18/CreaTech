import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
data = {
    "Project_ID": ["P001", "P002", "P003"],
    "Budget": [50, 30, 100],  # in Crores
    "Planned_Duration": [24, 12, 36],  # in Months
    "Actual_Duration": [30, 12, 40],  # in Months
    "Workforce_Size": [150, 100, 250],
    "Material_Costs": [20, 10, 40],  # in Crores
    "Weather_Conditions": ["Rainy", "Sunny", "Stormy"],
    "Risk_Level": ["High", "Low", "High"]
}
df = pd.DataFrame(data)

# Dashboard Title
st.title("Construction Project Monitoring Dashboard")
st.subheader("Visualizing Budgets, Durations, and Resource Allocation")

# Section 1: Budget vs Material Costs
st.header("Budget vs Material Costs")
plt.figure(figsize=(8, 6))
sns.barplot(x="Project_ID", y="Budget", data=df, color="blue", label="Budget")
sns.barplot(x="Project_ID", y="Material_Costs", data=df, color="orange", label="Material Costs")
plt.xlabel("Project ID")
plt.ylabel("Amount (₹ Crores)")
plt.title("Budget vs Material Costs")
plt.legend()
st.pyplot(plt)

# Section 2: Planned vs Actual Duration
st.header("Planned vs Actual Duration")
plt.figure(figsize=(8, 6))
sns.barplot(x="Project_ID", y="Planned_Duration", data=df, color="green", label="Planned Duration")
sns.barplot(x="Project_ID", y="Actual_Duration", data=df, color="red", label="Actual Duration")
plt.xlabel("Project ID")
plt.ylabel("Duration (Months)")
plt.title("Planned vs Actual Duration")
plt.legend()
st.pyplot(plt)

# Section 3: Workforce Size Distribution
st.header("Workforce Size Distribution")
plt.figure(figsize=(8, 6))
sns.barplot(x="Project_ID", y="Workforce_Size", data=df, palette="Blues_d")
plt.xlabel("Project ID")
plt.ylabel("Workforce Size")
plt.title("Workforce Size Across Projects")
st.pyplot(plt)

# Section 4: Risk Level Heatmap
st.header("Risk Level Heatmap")
risk_map = df.pivot(index="Project_ID", columns="Weather_Conditions", values="Actual_Duration")
plt.figure(figsize=(8, 6))
sns.heatmap(risk_map, annot=True, cmap="coolwarm", cbar_kws={'label': 'Risk Level'})
plt.title("Risk Level by Weather Conditions")
st.pyplot(plt)