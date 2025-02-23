import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Sample data (replace this with actual dataset)
data = {
    'Project ID': ['P001', 'P002', 'P003'],
    'Budget (₹ Crores)': [50, 30, 100],
    'Planned Duration (Months)': [24, 12, 36],
    'Actual Duration (Months)': [30, 12, 40],
    'Workforce Size': [150, 100, 250],
    'Material Costs (₹)': [20, 10, 40],
    'Risk Factors': ['Weather, Labor Strike', 'None', 'Material Shortage'],
    'Completion Status': ['Delayed', 'On-time', 'Delayed'],
    'Incident Logs': ['Minor Injuries', 'None', 'Equipment Failure'],
    'Risk Level': ['High', 'Low', 'High']
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode categorical columns
df['Risk Level'] = label_encoder.fit_transform(df['Risk Level'])
df['Risk Factors'] = label_encoder.fit_transform(df['Risk Factors'])
df['Completion Status'] = label_encoder.fit_transform(df['Completion Status'])
df['Incident Logs'] = label_encoder.fit_transform(df['Incident Logs'])

# Features (X) and target (y)
X = df.drop(columns=['Project ID', 'Risk Level'])
y = df['Risk Level']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, feature_importance)
plt.title('Feature Importance')
plt.show()