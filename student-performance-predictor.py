# Student Performance Predictor
# This project predicts marks based on study habits.

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Step 1 : Create Sample dataset
data = {
    "hours_study": [1, 2, 3, 4, 5, 6, 7, 8],
    "attendance": [50, 60, 65, 70, 75, 80, 85, 90],
    "marks": [30, 35, 45, 50, 55, 65, 70, 80],
    "sleep_hours": [5, 6, 6, 7, 7, 8, 8, 9],
    "assignments_done": [2, 3, 4, 5, 6, 7, 8, 9]
}

# Convert to DataFrame and visualize data
df = pd.DataFrame(data)
plt.scatter(df["hours_study"], df["marks"])
plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Study Hours vs Marks")
plt.show()

# Step 3 : Define Features and Target
# Features (input)
X = df[["hours_study", "attendance", "sleep_hours", "assignments_done"]]

# Target (output)
y = df["marks"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

print("Model trained successfully!\n")

# Feature Importance
features = ["hours_study", "attendance", "sleep_hours", "assignments_done"]
importance = model.coef_

print("\nFeature Importance:")
for i in range(len(features)):
    print(f"{features[i]}: {importance[i]:.2f}")

# Plot feature Importance Graph
import matplotlib.pyplot as plt

plt.bar(features, importance)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Feature Importance in Prediction")
plt.show()
    
# Accuracy Check
y_pred = model.predict(X_test)
print("Model Accuracy (R2 Score):", r2_score(y_test, y_pred))

# Take user input
hours = float(input("Enter study hours: "))
attendance = float(input("Enter attendance: "))
sleep = float(input("Enter sleep hours:"))
assignments = float(input("Enter assignments done:"))

# Predict
new_data = pd.DataFrame([[ hours, attendance, sleep, assignments ]] , columns = ["hours_study" , "attendance" , "sleep_hours" , "assignments_done"])
prediction = model.predict(new_data)

print(f"Based on your input, predicted marks are: {prediction[0]:.2f}")
