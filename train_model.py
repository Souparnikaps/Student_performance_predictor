import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("datasets/student_data.csv")

#heatmap
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(data.corr(), annot=True, cmap="coolwarm")

plt.title("Correlation Between Student Performance Factors")
plt.show()

# Select features
X = data[['Attendance', 'StudyHours', 'InternalMarks', 'Assignments']]

# Target variable
y = data['FinalMarks']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)
rf_predictions = rf_model.predict(X_test)

# Evaluate
print("Mean Squared Error:", mean_squared_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))
print("Random Forest Results")
print("Mean Squared Error:", mean_squared_error(y_test, rf_predictions))
print("R2 Score:", r2_score(y_test, rf_predictions))
new_student = [[85, 3, 17, 16]]

prediction = model.predict(new_student)

print("Predicted Final Marks:", round(prediction[0], 2))

rf_prediction = rf_model.predict(new_student)

print("Random Forest Predicted Marks:", round(rf_prediction[0],2))

import matplotlib.pyplot as plt

plt.scatter(y_test, predictions)
plt.xlabel("Actual Marks")
plt.ylabel("Predicted Marks")
plt.title("Actual vs Predicted Marks")
plt.show()

import pickle

pickle.dump(model, open("student_model.pkl", "wb"))