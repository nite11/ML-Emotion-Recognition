import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Load your dataset (replace with your data loading code)
data = pd.read_csv('trainingdata.csv')

# Example: Assume 'features' are your independent variables and 'target' is your dependent variable
X = data.iloc[:, 1:]
y = data.iloc[:, 0]

scaler = StandardScaler()
#X = scaler.fit_transform(X)
# Step 2: Transform target labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 3: Train Linear Regression models (one-vs-rest)
models = {}
for class_label in le.classes_:
    y_binary = (y == class_label).astype(int)
    #print(y_binary)
    model = LinearRegression()
    model.fit(X, y_binary)
    models[class_label] = model

# Step 4: Predict probabilities for each class
y_pred_probs = pd.DataFrame({class_label: model.predict(X) for class_label, model in models.items()})
print(pd.concat([y, y_pred_probs], axis=1))

# Predict the class with the highest probability
y_pred = y_pred_probs.idxmax(axis=1)
print(y_pred)

# Step 5: Evaluate the model
print(classification_report(y, y_pred))
print("Accuracy:", accuracy_score(y, y_pred))
