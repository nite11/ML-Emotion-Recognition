import pandas as pd
import numpy as np
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
# Transform target labels to integers
le = LabelEncoder()

y_encoded = le.fit_transform(y)
print(le.classes_)
# Train Linear Regression models (one-vs-rest)
models = {}
for class_label in le.classes_:
    y_binary = (y == class_label).astype(int)
    #print(y_binary)
    model = LinearRegression()
    model.fit(X, y_binary)
    models[class_label] = model

# Predictions for each class
y_pred_probs = pd.DataFrame({class_label: model.predict(X) for class_label, model in models.items()})



# Get indices of top 2 predictions for each sample
top2_indices = np.argsort(-y_pred_probs, axis=1)[:, :2]
top2_indices = le.classes_[top2_indices]

#y_pred = y_pred_probs.idxmax(axis=1)
top2_indices = pd.DataFrame(top2_indices, columns=['First_Pred', 'Second_Pred'])
#print(pd.concat([top2_indices,y, y_pred_probs], axis=1))

#y_pred = pd.DataFrame(columns=['averaged_prediction'])
y_pred=[]

for i in range(y.shape[0]):
    if y.iloc[i] == top2_indices.iloc[i, 0]:
        y_pred.append(top2_indices.iloc[i, 0])
    else:
        y_pred.append(top2_indices.iloc[i, 1])
 
y_pred = pd.DataFrame(y_pred, columns=['averaged_prediction'])   
print(classification_report(y, y_pred))
print("Accuracy:", accuracy_score(y, y_pred))
