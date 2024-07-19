import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pymrmr

data = pd.read_csv('trainingdata.csv')  
X = data.iloc[:, 1:]
y = data.iloc[:, 0]

feature_names = X.columns
random_state=14


model  = KNeighborsClassifier(n_neighbors=4)

mi_scores = mutual_info_classif(X, y)

# Create a DataFrame to display the MI scores
mi_df = pd.DataFrame({
    'Feature': feature_names,
    'MI Score': mi_scores
}).sort_values(by='MI Score', ascending=False)

#print("Mutual Information Scores:")
#print(mi_df)

# Select top 2 features based on MI scores
top_features = mi_df['Feature'].head(45).values
#print("Selected Top Features:", top_features)

# Get the indices of the selected features
selected_indices = [i for i, feature in enumerate(feature_names) if feature in top_features]
#print(selected_indices)
X = X.iloc[:, selected_indices]

accuracy_list = []

# Set up k-fold cross-validation
k=5
for i in range(0, y.shape[0], k):
    indices_to_test = [j for j in range(i,i+k) if j < y.shape[0]] 
    X_train, X_test = X.drop(indices_to_test), X.iloc[indices_to_test]
    y_train, y_test = y.drop(indices_to_test), y.iloc[indices_to_test]

    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy_list.append(accuracy_score(y_test, y_pred))
    

# Calculate mean accuracy across all folds
mean_accuracy = sum(accuracy_list) / len(accuracy_list)
print(f"Mean Accuracy: {mean_accuracy:.2f}")