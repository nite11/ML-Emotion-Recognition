from skopt import BayesSearchCV
from skopt.space import Real, Integer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

data = pd.read_csv('trainingdata.csv')  
X = data.iloc[:, 1:]
y = data.iloc[:, 0]


file1 = open('RF_results.txt', 'a+')
for random_state in np.random.randint(1, 100, size=5):

    #############################################################################
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Define the hyperparameter search space
    param_space = {
        'n_estimators': Integer(100, 500), #Number of trees in the forest
        'max_features': Real(0.1, 1.0) #Fraction of features to consider when looking for the best split
        
        
    }

    # Initialize the Random Forest Classifier
    rf = RandomForestClassifier(random_state=random_state)

    # Define the cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Initialize Bayesian Search
    opt = BayesSearchCV(
        estimator=rf,
        search_spaces=param_space,
        n_iter=4,  # Number of parameter settings sampled
        cv=cv,
        n_jobs=-1, # The number of jobs to run in parallel (-1 means using all processors)
        scoring='accuracy',
        random_state=random_state
    )

    # Perform the search
    opt.fit(X_train, y_train)

    # Print the best parameters and the corresponding score
    file1.write(f"random_state: , {random_state}\n")
    file1.write(f"Best parameters found: , {opt.best_params_}\n")
    file1.write(f"Best cross-validation accuracy: , {opt.best_score_}\n")
    


    model = opt.best_estimator_

    accuracy_list = []

    # Set up k-fold cross-validation
    k=5
    for i in range(0, y.shape[0], k):
        indices_to_test = [j for j in range(i,i+k) if j < y.shape[0]]    
        #print(X.drop(indices_to_test), X.iloc[indices_to_test])
        X_train, X_test = X.drop(indices_to_test), X.iloc[indices_to_test]
        y_train, y_test = y.drop(indices_to_test), y.iloc[indices_to_test]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy_list.append(accuracy_score(y_test, y_pred))
        

    # Calculate mean accuracy across all folds
    mean_accuracy = sum(accuracy_list) / len(accuracy_list)
    file1.write(f"Mean Accuracy: {mean_accuracy:.2f}\n\n")
file1.close()
