# Lung Cancer Detection using KNN and Genetic Algorithm for Optimization

Lung cancer is one of the most common types of cancer that affects people worldwide. Early detection of lung cancer can significantly increase the survival rate of the patient. In this research paper, we present a method for detecting lung cancer using K-Nearest Neighbors (KNN) algorithm with Genetic Algorithm for Hyperparameter Optimization.

## Getting Started

To run this code, you will need the following software:

- Python 3
- pandas
- matplotlib
- scikit-learn
- joblib
- TPOT

To install the required packages, you can use pip:

Copy code

`pip install pandas matplotlib scikit-learn joblib tpot`

## Usage

1.  Clone the repository and navigate to the directory:

        `https://github.com/idealadarsh/GeneExpressionKNN.git

    cd GeneExpressionKNN`

2.  Download the gene expression data file 'lung_data.xlsx' and save it in the same directory as the code.
3.  Run the Jupyter Notebook:
4.  The program will output the accuracy of the model and display a scatter plot of the gene expression data with the results labeled.
5.  The optimized model will be saved as 'lung_cancer_model.joblib' in the same directory as the code.

## Parameters

The TPOTClassifier parameters can be adjusted for different optimization. These are the current parameters in the code:

makefileCopy code

`generations=5
population_size=50
verbosity=2
config_dict={'sklearn.neighbors.KNeighborsClassifier': {'n_neighbors': range(1, 11)}}`

- generations - the number of generations to run the genetic algorithm
- population_size - the number of individuals in each generation
- verbosity - how much output to display during the optimization process
- config_dict - the parameters to optimize for the KNN classifier

## Dataset

The dataset used in this research is lung cancer data in an Excel sheet named `lung_cancer_data.xlsx`. The columns represent different input features and the target variable 'Result'. The target variable indicates if the patient has lung cancer or not.

## Preprocessing

We start by importing the necessary libraries, including pandas for reading the dataset, matplotlib for plotting, sklearn for machine learning algorithms, and joblib for saving the trained model.

`import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
from tpot import TPOTClassifier`

Next, we read the Excel sheet into a pandas dataframe and replace 'Yes' with 1 and 'No' with 0 in the 'Result' column.

`# Reading the Excel sheet
data = pd.read_excel('lung_cancer_data.xlsx')`

## Replacing 'YES' with 1 and 'NO' with 0 in the 'Result' column

`data['Result'] = data['Result'].replace({'YES': 1, 'NO': 0})`

After preprocessing, we separate the input features and the target variable and split the data into training and testing sets using the `train_test_split()` method from sklearn.

`# Separating the input features and the target variable
X = data.iloc[:, 3:]
y = data['Result']`

## Splitting the data into training and testing sets

`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)`

## Genetic Algorithm for Hyperparameter Optimization

To optimize the hyperparameters of the KNN algorithm, we use the Tree-Based Pipeline Optimization Tool (TPOT), a machine learning tool that optimizes machine learning pipelines using genetic programming.

`# Running the genetic algorithm for hyperparameter optimization
tpot = TPOTClassifier(generations=5, population_size=50, verbosity=2, 
                      config_dict={'sklearn.neighbors.KNeighborsClassifier': {'n_neighbors': range(1, 11)}})
tpot.fit(X_train, y_train)`

## Getting the best model found by the genetic algorithm

`model = tpot.fitted_pipeline_`

We set the generations to 5 and the population size to 50. We also specify the range of the n_neighbors parameter for the KNN algorithm to optimize. After running the genetic algorithm, we get the best model found by TPOT and store it in the `model` variable.

## Model Evaluation

We make predictions on the test set using the `predict()` method of the `model` variable and calculate the accuracy of the model using the `accuracy_score()` method from sklearn.

`# Making predictions on the test set
y_pred = model.predict(X_test)`

## Calculating the accuracy of the model

`accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)`

Finally, we save the trained model using the `dump()` method from joblib.

`# Saving the trained model
dump(model, 'model.joblib')`

## Results

The accuracy of the model is displayed using the `print()` function.
