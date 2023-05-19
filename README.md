# Selection of Lung Cancer Mediating Genes

This project applies machine learning to gene expression data to predict if a gene is a mediator in lung cancer or not. Specifically, we use the XGBoost Classifier to perform this binary classification task.

## Prerequisites

Ensure that the following Python libraries are installed:

- pandas
- scipy
- scikit-learn
- xgboost
- matplotlib

You can install them via pip:

```bash
pip install pandas scipy scikit-learn xgboost matplotlib
```

You'll also need the following data files in the same directory as your notebook:

- `lung_expression_data.xlsx` : This is your main dataset which should contain gene expression data.
- `newncbi_Lung.csv` : This file should contain known gene names and aliases.

## Running the Code

1. Open Jupyter notebook. If you haven't installed Jupyter, you can install it with pip:

   ```bash
   pip install jupyter
   ```

   To run Jupyter notebook, use the following command:

   ```bash
   jupyter notebook
   ```

2. Navigate to the directory containing the Python code and data files.

3. Open a new Python notebook via the `New` button.

4. Copy the code into a new cell in the Jupyter notebook.

5. Run the cell by clicking `Run` or pressing `Shift+Enter`.

The script will load the gene expression data, preprocess it, perform t-tests to determine if a gene is overexpressed in cancer, train an XGBoost classifier, and make predictions. It will then evaluate the model and predict on the whole dataset to find all potential cancer mediating genes. It will also compare these predicted gene names against a list of known gene names and aliases, reporting how many match and don't match.

# Code Explainer

The code is divided into different sections, each performing a specific task. Here's a brief explanation of each part:

## 1. Importing Libraries:

```python
import pandas as pd
import csv
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
```

Here, we import all the necessary libraries for data manipulation (pandas), statistical testing (scipy), machine learning (sklearn, xgboost), and plotting (matplotlib).

## 2. Loading and Preprocessing the Data:

```python
df = pd.read_excel('lung_expression_data.xlsx').dropna()
df = df[df['GENE'] != 'NULL']

cancer_columns = [col for col in df.columns if col.startswith(('AD', 'L')) and not col.startswith('LN')]
normal_columns = [col for col in df.columns if col.startswith('LN')]
```

We load the data from an Excel file into a pandas DataFrame. We then drop any rows with missing values, and any rows where the 'GENE' value is 'NULL'. The columns are divided into two categories: those that represent cancerous samples and those that represent normal samples.

## 3. Performing t-tests:

```python
df['p_value'] = [ttest_ind(row[cancer_columns], row[normal_columns]).pvalue for _, row in df.iterrows()]
```

Here, we perform a t-test for each row, comparing the cancerous samples to the normal samples. We store the p-value of the t-test in a new column. A lower p-value indicates that the means of the cancerous and normal samples are significantly different.

## 4. Creating Labels for the Model:

```python
df['is_cancer_mediator'] = ((df['p_value'] < 0.05) & (df[cancer_columns].mean(axis=1) > df[normal_columns].mean(axis=1))).astype(int)
```

We then label each gene as a potential cancer mediator (1) or not (0) based on the p-value and the mean difference between the cancerous and normal samples.

## 5. Training the Model:

```python
X_train, X_test, y_train, y_test = train_test_split(df[cancer_columns + normal_columns], df['is_cancer_mediator'], test_size=0.2, random_state=42)
clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss').fit(X_train, y_train)
```

We split the data into a training set and a test set. Then, we create an XGBoost classifier and fit the model using the training data.

## 6. Evaluating the Model:

```python
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

We use the model to make predictions on the test data, and then print a classification report, which includes precision, recall, f1-score, and support for both classes.

## 7. Plotting the ROC Curve:

```python
y_pred_proba = clf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.2f}')
```

Here, we calculate the probabilities of the positive class, and use these to compute the ROC curve, which is a plot of the true positive rate (sensitivity) against the false positive rate (1-specificity) at

## Results

The output will include a classification report which includes precision, recall, f1-score, and support for both classes. The model's accuracy and the area under the ROC curve (AUC-ROC) will also be displayed. The ROC curve itself will be plotted.

Finally, it will print the number of predicted cancer mediating genes that match and don't match the known list, and save the predicted gene names to a CSV file named `predicted_genes.csv`.

## License

This code is licensed under the [MIT License](https://opensource.org/license/mit). Feel free to modify and use it according to your needs.

## Author

This code was written by [Adarsh Kumar](https://github.com/idealadarsh).
