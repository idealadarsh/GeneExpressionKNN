# Gene Expression Analysis

This code performs gene expression analysis using the provided dataset. It calculates the p-values for each gene and selects genes with a p-value less than 0.05. The selected genes are then saved to an Excel file.

## Prerequisites

- Python 3.x
- Required Python packages: scikit-learn, pandas, numpy, statsmodels

You can install the required packages by running the following command in a Jupyter Notebook cell:

pythonCopy code

`!pip install scikit-learn pandas numpy statsmodels`

## Code Explanation

1.  The necessary packages are imported:
    `import pandas as pd`
    `import numpy as np`
    `from scipy.stats import ttest_ind`
2.  The dataset is loaded from the Excel file named `lung_expression_data.xlsx`:  
    `data = pd.read_excel('lung_expression_data.xlsx')`
3.  Rows with gene name "NULL" or empty gene names are removed from the dataset:
    `data = data[data['GENE'] != 'NULL']`
    `data = data[data['GENE'] != '']`
4.  The columns are split into two groups: cancerous and normal. Columns starting with 'AD' represent cancerous samples, and columns starting with 'L' represent normal samples:  
     `cancerous_cols = [col for col in data.columns if col.startswith('AD')]
normal_cols = [col for col in data.columns if col.startswith('L')]`
5.  A loop is performed over each row of the dataset to calculate the t-test and p-value for each gene:
    `p_values = []
for index, row in data.iterrows():
    cancerous_values = pd.to_numeric(row[cancerous_cols], errors='coerce')
    normal_values = pd.to_numeric(row[normal_cols], errors='coerce')
    t_stat, p_val = ttest_ind(cancerous_values, normal_values, nan_policy='omit')
    p_values.append(p_val)`
6.  The calculated p-values are added to the dataset:
    `data['p_value'] = p_values`
7.  Genes with a p-value less than 0.05 are selected:
    `selected_genes = data[data['p_value'] < 0.05]`
8.  The selected genes are displayed in the console:
    `print(selected_genes)`
9.  The selected genes are saved to an Excel file named `selected_genes.xlsx`:
    `selected_genes.to_excel("selected_genes.xlsx")`

### T-test

A t-test is a statistical hypothesis test used to determine if there is a significant difference between the means of two groups. It helps to assess whether the difference observed between the groups is likely due to chance or if it represents a real difference in the population.

In the context of gene expression analysis, a t-test can be used to compare the expression levels of a particular gene in cancerous samples and normal samples. The null hypothesis assumes that there is no difference in the mean expression levels of the gene between the two groups. The alternative hypothesis suggests that there is a significant difference.

The t-test calculates a t-statistic, which measures the difference between the means relative to the variation within each group. The magnitude of the t-statistic and its corresponding p-value are used to determine the statistical significance of the observed difference.

### P-value

The p-value is a measure of the evidence against the null hypothesis in a statistical hypothesis test. It quantifies the probability of observing a test statistic as extreme as, or more extreme than, the one calculated from the data, assuming that the null hypothesis is true.

In the context of gene expression analysis, the p-value associated with the t-test represents the probability of obtaining the observed difference in expression levels between cancerous and normal samples by chance alone, assuming that there is no true difference. A low p-value indicates that the observed difference is unlikely to be due to chance, suggesting that there may be a significant difference in the gene expression levels between the two groups.

Typically, a threshold value (e.g., 0.05) is chosen as the significance level. If the p-value is less than the significance level, the null hypothesis is rejected, and it is concluded that there is evidence of a significant difference. If the p-value is greater than or equal to the significance level, there is insufficient evidence to reject the null hypothesis.

It's important to note that the p-value alone does not provide information about the magnitude or direction of the difference. It only indicates whether the observed difference is statistically significant or not. Other measures, such as effect size, should be considered to assess the practical significance of the difference.

## Usage

1.  Make sure you have Python 3.x installed on your system.
2.  Install the required Python packages by running the above command in a Jupyter Notebook cell.
3.  Download the dataset file named `lung_expression_data.xlsx` and place it in the same directory as the Jupyter Notebook.
4.  Run the code cells in the Jupyter Notebook to execute the analysis.

## Dataset

The dataset used for gene expression analysis is provided in an Excel file named `lung_expression_data.xlsx`. The dataset should have the following structure:

| GENE | In 4966 | AD001 | AD002 | AD003 | ... | L001 | L002 | L003 |
| ---- | ------- | ----- | ----- | ----- | --- | ---- | ---- | ---- |
| ...  | ...     | ...   | ...   | ...   | ... | ...  | ...  | ...  |
| ...  | ...     | ...   | ...   | ...   | ... | ...  | ...  | ...  |

- The `GENE` column contains the gene names.
- Columns starting with `AD` represent cancerous samples.
- Columns starting with `L` represent normal samples.

Ensure that the dataset file is placed in the same directory as the Jupyter Notebook.

## Output

The Jupyter Notebook cells will output the selected genes with a p-value less than 0.05. The selected genes will be displayed in a tabular format.

Additionally, the selected genes will be saved to an Excel file named `selected_genes.xlsx` in the same directory as the Jupyter Notebook. The file will contain the gene names and corresponding expression values.

Please note that if there are no genes with a p-value less than 0.05, the output file will be empty.

## License

This code is licensed under the [MIT License](https://opensource.org/license/mit). Feel free to modify and use it according to your needs.
