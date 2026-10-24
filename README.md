# Regression and Regularization: Two Machine Learning Case Studies

This notebook explores **Logistic Regression for Alzheimer's diagnosis classification** and **Linear Regression for concrete compressive strength prediction**. It compares unregularized models with L1 and L2 regularization, evaluates predictions, and interprets model coefficients.

## Project Overview

| Case study | Task | Models |
| --- | --- | --- |
| Alzheimer's diagnosis | Binary classification | Unregularized, L1, and L2 Logistic Regression |
| Concrete compressive strength | Numerical prediction | Linear Regression, Ridge, and Lasso |

Both studies use an **80/20 train/test split** with `random_state=42`. Scaling is fitted on training data and applied to the test data.

## 1. Alzheimer's Diagnosis Classification

### Dataset

The notebook loads `alzheimers_disease_data.csv`. Its saved output shows **2,149 rows and 35 columns**, with no missing values.

- **Target:** `Diagnosis` (`0` = No AD, `1` = AD).
- **Excluded columns:** `PatientID` and `DoctorInCharge`, when present.
- **Predictors:** the remaining 32 columns.

### Workflow

1. Inspect data types, missing values, duplicate counts, descriptive statistics, and class proportions.
2. Remove identifier and doctor columns.
3. Split the data with stratification to preserve class proportions.
4. Use `ColumnTransformer` to standardize numeric predictors and one-hot encode nonnumeric predictors.
5. Compare Logistic Regression without regularization, with L2 regularization, and with L1 regularization.
6. Report classification metrics and training/test ROC-AUC.
7. Interpret the unregularized model's coefficients and odds ratios.
8. Plot the L1 model's confusion matrix, ROC curve, and precision-recall curve.

In the supplied dataset, the retained predictors are all numeric. Numeric-coded categories, including `Ethnicity`, are therefore standardized rather than one-hot encoded by the current implementation.

### Results Recorded in the Notebook

| Model | Test accuracy | Test precision | Test recall | Test F1 | Train ROC-AUC | Test ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Unregularized | 0.8163 | 0.7417 | 0.7368 | 0.7393 | 0.9119 | 0.8854 |
| L2, C = 1.0 | 0.8163 | 0.7417 | 0.7368 | 0.7393 | 0.9119 | 0.8854 |
| L1, C = 1.0 | 0.8186 | 0.7434 | 0.7434 | 0.7434 | 0.9121 | 0.8858 |

The L1 model has the highest recorded test F1 and ROC-AUC among these configurations. Its saved confusion matrix contains:

| Actual / Predicted | No AD | AD |
| --- | ---: | ---: |
| No AD | 239 | 39 |
| AD | 39 | 113 |

The largest absolute coefficients in the unregularized model belong to **FunctionalAssessment, ADL, MemoryComplaints, BehavioralProblems, and MMSE**. Because predictors are standardized, the odds ratios describe a one-training-standard-deviation increase, holding other predictors constant. They describe model associations, not causal effects.

## 2. Concrete Compressive Strength Prediction

### Dataset

The notebook loads `concrete_data.csv`. Its saved output shows **1,030 rows and 9 columns**, no missing values, and **25 fully duplicated rows**. Removing those duplicates leaves **1,005 records**.

The eight predictors are cement, blast furnace slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate, and age in days.

The target column is named `Concrete compressive strength `, including a **trailing space**. Predictions are expressed in MPa.

### Workflow

1. Inspect the dataset and remove fully duplicated rows before splitting.
2. Separate the eight predictors from the strength target.
3. Create the train/test split and standardize predictors using training data only.
4. Fit Linear Regression, Ridge (`alpha=1.0`), and Lasso (`alpha=0.1`).
5. Compare training R², test MAE, test RMSE, and test R².
6. Inspect standardized coefficients and the intercept.
7. Plot predicted versus actual strength, residuals versus predictions, and residuals versus age.

### Results Recorded in the Notebook

| Model | Train R² | Test MAE | Test RMSE | Test R² |
| --- | ---: | ---: | ---: | ---: |
| Linear Regression | 0.6098 | 8.8960 | 11.1922 | 0.5801 |
| Ridge, alpha = 1.0 | 0.6098 | 8.8983 | 11.1942 | 0.5800 |
| Lasso, alpha = 0.1 | 0.6078 | 8.9232 | 11.2302 | 0.5773 |

Linear Regression has the lowest recorded test error among these configurations, although the differences are small. Cement, blast furnace slag, and age have the largest absolute standardized coefficients in that model.

## Installation

Use Google Colab or a local Jupyter environment. Install the required packages locally with:

```bash
python -m pip install pandas numpy matplotlib seaborn "scikit-learn>=1.2,<1.8" jupyterlab
```

The scikit-learn range supports the Logistic Regression API used in the notebook. This is a compatibility range, not a tested, locked environment.

## How to Run

1. Download the notebook and obtain both dataset CSV files from their original providers.
2. In Google Colab, upload the CSV files to `/content/`. The notebook currently expects:

   ```text
   /content/alzheimers_disease_data.csv
   /content/concrete_data.csv
   ```

3. For local execution, place the CSV files beside the notebook and change the two `pd.read_csv()` paths to:

   ```python
   pd.read_csv("alzheimers_disease_data.csv")
   pd.read_csv("concrete_data.csv")
   ```

4. Start JupyterLab if working locally:

   ```bash
   jupyter lab
   ```

5. Open the notebook and run all cells in order. The concrete section reuses variable names from the classification section; rerun a section from its beginning when returning to it.

Check that the concrete target column matches the spelling and trailing space used in the code. Dataset files, source links, and licensing information must be supplied separately.

## Notes and Limitations

- Results above are transcribed from saved notebook outputs; they were not independently reproduced for this README.
- An initial Logistic Regression cell trains on raw features and records convergence warnings. The later comparison trains separate models on preprocessed features. Inspect warnings when rerunning; the raw-feature models are not the models used in the comparison table.
- Regularization strengths are fixed rather than selected through cross-validation. Choosing a preferred model from test results makes that comparison exploratory; use training-only validation before a final held-out evaluation.
- Numeric-coded nominal categories should be reviewed before treating them as continuous predictors.
- Missing values are inspected but not imputed. The supplied outputs report none.
- Classification coefficient names currently assume there are no nonnumeric predictors; adapting the dataset may require using `preprocessor.get_feature_names_out()`.
- These educational results do not establish clinical suitability or generalization to other populations or concrete datasets.

## Author

Shatha Ajjaj

## Dataset Attribution

Add the original source, citation, and applicable license for each dataset before publishing or redistributing the data.
