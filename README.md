# House Prices – Machine Learning

A Machine Learning regression project based on Kaggle's **House Prices: Advanced Regression Techniques** competition. The objective is to predict residential property sale prices using the Ames Housing dataset.

The project follows a structured Machine Learning workflow, from exploratory data analysis to model optimization and ensemble learning, with an emphasis on sound preprocessing practices and model evaluation.

## Project Overview

- **Problem type:** Supervised Learning – Regression
- **Dataset:** Ames Housing
- **Target variable:** `SalePrice`
- **Training samples:** 1,460
- **Test samples:** 1,459
- **Evaluation metric:** Root Mean Squared Logarithmic Error (RMSLE)

Competition: [House Prices – Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

## Workflow

The project was developed in several stages:

### 1. Data Inspection

- Inspected dataset dimensions, data types and target variable.
- Checked missing values, duplicated records and potential inconsistencies.
- Investigated missing values representing the structural absence of property features, such as garages, basements and swimming pools.
- Reviewed the training and test datasets for consistency.

### 2. Exploratory Data Analysis (EDA)

- Analyzed the distribution of `SalePrice` and its relationship with the main predictors.
- Investigated numerical correlations and categorical feature distributions.
- Identified important predictors, including overall quality, living area, garage capacity and year built.
- Examined potential outliers and retained expensive properties when their prices were supported by property characteristics.
- Applied a logarithmic transformation to the target variable to reduce its positive skewness.

### 3. Data Preprocessing

Built reusable preprocessing pipelines using Scikit-learn.

The preprocessing workflow includes:

- Missing value imputation.
- Numerical and categorical feature handling.
- Categorical encoding.
- Numerical scaling when required by the model.
- Consistent transformations between training, validation and test datasets.

Preprocessing was integrated into the model pipelines to ensure transformations were fitted using training data.

### 4. Model Selection

Compared several regression algorithms using cross-validation and a validation holdout set.

Models evaluated included:

- Linear Regression and Ridge Regression.
- K-Nearest Neighbors.
- Random Forest and Extra Trees.
- Gradient Boosting.
- Support Vector Regression (SVR).

The main evaluation metric was RMSLE, with MAE and RMSE also monitored to better understand prediction errors.

SVR with a logarithmic target transformation emerged as the strongest initial candidate.

### 5. Feature Engineering

Created additional features based on property characteristics and domain knowledge:

| Feature | Description |
|---|---|
| `TotalLivingArea` | Combination of above-ground living area and total basement area. |
| `TotalBathrooms` | Total bathrooms, accounting for half bathrooms as 0.5. |
| `TotalFinishedBsmtSF` | Combined finished basement area. |
| `HouseAgeAtSale` | Property age at the time of sale. |
| `YearsSinceRemodel` | Years since the last remodeling. |
| `TotalOutdoorArea` | Combined deck, porch and outdoor living areas. |
| `QualityAreaInteraction` | Interaction between overall quality and total area. |

The engineered features were evaluated against the original feature set. They provided a small improvement in cross-validation and validation performance.

### 6. Model Optimization and Ensembling

Expanded model experimentation with:

- XGBoost.
- LightGBM.
- Further hyperparameter optimization using `RandomizedSearchCV`.
- Comparison of individual models and weighted ensemble configurations.

The final ensemble combines two complementary models:

- **SVR:** Higher weight.
- **Gradient Boosting:** Lower weight.

The selected weighting was **2:1 (SVR:Gradient Boosting)**. Predictions were combined in the original price scale.

### 7. Final Training and Kaggle Submission

The selected models were retrained using the complete training dataset (1,460 observations).

The ensemble generated predictions for the 1,459 records in the test dataset, producing the final Kaggle submission.

## Results


| Model | CV RMSLE | Validation RMSLE | Kaggle RMSLE |
|---|---:|---:|---:|
| Initial SVR | 0.117979 | 0.131341 | 0.11943 |
| Final SVR + Gradient Boosting Ensemble (2:1) | **0.113513** | **0.129481** | **0.11752** |

*Lower RMSLE values indicate better performance. Cross-validation and validation scores are internal model evaluation results; Kaggle RMSLE is the competition submission score.*

The final ensemble improved the Kaggle score from **0.11943 to 0.11752**, a relative reduction of approximately **1.60%** in RMSLE.

## Technologies

- **Python** – Main programming language.
- **Pandas and NumPy** – Data manipulation and numerical operations.
- **Scikit-learn** – Preprocessing pipelines, model training, cross-validation and hyperparameter optimization.
- **XGBoost and LightGBM** – Gradient boosting algorithms.
- **Jupyter Notebook** – Experimentation and documentation.
- **Matplotlib / Seaborn** – Exploratory data analysis and visualization.
- **Git and GitHub** – Version control.

## Project Structure

```text
house-prices-machine-learning/
│
├── data/                         # Dataset files
│
├── notebooks/                    # Analysis and experimentation
│   ├── 03_model_selection.ipynb
│   ├── 04_final_model_submission.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_model_optimization_ensembles.ipynb
│   └── 07_final_model_submission.ipynb
│
├── src/
│   ├── preprocessing.py           # Reusable preprocessing pipeline
│   └── feature_engineering.py     # Engineered feature creation
│
├── config/
│   └── phase7_best_params.json    # Optimized model parameters
│
├── submissions/
│   └── submission_phase8_ensemble_svr_gb.csv
│
└── README.md
```

*The structure above highlights the main project components. Add or adjust notebook and configuration filenames to match the final repository contents.*

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/AdriaFerOrtiz/house-prices-machine-learning.git
   cd house-prices-machine-learning
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install the main dependencies:

   ```bash
   pip install pandas numpy scikit-learn xgboost lightgbm jupyter matplotlib seaborn
   ```

4. Download the competition dataset from Kaggle and place the training and test CSV files in the location expected by the notebooks.

5. Open the notebooks in Jupyter and follow the project workflow.

## Key Learnings

This project provided practical experience with the complete Machine Learning regression workflow:

- Performing meaningful exploratory data analysis and identifying relevant patterns.
- Understanding and handling missing values, including structurally missing data.
- Building reusable preprocessing pipelines to avoid inconsistent transformations.
- Comparing different regression algorithms using appropriate evaluation metrics.
- Applying feature engineering based on domain knowledge.
- Optimizing model hyperparameters through cross-validation.
- Combining multiple models to improve predictive performance.
- Producing and evaluating submissions in a real Kaggle competition.

## Final Outcome

The project resulted in a weighted SVR and Gradient Boosting ensemble achieving a **0.11752 RMSLE Kaggle score**.

More importantly, it provided hands-on experience with the end-to-end Machine Learning development process, from raw data analysis to final model deployment in a competition submission.

---

**Author:** [AdriaFerOrtiz](https://github.com/AdriaFerOrtiz)
