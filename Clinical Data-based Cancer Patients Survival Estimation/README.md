# Project - Clinical Data-based Cancer Patients Survival Estimation

## Introduction
<p align="justify">
The project focuses on predicting survival outcomes of cancer patients using machine learning models. The models are trained on synthetic clinical data with an emphasis on lung, stomach, and liver cancer. Different machine learning algorithms are applied and evaluated to build predictive models to estimate survival probabilities. The project also highlights the role of regional disparities and lifestyle factors in cancer prognosis.

## Dataset
The datasets used in this project reflects actual cancer epidomology trends associated with patient lifestyle, charactorstics of cancer, treatment information, and survival status. The dataset contains 10000 unique patient records and 20 features. 
## Objective
The goal of this project is to assist in early prognosis, identify critical risk factors, and support data-driven decisions.
## Project Flowchart
![Capture1](https://github.com/user-attachments/assets/347cab9e-9b91-4cb5-9d60-caaefe84a1e1)
</p>

<p align="justify">
  
## Project Workflow
### Step 01: Data Exploration
<p align="justify">
Basic data cleaning and formatting are done by removing missing values and encoding categorical features. Explored feature distributions, correlations, and survival rates by cancer stage and tumor type to understand key patterns in the data.
</p>
  
### Step 02: Feature Engineering
<p align="justify">
Feature engineering is done to improve the dataset for prediction. A new column is created to show whether a patient had a surgery. Missing values in AlcoholUse, GeneticMutation, and Comorbidities are filled with "Unknown" or "None". MonthsSinceDiagonosis is calculated to show how much time has passed since the patient was diagonised. Unnecessary data columns are dropped to keep the data clean and focused.
</p>

### Step 03: Machine Learning Model Implementation
<p align="justify">
SMOTE (Synthetic Minority Oversampling Technique) is applied to address class imbalance in the survival data and ensure balanced representation of outcome. Three classification models: Decision Tree, Random forest, and Logistic Regression are implemented using default parameters. Each model is evaluated using 5-fold cross validation to estimate their accuracy.
</p>

### Step 04: Model Selection and Hyperparameter tuning
<p align="justify">
RandomizedSearchCV is used to improve the performance of Decision Tree, Random forest, and Logistic Regression models by tuning hyperparameters. Each model is trained with different combinations of parameters using 5-fold cross-validation to identify the best models. Then the models are compared based on their best cross-validation accuracy. The model with the highest accuracy is selected as the best perfroming model for predicting survival outcomes.
</p>

### Step 05: Model Evaluation
<p align="justify">
The best model is evaluated on the test set and achieved an accuracy 92.1% which indicates strong overall predictive ability. The confusion matrix shows very false positive and false negatives along with balanced performance. This model had high precision and recall for both 'Alive' and 'Deceased' classes according to consistent and reliable classification across both survival outcomes.
</p>

