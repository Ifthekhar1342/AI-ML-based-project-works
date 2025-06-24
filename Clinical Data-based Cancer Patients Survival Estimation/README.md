# Project - Clinical Data-based Cancer Patients Survival Estimation

## Introduction
<p align="justify">
The project focuses on predicting survival outcomes of cancer patients using machine learning models. The models are trained on synthetic clinical data with an emphasis on lung, stomach, and liver cancer. Different machine learning algorithms are applied and evaluated to build predictive models to estimate survival probabilities. The project also highlights the role of regional disparities and lifestyle factors in cancer prognosis.

## Dataset
The datasets used in this project reflect actual cancer epidemiology trends associated with patient lifestyle, characteristics of cancer, treatment information, and survival status. The dataset contains 10000 unique patient records and 20 features. 
## Objective
The goal of this project is to facilitate in early prognosis, identify critical risk factors, and support data-driven decision-making.
## Project Flowchart
![Capture1](https://github.com/user-attachments/assets/347cab9e-9b91-4cb5-9d60-caaefe84a1e1)
</p>

<p align="justify">
  
## Project Workflow
### Step 01: Data Exploration
<p align="justify">
Basic data cleaning and formatting are done by removing missing values and encoding categorical features. Exploring feature distributions, correlations, and survival rates by cancer stage and tumor type to understand key patterns in the data.
</p>
  
### Step 02: Feature Engineering
<p align="justify">
Feature engineering is done to improve the dataset for prediction. A new column is created to indicate whether a patient has undergone surgery. Missing values in AlcoholUse, GeneticMutation, and Comorbidities are filled with "Unknown" or "None". MonthsSinceDiagonosis is calculated to show how much time has passed since the patient was diagnosed. Unnecessary data columns are dropped to keep the data clean and focused.
</p>

### Step 03: Machine Learning Model Implementation
<p align="justify">
SMOTE (Synthetic Minority Oversampling Technique) is applied to address the class imbalance in the survival data and ensure a balanced representation of the outcome. Three classification models: Decision Tree, Random Forest, and Logistic Regression are implemented using default parameters. Each model is evaluated using 5-fold cross-validation to estimate their accuracy.
</p>

### Step 04: Model Selection and Hyperparameter tuning
<p align="justify">
RandomizedSearchCV is used to improve the performance of the Decision Tree, Random Forest, and Logistic Regression model by tuning hyperparameters. Each model is trained with different combinations of parameters using 5-fold cross-validation to identify the best models. Then the models are compared based on their best cross-validation accuracy. The model with the highest accuracy is selected as the best-performing model for predicting survival outcomes.
</p>

### Step 05: Model Evaluation
<p align="justify">
The best model is the Random Forest classifier which is evaluated on the test set and achieved an accuracy of 92.1% which indicates strong overall predictive ability. The confusion matrix shows very false positives and false negatives along with balanced performance. This model had high precision and recall with an F1 score of 92% for both 'Alive' and 'Deceased' classes according to consistent and reliable classification across both survival outcomes.
</p>

## Conclusion
<p align="justify">
The project work demonstrates the use of machine learning algorithms to predict cancer patient survival based on clinical, treatment, and lifestyle data. The results indicate that it can be effectively potential for real-world applications in healthcare analytics and prognosis support.
</p>

## Future Work
<p align="justify">

-  Incorporate additional clinical metrics such as lab results and imaging data to improve prediction accuracy.
-  Use time series or sequential models to better capture the progression of diseases over time.
-  Develop a simple user interface to allow clinicians or researchers to input data and get survival predictions.
