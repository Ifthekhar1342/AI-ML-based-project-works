# Project - Machine Learning-Based Forecasting for Sustainable Transportation Markets

## Introduction
<p align="justify">
This project focuses on forecasting the sales of electric vehicles from 2025 to 2035 using machine learning and deep learning techniques. In recent years, a large growth in the automotive industry has brought a great challenge to plan the appropriate supply of necessary materials. The goal of this project is to provide data-driven decisions into the sustainable transportation markets as well as manufacturing and policy development across different geographic regions.
 </p>

## Dataset
<p align="justify">
The dataset contains sales data spanning the years from 2010 to 2035 including historical sales records (2010-2024) and forward-looking projections (2025-2035). Each entry includes 8 columns capturing key annual metrics (i.e. region, powertrain type, parameter). Preprocessing techniques are applied to the dataset to ensure optimal performance.   
</p>

## Project Workflow
### Step 1: Data Exploration
<p align="justify">
Analyzed the features to identify the detailed trend analysis that could help with predictive modeling for the future of electric mobility across different markets. 
</p>

### Step 2: Feature Engineering
<p align="justify">
Provide a clear view of annual market performance for each region using aggregated yearly unit sales, sales growth trajectory over time, and acceleration or deceleration patterns of market growth.
</p>

### Step 3: Data Visualization
<p align="justify">
Line plots, bar charts, and a heatmap are used to visualize global and regional sales trends, powertrain-wise growth, top-selling regions, and yearly distribution patterns.
</p>

### Step 4: Model Training and Evaluation
<p align="justify">

*  Data preprocessing: This dataset is filtered to include only relevant annual sales data. Total yearly sales are aggregated as well as features (x) and targets (y) are defined accordingly. The data is then split into training and testing sets using an 80/20 ratio to evaluate model performance effectively.
</p>
  
*   Algorithms used: To predict future sales trends based on historical data, the following machine learning regression algorithms are employed:
    1. #### Linear Regression
    2. #### Ridge Regression
    3. #### Lasso Regression
    4. #### ElasticNet Regression
    5. #### Decision Tree Regressor
    6. #### Random Forest Regressor
    7. #### Gradient Boosting Regressor
    8. #### AdaBoost Regressor
    9. #### Support Vector Regressor
    10. #### K-Nearest Neighbors Regressor
    11. #### Multi-Layer Perception Regressor
    
Each model is trained and evaluated using metrics like Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R-squared (R^2), and training time. The results are compiled in a comparison table to identify the best-performing algorithm.

![results](https://github.com/user-attachments/assets/57c3486a-454c-401a-9519-65d3da836269)
*  Result: A linear regression model performs well to forecast sales from 2025 to 2035 along with visualizing both historical and predicted trends for clear comparison.
</p>
<p align="justify">
  
### Step 5: Validation Loss Comparison
To evaluate the performance of different deep learning techniques, a validation loss comparison is conducted over 200 training epochs for three models:

*  The Dense Neural Network model indicates an efficient learning of overall patterns in the data and reaches low validation loss fast.
*  The GRU model initially performs slightly better than the LSTM and nearly matches the performance of the Dense Neural Network.
*  The LSTM model shows consistent improvement over time and has a slightly higher validation loss than the others.

### Step 6: Comparison Between Actual and Predicted Values
Based on the visual comparison, the Decision Tree model is the best model which gives the most actual predictions. On the other hand, the Support Vector Machine (SVM) model performs the worst due to its flat and inaccurate predictions.

## Conclusion
This project successfully explores and forecasts future trends in global sales using both machine learning and deep learning models. Identifying key patterns in historical data and generating accurate predictions for the 2025-2035 period through data preprocessing, visualization, and model evaluation can support strategic planning in technology development, infrastructure investment, and sustainable mobility transitions.

## Future Work

*  Experiment with hyperparameter tuning and ensemble methods to further enhance model performance.
*  Explore transformer-based models for improved time-series forecasting.
*  Build an interactive dashboard through a web app for real-time forecasting and visualization.
</p>
