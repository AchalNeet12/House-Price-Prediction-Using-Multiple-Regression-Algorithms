# House Price Prediction Using Multiple Regression Algorithms
---
## 📜 Project Description
- This project aims to predict house prices based on various features such as average area income, house age, number of rooms, bedrooms, and area population. Using regression algorithms, the system provides 
  predictions and evaluates the models based on metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²). A Flask-based web application is developed to allow users to input house 
  data and select a regression model for price prediction.
---
## 📝 Overview
- The project demonstrates a comprehensive machine learning workflow, from data preprocessing and model training to deployment using Flask. It evaluates 13 regression models and stores their results for 
  analysis. The web application serves as a user-friendly interface, enabling predictions using pre-trained models.
---
## 🔍 Dataset
- **Source:** The dataset used is `USA_Housing.csv.`
- **Features:**
 - `Avg. Area Income`: Average income of the area.
 - `Avg. Area House Age`: Average age of houses in the area.
 - `Avg. Area Number of Rooms`: Average number of rooms per house.
 - `Avg. Area Number of Bedrooms`: Average number of bedrooms per house.
 - `Area Population`: Population of the area.
 - `Price (Target Variable)`: Price of the house.
 - `Address`: Dropped as it doesn't contribute to numerical predictions.
- **Size:** ~5000 rows of data.
- **Target:** `Price` (dependent variable).
---
## 🤖 Technologies Used
- **Programming Language:** `Python`
- **Libraries:**
 - Data manipulation: `pandas`, `numpy`
 - Machine Learning: `scikit-learn`, `xgboost`, `lightgbm`
 - Evaluation Metrics: `mean_absolute_error`, `mean_squared_error`, `r2_score`
 - Web Framework: `Flask`
- **Tools:**
 - pickle for saving and loading models
 - HTML/CSS for frontend
 - CSV for data storage and logging
---
## ⚙ Algorithms Used
`1. **Linear Regression**`
  - Models the relationship between independent variables and a continuous target variable.
  - Simple and interpretable, but assumes a linear relationship.
    
`2. **Ridge Regression**`
  - Adds L2 regularization to linear regression to prevent overfitting.
  - Effective when there are many correlated features.
    
`3. **Lasso Regression**`
  - Uses L1 regularization to shrink some coefficients to zero, performing feature selection.
  -Helps in high-dimensional datasets by selecting only relevant features.

`4. **ElasticNet**`
  - Combines L1 and L2 regularization, balancing the benefits of Lasso and Ridge.
  - Works well when there are correlated features.
    
`5. **Polynomial Regression**`
  - Extends linear regression by adding polynomial terms to model non-linear relationships.
  - Prone to overfitting if too many polynomial features are added.
    
`6. **SGD Regressor**`
  - Uses iterative gradient descent for optimizing linear models.
  - Efficient for large datasets but requires careful parameter tuning.
    
`7. **ANN** (Artificial Neural Network)`
  - Learns complex, non-linear relationships through multiple layers of neurons.
  - Highly flexible, but computationally intensive.
    
`8. **Random Forest Regressor**`
  - An ensemble method that uses multiple decision trees and averages their predictions.
  - Robust to overfitting and handles complex relationships.
    
`9. **SVR** (Support Vector Regressor)`
  - Uses Support Vector Machine principles for regression with a margin of tolerance.
  - Effective for high-dimensional data but requires careful tuning.
    
`10. **LightGBM**`
  - A gradient boosting method optimized for speed and efficiency.
  - Handles large datasets well and supports categorical features.
    
`11. **XGBoost**`
  - A high-performance gradient boosting algorithm that minimizes overfitting.
  - Highly accurate, especially on structured/tabular data.
    
`12. **KNN Regressor** (K-Nearest Neighbors)`
  - Predicts the target by averaging the values of the k-nearest neighbors.
  - Simple but computationally expensive for large datasets.
    
`13. **Huber Regressor**`
  - Combines squared loss for small errors and absolute loss for large errors, making it robust to outliers.
  - Works well with noisy data.
---
## 📌 Data Preprocessing
 - **Load Data:** The dataset `USA_Housing.csv` was loaded into a Pandas DataFrame.
 - **Feature Selection:** The target variable `Price` was selected, and the `Address` feature was dropped since it is not useful for prediction.
 - **Missing Values:** Assumed that the dataset had no missing values, but any missing values would typically be handled by removing or imputing them.
 - **Data Splitting:** The data was split into training (80%) and testing (20%) sets using `train_test_split.`
 - **Feature Scaling:** Not explicitly done here, but would be important for algorithms like SVM or KNN that are sensitive to feature scales.
 - **Categorical Encoding:** No categorical variables were mentioned, but encoding would be necessary if there were any.
---
## 📈 Results
The models were evaluated using the following metrics:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R2 Score (coefficient of determination)
    
                                    **Model Evaluation Result**

  
|       Model         |        MAE               |         MSE           |            R2         |
|---------------------|--------------------------|-----------------------|-----------------------|                                 
| LinearRegression    |  82657.946058922         |  10549721686.159279   |  0.9146454505138069   |
| RobustRegression    |  199465.5595542928       |  61664910393.24936    |  0.5010881990728189   |
| RidgeRegression     |  82659.67244409776       |  10549745186.670172   |  0.91464526037841     |
| LassoRegression     |  82657.9466217223        |  10549717660.356375   |  0.9146454830853386   |
| ElasticNet          |  99126.80843102589       |  15081376466.55959    |  0.8779812271613102   |
| PolynomialRegression|  82894.44762396756       |  10604968581.895864   |  0.9141984648931063   |
| SGDRegressor        |  2.5551692062162115e+18  |  6.815138040199854e+36|  5.513918323273319e+25|
| ANN                 |  199295.5935104605       |  61472015131.63051    |  0.5026488552344974   |
| RandomForest        |  98664.59755352655       |  15131390689.874855   |  0.8775765774818237   |
| SVM                 |  282947.68758691323      |  123546565157.52672   |  0.0004227861838135283|
| LGBM                |  92133.9888284538        |  13097708114.507051   |  0.8940304769478327   |
| XGBoost             |  101565.19208841266      |  16138680641.877495   |  0.8694269046645823   |
| KNN                 |  198086.23684543537      |  60395811313.31432    |  0.5113560889227768   |

---
## 📊 Conclusion
  - The performance of each model varies based on the dataset. Models like Random Forest, XGBoost, and LightGBM generally perform well in handling complex relationships in the data.
  - Linear models like Linear Regression, Ridge, and Lasso may perform well with simple datasets but struggle with more complex patterns.
  - Ensemble models (e.g., Random Forest, XGBoost) and Neural Networks (ANN) tend to show better results for complex tasks, but may require more computational resources.
---
## 🎯 Future Implementation
  - **Hyperparameter Tuning:** Fine-tuning the model parameters using techniques like Grid Search or Randomized Search could improve model performance.
  - **Feature Engineering:** Creating new features based on existing data or using domain knowledge could improve model accuracy.
  - **Model Optimization:** Using advanced techniques like Stacking or Boosting could improve the predictive power of models.
  - **Deployment:** The models can be integrated into a production environment where users can input data and get price predictions.














