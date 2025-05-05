# Linear Regression

## Explanation
Linear Regression models the relationship between variables by fitting a linear equation:
y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε

Where:
- y is the target variable
- x₁, x₂, ..., xₙ are features
- β₀, β₁, ..., βₙ are coefficients (weights)
- ε is the error term

## Implementation
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create and train model
regr = LinearRegression()
regr.fit(X_train, y_train)

# Make predictions
y_pred = regr.predict(X_test)

# Access model parameters
print("Coefficients:", regr.coef_)
print("Intercept:", regr.intercept_)
```

## Loss Function
Mean Squared Error (MSE): Measures the average squared difference between predicted and actual values.
MSE = (1/n) Σ(y_actual - y_predicted)²

## Model Evaluation
1. R-squared (coefficient of determination): Proportion of variance explained by the model
   - R² = 1 - (sum of squared residuals / total sum of squares)
   - Range: 0 to 1 (higher is better)

2. Mean Squared Error (MSE): Average squared difference between predictions and actual values
   - Lower is better

3. Root Mean Squared Error (RMSE): Square root of MSE, in same units as target variable

## Hyperparameter Tuning
Linear Regression has few hyperparameters to tune, but you can optimize:

1. For regularized versions (Ridge, Lasso):
```python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
ridge = Ridge()
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_alpha = grid_search.best_params_['alpha']
```

2. For polynomial features, tune the degree:
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

degrees = [1, 2, 3, 4, 5]
best_rmse = float('inf')
best_degree = 1

for degree in degrees:
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    poly_model.fit(X_train, y_train)
    y_pred = poly_model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    if rmse < best_rmse:
        best_rmse = rmse
        best_degree = degree
```

## Best Practices
1. Check for multicollinearity between features
2. Scale features if they're on different scales
3. Check residuals for normality and homoscedasticity
4. Consider regularization (Ridge, Lasso) for high-dimensional data
5. Split data into training and testing sets
6. Use cross-validation for more robust evaluation

## Interview Questions and Answers

1. **When would you use Linear Regression vs. other algorithms?**
   - Answer: Linear Regression is best when there's a linear relationship between features and target. It's simple, interpretable, and computationally efficient. Use it for baseline models, when interpretability is important, or when you have limited computational resources.

2. **How do you handle categorical variables in Linear Regression?**
   - Answer: Convert them to numerical values using one-hot encoding, dummy encoding, or label encoding. One-hot encoding creates binary columns for each category, dummy encoding does the same but drops one category to avoid multicollinearity, and label encoding assigns a unique number to each category.

3. **What are the assumptions of Linear Regression?**
   - Answer: (1) Linearity: relationship between features and target is linear; (2) Independence: observations are independent; (3) Homoscedasticity: constant variance of errors; (4) Normality: errors are normally distributed; (5) No multicollinearity: predictors are not highly correlated.

4. **Explain the difference between R-squared and adjusted R-squared.**
   - Answer: R-squared measures the proportion of variance explained by the model (0 to 1). Adjusted R-squared penalizes adding unnecessary variables by adjusting for the number of predictors. It increases only if new variables improve the model more than would be expected by chance.

5. **How do you detect and handle outliers in Linear Regression?**
   - Answer: Detect outliers using visualization (scatter plots, box plots), statistical methods (z-scores, IQR), or model-based methods (Cook's distance, DFFITS). Handle them by removing, transforming (log, square root), winsorizing, or using robust regression methods less sensitive to outliers.

## Real-Life Implementation Challenge

One common challenge I faced was dealing with multicollinearity in a housing price prediction model. Several features like square footage, number of rooms, and number of bathrooms were highly correlated, causing unstable coefficient estimates and making interpretation difficult.

Solution: I used a combination of:
1. Variance Inflation Factor (VIF) analysis to identify problematic features
2. Feature selection to remove redundant variables
3. Ridge regression to stabilize coefficients
4. Principal Component Analysis (PCA) to create uncorrelated composite features

This approach improved model stability and prediction accuracy while maintaining interpretability of the key factors affecting housing prices.

