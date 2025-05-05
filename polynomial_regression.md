# Polynomial Regression

## Explanation
Polynomial Regression extends Linear Regression by modeling the relationship between variables using polynomial functions:
y = β₀ + β₁x + β₂x² + β₃x³ + ... + βₙxⁿ + ε

Where:
- y is the target variable
- x is the feature
- β₀, β₁, ..., βₙ are coefficients
- ε is the error term

## Implementation
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Create polynomial regression model
degree = 3
polyreg = Pipeline([
    ('poly', PolynomialFeatures(degree=degree)),
    ('linear', LinearRegression())
])

# Train model
polyreg.fit(X_train, y_train)

# Make predictions
y_pred = polyreg.predict(X_test)

# Access coefficients
coefficients = polyreg.named_steps['linear'].coef_
intercept = polyreg.named_steps['linear'].intercept_
```

## Loss Function
Mean Squared Error (MSE): Same as Linear Regression
MSE = (1/n) Σ(y_actual - y_predicted)²

## Model Evaluation
1. R-squared (coefficient of determination)
   - R² = 1 - (sum of squared residuals / total sum of squares)
   - Range: 0 to 1 (higher is better)

2. Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
   - Lower is better

3. Cross-validation scores to check for overfitting

## Hyperparameter Tuning
The main hyperparameter in Polynomial Regression is the degree of the polynomial:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

# Create pipeline with polynomial features and regularized regression
polynomial_pipeline = Pipeline([
    ('poly', PolynomialFeatures()),
    ('ridge', Ridge())
])

# Define parameter grid
param_grid = {
    'poly__degree': [1, 2, 3, 4, 5],
    'ridge__alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
}

# Perform grid search
grid_search = GridSearchCV(
    polynomial_pipeline,
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error'
)
grid_search.fit(X_train, y_train)

# Get best parameters
best_degree = grid_search.best_params_['poly__degree']
best_alpha = grid_search.best_params_['ridge__alpha']
```

Key hyperparameters:
1. degree: Degree of the polynomial (higher = more complex model)
2. include_bias: Whether to include a bias column (default is True)
3. interaction_only: If True, only interaction features are produced
4. Regularization parameters (when combined with Ridge or Lasso)

## Best Practices
1. Choose an appropriate polynomial degree to avoid overfitting
2. Scale features before creating polynomial features
3. Use regularization for higher-degree polynomials
4. Visualize the fitted curve to check for overfitting
5. Use cross-validation to select the optimal degree
6. Consider feature selection for higher-degree models

## Interview Questions and Answers

1. **When would you use Polynomial Regression instead of Linear Regression?**
   - Answer: Use Polynomial Regression when the relationship between variables is non-linear and follows a polynomial pattern. It's useful when a scatter plot of data shows curves, parabolas, or more complex patterns that a straight line cannot capture.

2. **How do you determine the optimal degree for Polynomial Regression?**
   - Answer: Use cross-validation to find the degree that minimizes validation error. Plot training and validation errors against polynomial degrees - the optimal degree is typically where validation error is minimized before it starts increasing again (indicating overfitting).

3. **What are the risks of using a high-degree polynomial?**
   - Answer: High-degree polynomials can lead to overfitting, where the model captures noise in the training data rather than the underlying pattern. This results in poor generalization to new data, extreme coefficient values, and high variance in predictions.

4. **How does Polynomial Regression handle multivariate data?**
   - Answer: For multivariate data, Polynomial Regression creates interaction terms between features in addition to polynomial terms for each feature. This significantly increases the number of features (combinatorial explosion), making regularization and feature selection important.

5. **What is the relationship between Polynomial Regression and Linear Regression?**
   - Answer: Polynomial Regression is a special case of multiple linear regression where the original features are transformed into polynomial features. The model is still linear with respect to the parameters (coefficients), but non-linear with respect to the original features.

## Real-Life Implementation Challenge

When implementing Polynomial Regression for predicting energy consumption based on temperature, I encountered severe overfitting with higher-degree polynomials that produced unrealistic predictions outside the training data range.

Solution:
1. Implemented regularization (Ridge regression) to control coefficient magnitudes
2. Used cross-validation to find the optimal polynomial degree (3 was sufficient)
3. Applied feature scaling before creating polynomial features to improve numerical stability
4. Limited predictions to the range of observed data or used extrapolation warnings
5. Incorporated domain knowledge to constrain the model behavior at extreme values

This approach resulted in a model that captured the non-linear relationship between temperature and energy consumption while maintaining reasonable predictions across the entire range of possible temperatures.