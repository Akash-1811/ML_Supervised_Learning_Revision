# Logistic Regression

## Explanation
Logistic Regression is a classification algorithm that models the probability of a binary outcome using the logistic function:
p(y=1|x) = 1 / (1 + e^-(β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ))

Where:
- p(y=1|x) is the probability of the positive class
- x₁, x₂, ..., xₙ are features
- β₀, β₁, ..., βₙ are coefficients (weights)

## Implementation
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Create and train model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)

# Make predictions
y_pred = logreg.predict(X_test)
y_prob = logreg.predict_proba(X_test)

# Access model parameters
print("Coefficients:", logreg.coef_)
print("Intercept:", logreg.intercept_)
```

## Loss Function
Binary Cross-Entropy (Log Loss): Measures the performance of a classification model whose output is a probability value between 0 and 1.
Log Loss = -[y·log(p) + (1-y)·log(1-p)]

## Model Evaluation
1. Accuracy: Proportion of correct predictions
   - Accuracy = (TP + TN) / (TP + TN + FP + FN)

2. Precision: Proportion of positive identifications that were actually correct
   - Precision = TP / (TP + FP)

3. Recall (Sensitivity): Proportion of actual positives that were identified correctly
   - Recall = TP / (TP + FN)

4. F1 Score: Harmonic mean of precision and recall
   - F1 = 2 * (Precision * Recall) / (Precision + Recall)

5. ROC-AUC: Area under the Receiver Operating Characteristic curve
   - Measures the model's ability to discriminate between classes

## Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
    'class_weight': [None, 'balanced']
}

# Note: Not all solvers support all penalties
# For example, 'liblinear' doesn't support 'none', and only 'saga' supports 'elasticnet'

logreg = LogisticRegression(max_iter=1000)
grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
```

Key hyperparameters:
1. C: Inverse of regularization strength (smaller values = stronger regularization)
2. penalty: Type of regularization ('l1', 'l2', 'elasticnet', 'none')
3. solver: Algorithm for optimization problem
4. class_weight: Weights for classes (useful for imbalanced datasets)
5. max_iter: Maximum number of iterations for solver

## Best Practices
1. Scale features for better convergence
2. Handle class imbalance using class_weight parameter
3. Use regularization (L1 or L2) to prevent overfitting
4. Check for multicollinearity between features
5. Use cross-validation for hyperparameter tuning
6. Consider the threshold value for classification (default is 0.5)

## Interview Questions and Answers

1. **When would you use Logistic Regression vs. other classification algorithms?**
   - Answer: Logistic Regression is ideal when you need a probabilistic framework, interpretable results, or when computational resources are limited. It works well for linearly separable data and serves as a good baseline. It's also preferred when understanding feature importance is crucial.

2. **How does Logistic Regression differ from Linear Regression?**
   - Answer: Linear Regression predicts continuous values and uses the least squares method to minimize MSE. Logistic Regression predicts probabilities (0-1) for classification and uses maximum likelihood estimation to minimize log loss. The output of Logistic Regression is transformed using the sigmoid function.

3. **What is the sigmoid function and why is it used in Logistic Regression?**
   - Answer: The sigmoid function f(x) = 1/(1+e^(-x)) transforms any real-valued number into a value between 0 and 1. It's used in Logistic Regression to convert the linear combination of features into a probability value, making it suitable for binary classification.

4. **How do you handle multiclass classification with Logistic Regression?**
   - Answer: Two common approaches: (1) One-vs-Rest (OvR): Train n binary classifiers, one for each class vs. all others; (2) Multinomial Logistic Regression: Extends binary logistic regression to multiple classes using the softmax function instead of sigmoid.

5. **What are the assumptions of Logistic Regression?**
   - Answer: (1) Binary/categorical outcome variable; (2) Independence of observations; (3) Little or no multicollinearity among predictors; (4) Linear relationship between log-odds and features; (5) Large sample size; (6) No extreme outliers.

## Real-Life Implementation Challenge

When implementing Logistic Regression for credit risk assessment, I faced a significant class imbalance problem (few defaults compared to non-defaults), leading to a model that predicted "no default" for almost all cases despite high accuracy.

Solution:
1. Used class_weight='balanced' to give higher weight to the minority class
2. Implemented SMOTE (Synthetic Minority Over-sampling Technique) to generate synthetic samples of the minority class
3. Changed the evaluation metric from accuracy to F1-score and ROC-AUC
4. Adjusted the classification threshold based on the business cost of false positives vs. false negatives
5. Ensemble with other models to improve minority class detection

This approach significantly improved the model's ability to identify potential defaults while maintaining a reasonable false positive rate.

