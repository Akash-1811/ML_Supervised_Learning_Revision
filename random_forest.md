# Random Forest

## Explanation
Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode (classification) or mean (regression) of individual trees. It uses two key techniques:
1. Bagging (Bootstrap Aggregating): Training each tree on a random subset of data
2. Feature randomness: Considering only a random subset of features at each split

## Implementation
```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# Classification
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)

# Regression
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)

# Feature importance
importances = rf_clf.feature_importances_
```

## Key Parameters
1. n_estimators: Number of trees in the forest
2. max_depth: Maximum depth of each tree
3. max_features: Maximum number of features to consider for each split
4. min_samples_split: Minimum samples required to split a node
5. min_samples_leaf: Minimum samples required at a leaf node
6. bootstrap: Whether to use bootstrap samples

## Model Evaluation
1. For Classification:
   - Accuracy, Precision, Recall, F1-score
   - ROC-AUC

2. For Regression:
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)

3. Feature importance: rf.feature_importances_
4. Out-of-bag (OOB) error: rf.oob_score_

## Best Practices
1. Use a sufficient number of trees (n_estimators)
2. Tune max_depth to control overfitting
3. Consider max_features (rule of thumb: sqrt(n_features) for classification, n_features/3 for regression)
4. Use feature_importances_ for feature selection
5. Enable oob_score=True to estimate model performance without a separate validation set
6. Parallelize computation with n_jobs parameter
7. Balance datasets for classification problems

## Interview Questions
1. How does Random Forest differ from a single Decision Tree?
2. What is bagging and how does it help in Random Forest?
3. How does Random Forest prevent overfitting?
4. What is the Out-of-Bag (OOB) error and how is it calculated?
5. How does Random Forest handle feature importance?
6. What are the advantages and disadvantages of Random Forest?
7. How does Random Forest handle missing values and categorical features?
8. When would you use Random Forest over Gradient Boosting methods?
9. How does increasing the number of trees affect bias, variance, and computational cost?
10. What is the difference between Random Forest and Extra Trees?