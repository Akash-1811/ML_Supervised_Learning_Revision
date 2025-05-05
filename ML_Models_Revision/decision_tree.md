# Decision Tree

## Explanation
Decision Trees are non-parametric supervised learning algorithms that create a model that predicts the target by learning simple decision rules from the data. They work by recursively splitting the data based on feature values to create a tree-like structure.

## Implementation
```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# Classification
dt_clf = DecisionTreeClassifier(max_depth=5, criterion='gini')
dt_clf.fit(X_train, y_train)
y_pred = dt_clf.predict(X_test)

# Regression
dt_reg = DecisionTreeRegressor(max_depth=5, criterion='squared_error')
dt_reg.fit(X_train, y_train)
y_pred = dt_reg.predict(X_test)

# Visualize the tree
from sklearn.tree import export_graphviz
import graphviz
dot_data = export_graphviz(dt_clf, feature_names=feature_names, 
                           class_names=class_names, filled=True)
graph = graphviz.Source(dot_data)
```

## Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

# For classification
param_grid = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy'],
    'max_features': [None, 'sqrt', 'log2']
}

dt_clf = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt_clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

# For regression
param_grid = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'max_features': [None, 'sqrt', 'log2']
}

dt_reg = DecisionTreeRegressor(random_state=42)
grid_search = GridSearchCV(dt_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
```

Key hyperparameters:
1. max_depth: Maximum depth of the tree
2. min_samples_split: Minimum samples required to split a node
3. min_samples_leaf: Minimum samples required at a leaf node
4. criterion: Function to measure the quality of a split
5. max_features: Number of features to consider for best split
6. class_weight: Weights for classes (for imbalanced datasets)

## Split Criteria
1. For Classification:
   - Gini impurity: Measures the probability of misclassifying a randomly chosen element
   - Entropy: Measures the information gain at each split

2. For Regression:
   - Mean Squared Error (MSE): Minimizes the variance in each node
   - Mean Absolute Error (MAE): Minimizes the absolute difference in each node

## Model Evaluation
1. For Classification:
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrix

2. For Regression:
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)

3. Feature importance: tree.feature_importances_

## Best Practices
1. Control tree complexity with parameters like max_depth, min_samples_split, min_samples_leaf
2. Prune the tree to avoid overfitting
3. Use cross-validation to find optimal hyperparameters
4. Consider ensemble methods (Random Forest, Gradient Boosting) for better performance
5. Balance the dataset for classification problems
6. Visualize the tree to understand decision rules
7. Use feature_importances_ to identify important features

## Interview Questions and Answers

1. **How does a Decision Tree make splits?**
   - Answer: Decision Trees make splits by selecting the feature and threshold that maximizes information gain (or minimizes impurity). For each potential split, the algorithm calculates how much it would reduce impurity (using metrics like Gini or entropy for classification, or variance reduction for regression), then chooses the best one. This process continues recursively until stopping criteria are met.

2. **What is the difference between Gini impurity and Entropy?**
   - Answer: Both measure node impurity, but with different calculations. Gini impurity measures the probability of misclassifying a randomly chosen element if it were randomly labeled according to the class distribution in the node. Entropy measures the average level of "information" or "surprise" in the node. Gini is computationally simpler, while entropy can be more sensitive to class imbalances. In practice, they often yield similar trees.

3. **How do you prevent overfitting in Decision Trees?**
   - Answer: Prevent overfitting by: (1) Limiting tree depth with max_depth; (2) Requiring minimum samples for splits (min_samples_split) and leaf nodes (min_samples_leaf); (3) Pruning branches that don't significantly improve performance; (4) Using ensemble methods like Random Forests; (5) Cross-validation to find optimal hyperparameters; and (6) Setting a minimum threshold for information gain to create
