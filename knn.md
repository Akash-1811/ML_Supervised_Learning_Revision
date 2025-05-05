# K-Nearest Neighbors (KNN)

## Explanation
KNN is a non-parametric, instance-based learning algorithm that classifies data points based on the majority class of their k nearest neighbors or predicts values based on the average of k nearest neighbors.

## Implementation
```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# Classification
knn_clf = KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn_clf.fit(X_train, y_train)
y_pred = knn_clf.predict(X_test)

# Regression
knn_reg = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn_reg.fit(X_train, y_train)
y_pred = knn_reg.predict(X_test)
```

## Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

# For classification
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2]  # p=1 for manhattan, p=2 for euclidean
}

knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_

# For regression
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski'],
    'p': [1, 2]
}

knn_reg = KNeighborsRegressor()
grid_search = GridSearchCV(knn_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
```

Key hyperparameters:
1. n_neighbors: Number of neighbors to consider
2. weights: Weight function ('uniform' or 'distance')
3. metric: Distance metric to use
4. p: Power parameter for Minkowski metric
5. algorithm: Algorithm used to compute nearest neighbors
6. leaf_size: Leaf size for tree-based algorithms

## Distance Metrics
1. Euclidean distance (default): sqrt(Σ(xᵢ - yᵢ)²)
2. Manhattan distance: Σ|xᵢ - yᵢ|
3. Minkowski distance: (Σ|xᵢ - yᵢ|ᵖ)^(1/p)
4. Hamming distance: For categorical features

## Model Evaluation
1. For Classification:
   - Accuracy, Precision, Recall, F1-score
   - Confusion matrix
   - ROC-AUC

2. For Regression:
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - Mean Absolute Error (MAE)

## Best Practices
1. Scale features as KNN is sensitive to the scale of input features
2. Choose k carefully (typically odd to avoid ties in binary classification)
3. Use cross-validation to find optimal k
4. Consider using weighted voting (weights='distance')
5. Use dimensionality reduction for high-dimensional data
6. Be aware of the curse of dimensionality
7. Consider approximate nearest neighbors for large datasets

## Interview Questions and Answers

1. **How do you choose the optimal value of k in KNN?**
   - Answer: The optimal k is typically chosen using cross-validation to find the value that minimizes error on validation data. Generally, larger k values reduce the effect of noise but make boundaries between classes less distinct. Smaller k values create more complex decision boundaries but may overfit. For binary classification, odd values of k are preferred to avoid ties.

2. **What are the advantages and disadvantages of KNN?**
   - Answer: Advantages: Simple to understand and implement, no training phase, naturally handles multi-class problems, and makes no assumptions about data distribution. Disadvantages: Computationally expensive for large datasets, sensitive to irrelevant features and feature scaling, requires feature engineering, and storage of the entire training dataset.

3. **How does KNN handle categorical features?**
   - Answer: KNN works with numerical distances, so categorical features need to be converted. Options include: one-hot encoding for nominal features, label encoding for ordinal features, or using specialized distance metrics like Hamming distance. For mixed data types, you can use combination metrics or convert everything to a common representation.
   
   When dealing with nominal features that have many unique values, one-hot encoding can lead to the curse of dimensionality. Alternative approaches include:
   
   1. **Feature Hashing/Hash Encoding**: Maps high-cardinality features to a fixed-dimensional space using hash functions
   ```python
   from sklearn.feature_extraction import FeatureHasher
   hasher = FeatureHasher(n_features=20, input_type='string')
   hashed_features = hasher.transform(categorical_column)
   ```
   
   2. **Target Encoding**: Replace categories with their target mean value
   ```python
   target_means = df.groupby('categorical_column')['target'].mean()
   df['encoded_feature'] = df['categorical_column'].map(target_means)
   ```
   
   3. **Binary Encoding**: Represents each category as binary code, requiring only log2(n) features
   ```python
   from category_encoders import BinaryEncoder
   encoder = BinaryEncoder(cols=['categorical_column'])
   binary_encoded = encoder.fit_transform(df['categorical_column'])
   ```
   
   4. **Dimensionality Reduction**: Apply PCA or t-SNE after one-hot encoding
   ```python
   from sklearn.decomposition import PCA
   pca = PCA(n_components=10)
   reduced_features = pca.fit_transform(one_hot_encoded_features)
   ```
   
   5. **Similarity-based approaches**: Use specialized metrics like HEOM (Heterogeneous Euclidean-Overlap Metric) that handle mixed data types directly

4. **What is the effect of the curse of dimensionality on KNN?**
   - Answer: As dimensionality increases, the available data becomes sparse in the feature space. This makes distance measurements less meaningful as points tend to be equidistant from each other in high dimensions. KNN's performance degrades significantly because the concept of "nearest" becomes less relevant, leading to poor classification or regression results.

5. **How does the choice of distance metric affect KNN performance?**
   - Answer: The distance metric defines how similarity between points is calculated. Euclidean distance works well for continuous features in low dimensions. Manhattan distance is less sensitive to outliers. Specialized metrics like cosine similarity work better for text data. The optimal metric depends on the data characteristics and should be selected through cross-validation.

## Real-Life Implementation Challenge

When implementing KNN for a customer segmentation project, I faced significant performance issues with a large dataset (millions of customers with dozens of features), making the algorithm impractically slow for both training and prediction.

Solution:
1. Applied dimensionality reduction using PCA to reduce features from 50 to 10 while preserving 85% of variance
2. Implemented approximate nearest neighbors using ball tree and KD-tree data structures
3. Used locality-sensitive hashing (LSH) for initial filtering of candidate neighbors
4. Created a hybrid approach that used K-means clustering first to identify the general neighborhood, then applied KNN within that cluster
5. Parallelized the computation across multiple cores

This approach reduced prediction time from minutes to milliseconds per customer while maintaining 95% of the original accuracy, making it feasible to deploy the model in a real-time recommendation system.

