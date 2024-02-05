"""main.py: Starter file for assignment on Decision Trees, SVM, and K-NN """

__author__ = "Bryan Tuck"
__version__ = "1.0.0"
__copyright__ = "All rights reserved.  This software  \
                should not be distributed, reproduced, or shared online, without the permission of the author."

# Data Manipulation and Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


# Model Evaluation and Hyperparameter Tuning
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, GridSearchCV 
from sklearn.metrics import accuracy_score, precision_score, recall_score

__author__ = "Josh Aneke"
__version__ = "1.1.0"

'''
Github Username: JoshAneke
PSID:1828214
'''

# Reading of training and testing files
df_train = pd.read_json('emotion_train.json', lines=True)
df_test = pd.read_json('emotion_test.json', lines=True)

# Task 1: Decision Trees

''' Task 1A: Build Decision Tree Models with Varying Depths '''
# Using all attributes, train Decision Tree models with maximum depths of 3, 7, 11, and 15.
X_train = df_train.loc[:, 'enc0':'enc127']  # Feature columns
y_train = df_train['label']  # Target column

X_test = df_test.loc[:, 'enc0':'enc127']  # Feature columns
y_test = df_test['label']  # Target column

# Define the maximum depths you want to use
max_depths = [3, 7, 11, 15]

# Loop through different maximum depths
for max_depth in max_depths:
    # Create and train the Decision Tree classifier
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_train, y_train)
    
    # Make predictions on the testing data
    y_pred = clf.predict(X_test)
''' Task 1B: 5-Fold Cross-Validation for Decision Trees '''
# Perform 5-fold cross-validation on each Decision Tree model. Compute and store the mean accuracy, precision, and recall for each depth. Generate the table.
results_df = pd.DataFrame(columns=['Max Depth', 'Accuracy', 'Precision', 'Recall'])

# Loop through different maximum depths
results = []

# Loop through different maximum depths
for max_depth in max_depths:
    # Create a Decision Tree classifier with the specified max_depth
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_train, y_train)

    
    # Perform 5-fold cross-validation for accuracy
    accuracy_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    
    # Perform 5-fold cross-validation for precision and recall
    y_pred_cv = cross_val_predict(clf, X_train, y_train, cv=5)
    precision_scores = precision_score(y_train, y_pred_cv, average='weighted')
    recall_scores = recall_score(y_train, y_pred_cv, average='weighted')
    
    # Compute mean accuracy, precision, and recall
    mean_accuracy = accuracy_scores.mean()
    mean_precision = precision_scores
    mean_recall = recall_scores
    
    # Append the results to the list
    results.append([max_depth, mean_accuracy, mean_precision, mean_recall])

# Create a DataFrame from the results
results_df = pd.DataFrame(results, columns=['Max Depth', 'Accuracy', 'Precision', 'Recall'])

# Print the results table
print(results_df)

''' Task 1C: Interpret Decision Tree Depths '''
# Provide explanations on how the tree depth impacts overfitting and underfitting.
#!!!Overfitting occurs when a decision tree model becomes too complex, capturing noise and random fluctuations in the training data rather than the underlying patterns. This leads to excellent performance on the training data but poor generalization to unseen data. Increasing the tree depth allows the model to create more complex and detailed decision boundaries, potentially fitting the training data very closely. However, this also makes the model more prone to overfitting, as it may capture noise in the data. When the tree depth is very high, each leaf node may contain only a few data points, and the model effectively memorizes the training data, leading to poor generalization.")
#!!!Underfitting occurs when a decision tree model is too simple, unable to capture the underlying patterns in the data. This results in poor performance on both the training data and unseen data.
#!!!Decreasing the tree depth makes the model simpler, as it can only create coarse decision boundaries. When the tree depth is too shallow, the model may not have enough capacity to capture complex relationships in the data, leading to underfitting.")

''' Task 1D: Interpret Decision Tree Metrics '''
# Explain the significance of differences in accuracy, precision, and recall if any notable differences exist.
#!!!The accuracy values vary across different max depths, and the differences are noticeable. The highest accuracy is achieved at max depth 7 (0.6018), while the lowest is at max depth 3 (0.5683). These differences in accuracy are significant and indicate that the choice of max depth significantly impacts the overall correctness of predictions.
#!!!Recall values show variation across different max depths. The highest recall is achieved at max depth 7 (0.6015), while the lowest is at max depth 3 (0.5683). These differences in recall are significant and indicate that the choice of max depth affects the model's ability to capture positive instances.

# Task 2: K-NN

''' Task 2A: Build k-NN Models with Varying Neighbors '''
# Train K-NN models using 3, 9, 17, and 25 as the numbers of neighbors.
k_values = [3, 9, 17, 25]
results_list = []
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

for k in k_values:
    # Create a k-NN classifier with the current k value
    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train_std, y_train)
    y_pred = knn.predict(X_test_std)

''' Task 2B: 5-Fold Cross-Validation for K-NN '''
# Perform 5-fold cross-validation on each K-NN model. Compute and store the mean accuracy, precision, and recall for each neighbor size. Generate the table.
results_df2 = pd.DataFrame(columns=['Neighbors', 'Accuracy', 'Precision', 'Recall'])

# Loop through different maximum depths
results2 = []

for k in k_values:
    # Create a k-NN classifier with the current k value
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_std, y_train)

    accuracy_scores2 = cross_val_score(knn, X_train_std, y_train, cv=5, scoring='accuracy')
    
    # Perform 5-fold cross-validation for precision and recall
    y_pred_cv2 = cross_val_predict(knn, X_train_std, y_train, cv=5)
    precision_scores2 = precision_score(y_train, y_pred_cv2, average='weighted')
    recall_scores2 = recall_score(y_train, y_pred_cv2, average='weighted')
    
    # Compute mean accuracy, precision, and recall
    mean_accuracy2 = accuracy_scores2.mean()
    mean_precision2 = precision_scores2
    mean_recall2 = recall_scores2
    
    # Append the results to the list
    results2.append([k, mean_accuracy2, mean_precision2, mean_recall2])

# Create a DataFrame from the results
results_df2 = pd.DataFrame(results2, columns=['Neighbors', 'Accuracy', 'Precision', 'Recall'])

# Print the results table
print(results_df2)

''' Task 2C: Interpret K-NN Neighbor Sizes '''
# Discuss how the number of neighbors impacts overfitting and underfitting.
#!!!When neighbors is small, such as neighbors=1, the model tends to be highly sensitive to noise and fluctuations in the training data. The decision boundary is highly flexible and may follow individual data points closely, which can lead to overfitting. The model may capture noise in the training data and not generalize well to new, unseen data. Low neighbor models have low bias because they can fit the training data very closely. However, they have high variance because they are sensitive to variations in the training data.
#!!!When neighbors is large, such as neighbors=100, the model tends to be less sensitive to noise and individual data points. The decision boundary becomes smoother and may not fit the training data as closely. The model may underfit the training data, failing to capture important patterns. High neighbors models have high bias because they make more general assumptions about the data. They have low variance because they are less sensitive to small fluctuations in the training data.
''' Task 2D: Interpret K-NN Metrics '''
# Explain any significant differences in accuracy, precision, and recall among the different neighbor sizes if any notable differences exist..
#!!! Precision shows an increasing trend as k increases from 3 to 25. The highest precision is achieved at neighbors=25 (0.6433), while the lowest is at neoghbors=3 (0.5981). The differences in precision are relatively moderate but show an improvement with larger k values, signifying as more neighbors usually has more acuurate predictions then ones with less neighbors.

# Task 3: SVM

''' Task 3A: Build SVM Models with Varying Kernel Functions '''
# Train SVM models using linear, polynomial, rbf, and sigmoid kernels. Store each trained model.
svm_error_margin=1000
Models = ['linear', 'poly', 'rbf', 'sigmoid']

for Model in Models:
    svm_linear = SVC(kernel=Model)

    svm_linear.fit(X_train_std, y_train)
    y_pred = svm_linear.predict(X_test_std)

''' Task 3B: 5-Fold Cross-Validation for SVM '''
# Perform 5-fold cross-validation on each SVM model. Compute and store the mean accuracy, precision, and recall for each kernel. Generate the table.
results_df3 = pd.DataFrame(columns=['Kernel', 'Accuracy', 'Precision', 'Recall'])

results3 = []

for Model in Models:
    # Create an SVM classifier with the current kernel
    svm = SVC(kernel=Model)
    svm.fit(X_train_std, y_train)

    accuracy_scores3 = cross_val_score(svm, X_train_std, y_train, cv=5, scoring='accuracy')
    
    # Perform 5-fold cross-validation for precision and recall
    y_pred_cv3 = cross_val_predict(svm, X_train_std, y_train, cv=5)
    precision_scores3 = precision_score(y_train, y_pred_cv3, average='weighted')
    recall_scores3 = recall_score(y_train, y_pred_cv3, average='weighted')
    
    # Compute mean accuracy, precision, and recall
    mean_accuracy3 = accuracy_scores3.mean()
    mean_precision3 = precision_scores3
    mean_recall3 = recall_scores3
    
    # Append the results to the list
    results3.append([Model, mean_accuracy3, mean_precision3, mean_recall3])

# Create a DataFrame from the results
results_df3 = pd.DataFrame(results3, columns=['Kernel', 'Accuracy', 'Precision', 'Recall'])

# Print the results table
print(results_df3)

''' Task 3C: Interpret SVM Kernel Functions '''
# Discuss the impact of different kernel functions on the performance of the SVM models.
#!!!The linear kernel performs quite well on the dataset, with accuracy, precision, and recall all above 0.7. It is effective when the data has a clear linear separation. In this case, the data is reasonably well-separated by linear boundaries.
#!!!The polynomial kernel shows the lowest performance among the tested kernels, with accuracy, precision, and recall around 0.55. #!!!The low performance suggests that the data might have complex, non-linear relationships that the polynomial kernel struggles to capture effectively.
#!!!The RBF kernel performs well, with accuracy, precision, and recall around 0.716. The RBF kernel is known for its adaptability to various data distributions, making it a suitable choice when the data has non-linear patterns.
#!!!The sigmoid kernel also performs well, with accuracy, precision, and recall around 0.718. While less common than other kernels, it shows competitive results in this case. It may work effectively when the relationship between features and the target variable has logistic characteristics.

''' Task 3D: Interpret SVM Metrics '''
# Explain any significant differences in accuracy, precision, and recall among the different kernels.
#!!!In the data, Linear, rbf, and sigmoid all had slightly similar points. On the other hand, we see that the polynomial kernal does not do as good as the rest of the kernels. The polynomial kernal is used to create non-linear decision boundaries and better handle data that is not linearly separable in the original feature space. Since the poly kernal clearly does not have high marks such as the other kernals, we can assume that the data set has some type of linear approach

# Task 4: Interpretation and Comparison

''' Task 4: Interpret Tables and Model Comparison '''
# Compare the performance metrics (accuracy, precision, and recall) of the Decision Tree, K-NN, and SVM models. Discuss which model performs better and why.
#!!!Taking into account all of the data that is given to us, the model that performs better is the SVM model, specifically the linear, rbf, and sigmoid. These catagories had higher accuracies, precisions, and recalls for many reasons, including: 
#!!!Effective Handling of Data: High-dimensional data, which is typical in many real-world applications, works well with SVMs. They are able to identify the best hyperplanes for dividing data in high-dimensional spaces.
#!!!Resilient to Overfitting: SVMs can leverage the kernel trick to transform data into higher-dimensional spaces, making them capable of capturing complex non-linear relationships between features. This is particularly valuable when data is not linearly separable, as is often the case in real-world scenarios.
#!!!Efficieny: SVMs are very efficient with organizing data, especially with high demand.
''' Recommendations for Model Improvement '''
# Provide suggestions on how you might improve each model’s performance.
#!!!To improve the decision trees we could change the maximum depth to enhance performance
#!!!To improve K-NN, the best chouce would be to chose an appropriate number of neighbors that would better fit the set.
# !!!To improve the SVM, instead of using the poly kernel, we could try to experiment with other functions to see which would be a better fit. 
''' Conclusion '''
# Summarize the key findings and insights from this assignment.
#!!!I learned many things from this assignment. Firstly, I learned more in depth about the three types of models, K-NN, SVM, and decision trees. 
#!!!For the decision trees, I learned about overfitting. If you have to much of the parameter, depth in this case, then you will overfit, and vice versa for underfitting. Based on the data that we got, we could see that the depth of 7 yeilds the best statistics
#!!!For the K-NN models, we were given various neighbors. We saw something similar to the depth trees with the overfitting and underfitting. If you don't put in the right amunt or an amount that is close, you will overfit or underfit, which is what happened to our data. We see tha that 17 is easily the best neighbor, since it has the best statistics.
#!!!For the SVM models, I learned about 4 different types of kernels, which were linear, polynomail, rbf, sigmoid. All kernels excluding the polynomial kernel, has exceptional results. With further research, I learned the reason that polynomial was lagging behind was becasue the date had more of a linear setup than nonlinear, which polynomial is not the best at. Overall, we could see that the SVM models outperformed all the other models.
#!!!In conclusion, the model and its hyperparameters should be chosen depending on the particulars of the dataset and the issue at hand. SVM models frequently provide reliable performance when the appropriate kernel is used, but for the best outcomes, fine-tuning and modeling experimentation are crucial.