[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/qzbVi4h1)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=12273087&assignment_repo_type=AssignmentRepo)
# COSC 3337 - Data Science I 
## Decison Trees, SVM, and K-NN Assignment ##

### Due Date: October 11, 11:56 PM ###

#### The goal of this assignment is to:
1. Understand how decision trees, k-nn, and support vector machines (SVMs) are used for classification tasks.
2. Learn to tune hyperparameters in decision trees, k-nn, and SVMs to avoid overfitting.
3. Gain proficiency in using recall, precision, and accuracy as metrics for evaluating classification model performance.
4. Learn to use cross-validation techniques to assess model performance.

## Dataset - Emotion Recognition
**Intended Outcome:** Apply machine learning algorithms for the task of emotion recognition in tweets. This assignment will focus on understanding and implementing classification algorithms such as decision trees, k-nearest neighbors (k-nn), and support vector machines (SVMs). Students will also gain skills in model evaluation and hyperparameter tuning.

**Dataset Description:** The dataset for this assignment is adapted from the SemEval2018's "Affects in Tweets" task. Originally a multi-label classification problem involving 11 emotions, it has been transformed into a multi-class classification problem for this assignment. Only tweets labeled with a single emotion are considered, and the assignment will focus on the four most common emotions: Anger, Joy, Sadness, and Optimism.

**Dataset Preprocessing:** The original dataset consisted of 'text' and 'label' columns. The text has been preprocessed with user mentions anonymized by a @USER token, line breaks and website links are removed. Following preprocessing, the Universal Sentence Encoder is used to create sentence embeddings for each tweet, afterwhich Principal Component Analyses is applied and the embedding dimensions reduced from 512 to 128. 

**Attributes**

The text column is not expected to be used in this assignment. It is presented for the students reference of the original tweet. The features used for this assignment are the 128 Principal Components dervied from the original 512 dimensonal sentence embedding. 
- *text*: Original text of the tweet
- *Class labels for each tweet*
    - Anger: 0
    - Joy: 1
    - Optimism: 2
    - Sadness: 3
- *feature columns*
    - enc_0 to enc_127

*Dataset pre-processed by Tuck, Bryan, E.*

### Assignment Tasks ###

**Task 1**

A. Using all attributes, build a Decision Tree model to predict emotions: Train the Decision Tree model using the given maximum depths (3, 7, 11, 15).  **8 points**

B. Perform 5-fold cross-validation for each of the 4 max depths and compute accuracy (mean of validation scores), precision and recall. Generate a table, as given below, for the obtained results. **5 points**   

**Decision Tree Experiments**
| Max Depths  | Accuracy    | Precision  | Recall     |
| ----------- | ----------- | ---------- | ---------- |
| 3           |             |            |            |
| 7           |             |            |            |
| 11          |             |            |            |
| 15          |             |            |            |

C. Explain how the tree size/depth affects model performance in the context of overfitting/underfitting. **3 points**

D. Explain the meaning of the difference in accuracy, precision and recall scores in relation to the task; only if there is a significant difference. **2 points**

**Task 2**

A. Using all attributes, build a k-nn classifier to predict emotions: Train the k-nn using the given neighbors (3, 9, 17, 25).  **8 points**

B. Perform 5-fold cross-validation for each of the 4 neighbor size and compute accuracy (mean of validation scores), precision and recall. Generate a table, as given below, for the obtained results. **5 points**

**K-NN Experiments**
|  Neighbors  |   Accuracy  | Precision  | Recall     |
| ----------- | ----------- | ---------- | ---------- |
| 3           |             |            |            |
| 9           |             |            |            |
| 17          |             |            |            |
| 25          |             |            |            |

C. Explain how the number of neighbors affects model performance in the context of overfitting/underfitting. **3 points**

D. Explain the meaning of the difference in accuracy, precision and recall scores in relation to the task; only if there is a significant difference. **2 points**

**Task 3**

A. Using all attributes, build an SVM Model to predict emotions: Train the SVM model using the given kernel functions (linear, polynomial, rbf, sigmoid). **10 points**

B. Perform 5-fold cross-validation for each of the 4 kernel functions and compute accuracy (mean of validation scores), precision and recall. Generate a table, as given below, for the obtained results. **5 points**

**SVM Experiments**
| Kernel Function |  Accuracy  |  Precision  |  Recall    |
| --------------- | ----------- | ---------- | ---------- |
| Linear               |             |            |            |
| Polynomial               |             |            |            |
| RBF              |             |            |            |
| Sigmoid              |             |            |            |

C. Discuss the impact of different kernels on model performance. **2 points**

D. Explain the meaning of the difference in accuracy, precision and recall scores in relation to the task. **2 points**

**Task 4**
- Interpret the tables you generated in questions 1B, 2B, 3B; compare the performance of the Decision Tree, K-NN and SVM models. Which model performs better? Why do you think that is the case? What would you recommend to further improve each model’s performance? **5 points** (*and up to 3 extra points*)



## Deliverables

1. A Python file with your code and analysis
2. A separate pdf of your report (1-2 pages) of your findings and the conclusions for Task 1C, 1D, 2C, 2D, 3C, 3D, and 4. Boldly state the task name before your explanations. 

## Submission Instructions
Please push and commit your developed solution to the github repository. Please ensure that your Github username, name, and PSID are filled in at the start of the file. Please ensure that all required plots are generated while running your solution and that your interpretations and conclusions, as required for each task, are included as comments after each task listed in the shell python file.
