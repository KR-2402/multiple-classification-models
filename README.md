# Machine Learning Classification Assignment

## a. Problem statement
The objective of this project is to perform multi-class image classification (recognizing handwritten digits from 0 to 9) using five distinct machine learning algorithms, evaluate their performance metrics using comprehensive testing criteria, and deploy an interactive Streamlit web application.

## b. Dataset description
- **Source:** Scikit-Learn Built-in Optical Recognition of Handwritten Digits Dataset.
- **Total Instances:** 1,797 samples (Satisfies the requirement of >= 500 instances).
- **Total Features:** 64 numerical pixel attributes (Satisfies the requirement of >= 12 features).

## c. Github Repository Link
- https://github.com/KR-2402/multiple-classification-models.git

## d. Models Used: Evaluation Metrics Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9722 | 0.9941 | 0.9725 | 0.9722 | 0.9723 | 0.9691 |
| **Decision Tree** | 0.8417 | 0.9182 | 0.8456 | 0.8417 | 0.8418 | 0.8238 |
| **KNN** | 0.9750 | 0.9982 | 0.9750 | 0.9750 | 0.9748 | 0.9722 |
| **Naive Bayes** | 0.8472 | 0.9654 | 0.8751 | 0.8472 | 0.8490 | 0.8329 |
| **Support Vector Machine** | 0.9806 | 0.9991 | 0.9809 | 0.9806 | 0.9805 | 0.9784 |

## Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved a high accuracy of 97.22% and an MCC of 0.9691, proving that linear decision boundaries perform exceptionally well on normalized feature spaces. |
| **Decision Tree** | Yielded a lower accuracy of 84.17% and an MCC of 0.8238, struggling with high-dimensional variance compared to linear and neighbor-based models. |
| **KNN** | Performed brilliantly with an accuracy of 97.50% and an MCC of 0.9722, effectively mapping close spatial distance neighborhoods. |
| **Naive Bayes** | Maintained a steady baseline accuracy of 84.72% with a precision peak of 0.8751, constrained slightly by feature correlation assumptions. |
| **Support Vector Machine** | Secured the top overall performance with an accuracy of 98.06% and an MCC of 0.9784, maximizing hyper-plane boundaries successfully. |
| **Overall, Winner for your dataset?** | **Support Vector Machine & KNN** (SVM achieved the highest overall accuracy and MCC score, closely followed by KNN). |