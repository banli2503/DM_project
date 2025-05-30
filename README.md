### Project Title
Design and Analysis of a Heart Disease Prediction Model Based on a Heart Disease Dataset

**Author**
LiCheng Rao

#### Abstract
Research summary, findings, and next steps.


#### Rationale
Why should anyone care about this question?


#### Research Question
What are you trying to answer?
This project aims to utilize the heart disease dataset provided by UCI to build a machine learning model for predicting heart disease risk. I completed data cleaning, feature selection, model training and evaluation, and compared the performance of multiple supervised learning models. Ultimately, Logistic Regression was selected as the primary model. Experimental results show that the model achieved high accuracy and good generalization ability on the test set.

#### Data Sources
What data will you use to answer you question?
UCI Machine Learning Repository: Heart Disease Dataset
This data includes several physiological indicator fields, such as age, gender, blood pressure, cholesterol, ECG results, etc.

#### Methodology
What methods are you using to answer the question?
Data Preprocessing: Handling missing values, standardizing numerical features, processing categorical variables.
Feature Selection: Correlation analysis, ANOVA (Analysis of Variance), L1 regularization.
Model Training and Comparison: Logistic Regression, KNN (K-Nearest Neighbors), Random Forest, SVM (Support Vector Machine).
Model Evaluation: Using metrics such as Accuracy, ROC-AUC curve, and F1-score.
Visualization: Feature importance/weight plots, confusion matrix, training process curves.

#### Results
What did your research find?
The Logistic Regression model achieved an accuracy of approximately 84.6%, a recall of 81.2%, and an AUC score of 0.89. The Random Forest model performed the best, reaching an accuracy of 89.7%, an AUC score of 0.93, and an F1-score of 0.90, indicating that the model performs well in balancing precision and recall. The SVM model achieved an accuracy of 83.2%, but its training time was significantly longer than the other models, and its performance was inferior to Random Forest when handling large-scale feature data. Furthermore, through visual analysis using confusion matrices and ROC curves, we found that the Random Forest model not only had the lowest misclassification rate but also better distinguished between positive (diseased) and negative (non-diseased) samples.

#### Next steps
Integrate and test with more medical clinical data.
Experiment with ensemble models (such as XGBoost, VotingClassifier) to improve accuracy.
Further investigate the model’s ability to handle borderline cases and its capability for uncertainty estimation.

#### Conclusion
Positive/Negative results, recommendations, caveats/cautions.
his study successfully constructed a well-performing heart disease prediction model. Although the current results are positive, more data and model optimization are still needed to adapt to more complex medical scenarios. Furthermore, the interpretability and transparency of the model are also important directions for future research.

### Bibliography 
Reference citations (Chicago style - AMS/AIP or ACM/IEEE).
Detrano, R., et al. “International application of a new probability algorithm for the diagnosis of coronary artery disease.” The American journal of cardiology 64.5 (1989): 304-310.
Pedregosa, F., et al. “Scikit-learn: Machine learning in Python.” Journal of machine learning research 12 (2011): 2825-2830.
IEEE Style: “Heart Disease Dataset.” UCI Machine Learning Repository, https://archive.ics.uci.edu/ml/datasets/heart+Disease

##### Contact and Further Information
Name: Rao Licheng
Student ID: A20563424
Class: 19G231
GitHub Project Address: