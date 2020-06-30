# Final-Project-Group3
# Description of the project
In this project, we set out to solve the problem of predicting the probability of users overdue in the next 6 months, based on users' historical behavior data of the Chinese Credit Platform by providing a tool to predict customer repayment ability so that reduce the expected loss level of credit.

Our project contains two parts, namely exploration and interaction. In the exploration part, we present our research methodology, algorithm used, analysis and visualization of the data. In the interaction part, we use the test dataset to predict the 'target', which is the result that users overdue or not, it is expressed by 0 and 1.

We use machine learning models for predictions. Test data will be fed into the model and the result will be displayed in the dataset as 0 and 1.

# Detail:
# Group-Proposal folder:
Final-Group-Project-Report.pdf: The proposal of our project plan.

# Code folder:
data_visualization.py contains codes on manipulating the data and generating some graphs, in order to find some patterns behind the data.

feature_engineering.py contains codes on data cleaning and feature engineering, in order to delete some unnecessary features and create some insightful features. We combined train and test dataset at first. Then, a dataset named “combined_train_test.csv” was generated containing all the new features.

feature_selection.py contains codes on feature selection. Then, a csv file named “feature_selected.csv” was created, which contains 200 the most important features and one target.

Model and Accuracy.py contains codes on building model and calculate the accuracy for the model. We calculate the confusion matrix, precision score, recall score, f1 score, and AUC to test the model.

We also provide the training set: PPD_Training_Master_GBK_3_1_Training_Set.csv, and test set: test.csv.

# Final-Group-Presentation:  
Final-Group-Presentation.pdf: PowerPoint of our final project presentation

# Final-Group-Project-Report:
The report of the group work.

# Jiaxi Jiang, Xueqi Zhou and Meijiao Wang folders:
they are the group members' individual work folder, included individual reports and python code. 


