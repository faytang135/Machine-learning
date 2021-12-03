# Machine learning: SVM and Naïve Bayes algorithm
## The purpsoe of the file is to compare two algorithms: SVM and Naive Bayes algorithm in predicting drug types.

**Naïve Bayes Classifier**: assumes that the presence of  a particular feature in a class is unrelated to the presence of any other features.
![image](https://user-images.githubusercontent.com/92997647/144664516-3848996b-6f2b-4a18-a4df-ed4df408308a.png)


**Support Vector Machines (SVM)**: a supervised machine learning algorithm by which we can perform Regression and Classification.
SVM generates the optimal hyperplane in an iterative manner, which is used to minimize an error. The core idea of SVM is to find a maximum marginal hyperplane (MMH) that best divides the dataset into classes. 
![image](https://user-images.githubusercontent.com/92997647/144664567-fbb3459e-f72a-4fac-99f0-6fbe893c7433.png)


**Metrics for evaluating the algorithms**: classification_report and confusion_matrix 
**Implementation**: Python

The following section demonstrates the detailed steps in Python:

### Step 1:  to load liabraries and load dataset

    #Load Libraries
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    %matplotlib inline
    
    #Load Dataset
    data=pd.read_csv('./drugdataset.csv')
    data.head()


### Step 2:  Explore the data

    #Explore dataset
    data.describe()
    data.info()
    
    #Boxplot Visualization
    plt.figure(figsize=(15,8))
    sns.boxplot(data=data)  

### Step 3:  Split training and test datasets and normalize the data

    #Create x and y variables
    x=data.drop('Drug', axis=1).to_numpy()
    y=data['Drug'].to_numpy()

    #Create Training and Test Datasets
    from sklearn.model_selection import train_test_split
    x_train, x_test,y_train, y_test = train_test_split(x, y, stratify=y,test_size=0.2,random_state=100)

    #Scale the Data
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train2 = sc.fit_transform(x_train)
    x_test2 = sc.transform(x_test)

### Step 4:  Generate algorithms for SVM and Naive Bayes using sklearn package in Python
    #Script for SVM and NB
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import classification_report, confusion_matrix  

    for name,method in [('SVM', SVC(kernel='linear',random_state=100)),
                    ('Naive Bayes',GaussianNB())]: 
      method.fit(x_train2,y_train)
      predict = method.predict(x_test2)
      target_names=['drugA','drugB','drugC','drugX','drugY']
      print('\nEstimator: {}'.format(name)) 
      print(confusion_matrix(y_test,predict))  
      print(classification_report(y_test,predict,target_names=target_names))
    

## Compare the outcome:
    Estimator: SVM
    [[ 5  0  0  0  0]
     [ 0  2  0  0  1]
     [ 0  0  3  0  0]
     [ 0  0  0 11  0]
     [ 0  0  0  1 17]]
                  precision    recall  f1-score   support

           drugA       1.00      1.00      1.00         5
           drugB       1.00      0.67      0.80         3
           drugC       1.00      1.00      1.00         3
           drugX       0.92      1.00      0.96        11
           drugY       0.94      0.94      0.94        18

        accuracy                           0.95        40
       macro avg       0.97      0.92      0.94        40
    weighted avg       0.95      0.95      0.95        40


    Estimator: Naive Bayes
    [[ 5  0  0  0  0]
     [ 0  3  0  0  0]
     [ 0  0  3  0  0]
     [ 0  0  0 10  1]
     [ 1  1  3  1 12]]
                  precision    recall  f1-score   support

           drugA       0.83      1.00      0.91         5
           drugB       0.75      1.00      0.86         3
           drugC       0.50      1.00      0.67         3
           drugX       0.91      0.91      0.91        11
           drugY       0.92      0.67      0.77        18

        accuracy                           0.82        40
       macro avg       0.78      0.92      0.82        40
    weighted avg       0.86      0.82      0.83        40

As the outcome shows, SVM algorithm performs better than Naïve Bayes algorithm in this case!

The End!
