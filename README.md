# **Object Recognition Task**

The objective is to use classical machine learning methods to develop a classification solution, specifically, we shall be solving an object recognition task. Each object is represented by a 28x28 dimensional image in a single 'flattened' 784 dimensional vector with an associated label (+1 or -1).

## Step by Step Approach

We start by understanding the expected accuracy of a random classifier (one that generates random labels for a given example) for this problem over the training and test datasets. We also plot AUC-ROC and AUC-PR of a random classifier for this problem over the training and test datasets and demonstrate a proof using a programming experiment as to why this would be the case.

We then perform 5-fold stratified cross-validation over the training dataset using a k=3 nearest neighbour (kNN) classifier and demonstrate the results by outputing the average and standard deviation for each of the following metrics across all folds and in a single table:

    * Accuracy
    * Balanced accuracy,
    * AUC-ROC  
    * AUC-PR

To study the the impact of various forms of [pre-processing](https://scikit-learn.org/stable/modules/preprocessing.html) (e.g., mean-standard deviation or standard scaling or min-max scaling) on the cross-validation performance, we perform the following pre-processing steps, repeat the above task and observe the results:

    *Mean-standard deviation
    *Standard scaling 
    *Min-max scaling

To choose an optimal classfier for our task, we experiment with the following classifiers and perform grid search as well as preprocessing data and report the cross validation results:
    * Perceptron
    * Naïve Bayes Classifier
    * Linear SVM 
    * Kernelized SVM

To evaluate the results, we use different performance metrics like accuracy, AUC-ROC curve and AUC-PR cuves etc.

Using PCA, we perform dimensionality reduction and then perform classification using a Kernelized SVM with PCA, optimize hyperparameters and observe the results by plotting a scatter plot of the results.

Finally, we develop an optimal pipeline for classification based on all the above analyses and reporting the result in a single column prediction file containing the prediction score of the corresponding example in Xtest.

As an additional task, using the data, we consider an alternate classification problem in which the label of an example is based on whether it is a part of the training set (label = -1) or the test set (label = +1). We then calculate the average and standard deviation of AUC-ROC using 5-fold stratified cross-validation for a classifier that is trained to solve this prediction task.

For a quick access, start by running the notebook file that uses 5 fold stratified cross-validation over the data with a different classifiers : 


- ```mining.ipynb```


