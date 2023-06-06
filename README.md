# KNN - From Scratch
## Simple implementation of K Nearest Neighbor Algorithm

KNN is a very simple supervised learning algorithm which is capable of developing complex decision boundries.
This is a very simple implementation of KNN algorithm designed for learning purpose.

Note - All the attributes have been normalized to ensure that one attribute does not get more importance than other attributes.

## Features
- The dataset file to run the code is present in the repo. Alternatively, any other .csv file can also be provided.
- I have used Euclidean Distance to calculate distance between attributes.
- To test the model, I have varied the value of K from 1 to 52 in steps of 2, and repeated the experiment 20 and averaged the results
- I have plotted accuracy and f1-score of model's performance over training and testing data.
- The effect of increasing the k value of the model can be studied through the graph

## Steps to run the code

1. The .ipynb file can be run through jupyter notebook
2. The .py file needs to be run through the following commands.
    * Ensure that all requirements have been installed using
        ```sh
            pip install -r requirements.txt
        ```
    * Run the file using
        ```sh
            python knn.py
        ```
