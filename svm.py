#-------------------------------------------------------------------------
# AUTHOR: William Armstrong
# FILENAME: svm.py
# SPECIFICATION: SVM
# FOR: CS 4210- Assignment #3
# TIME SPENT: 30 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#importing some Python libraries
from sklearn import svm
import csv

dbTest = []
X_training = []
Y_training = []
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]
highestAccuracy = 0

#reading the data in a csv file
with open('optdigits.tra', 'r') as trainingFile:
  reader = csv.reader(trainingFile)
  for i, row in enumerate(reader):
      X_training.append(row[:-1])
      Y_training.append(row[-1])

#reading the data in a csv file
with open('optdigits.tes', 'r') as testingFile:
  reader = csv.reader(testingFile)
  for i, row in enumerate(reader):
      dbTest.append (row)

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
#--> add your Python code here

bestAcc = 0
for c_val in c: #iterates over c
    for deg_val in degree: #iterates over degree
        for kern_val in kernel: #iterates kernel
           for shape_val in decision_function_shape: #iterates over decision_function_shape
                numTotal = 0
                numCorrect = 0

                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape as hyperparameters. For instance svm.SVC(c=1)
                clf = svm.SVC(C = c_val, degree = deg_val, kernel = kern_val, decision_function_shape = shape_val)

                #Fit SVM to the training data
                clf.fit(X_training, Y_training)

                #make the classifier prediction for each test sample and start computing its accuracy
                #--> add your Python code here
                for i, testSample in enumerate(dbTest):
                    X = testSample[0:-1]
                    Y = int(testSample[-1])
                    class_predicted = int(clf.predict([X])[0])
                    numTotal = numTotal + 1
                    if (Y == class_predicted):
                        numCorrect = numCorrect + 1

                #check if the calculated accuracy is higher than the previously one calculated. If so, update update the highest accuracy and print it together with the SVM hyperparameters
                #Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                #--> add your Python code here
                accuracy = numCorrect / float(numTotal)
                if (accuracy > bestAcc):
                    bestAcc = accuracy
                    print("Highest SVM accuracy so far: " + format(bestAcc,'.4f') + ", Parameters: c=" + str(c_val) + ", degree=" + str(deg_val) + ", kernel=" + kern_val + ", decision_function_shape=" + shape_val )









