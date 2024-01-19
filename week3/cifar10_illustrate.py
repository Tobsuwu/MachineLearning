import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
import timeit

start = timeit.default_timer()

def class_acc(pred, gt):
    """
    This function calculates the classification accuracy for predicted labels in relation to the ground truth
    (true labels)
    :param pred: Predicted labels of the images
    :param gt: True labels of the images
    :return accuracy: Accuracy of the classification
    """
    correct_pred = 0  # Correct predictions
    for i in range(len(pred)):
        if pred[i] == gt[i]:
            correct_pred += 1
    accuracy = correct_pred/len(pred)
    return accuracy

def cifar10_classifier_random(Y):
    """
    This function returns a random class label (0-9) for the input Y
    :param Y: The original labels
    :return: Y with labels replaced as random labels
    """
    Y_rand = np.zeros_like(Y)
    for i in range(len(Y)):
        rand_label = np.random.randint(0, 9)
        Y_rand[i] = rand_label
    return Y_rand

def cifar10_classifier_1nn(x, trdata, trlabels):
    """
    This function finds the best value for input vector x from the training set trdata and returns the same label.
    "The best value" is the closest picture in the training set calculated by the distance of the pixel values.
    Will run for a while.
    :param x: the test pictures to be classified (10000)
    :param trdata:  Training data (all pictures, 50000)
    :param trlabels: The labels of the training data
    :return Y_1NN: The labels of the nearest_neighbours (a.k.a closest pictures)
    """
    Y_1NN = np.zeros(len(x))
    for i in range(len(x)):
        # Calculates the distance of test image regards to all images in the training data and returns the results
        # as array. Replaced one for loop and improved computation time to 47 min
        sum_distance = np.sum((x[i] - trdata)**2, axis=(1, 2, 3))
        shortest_d = np.argmin(sum_distance)  # argument/index of the shortest distance from the array
        Y_1NN[i] = trlabels[shortest_d]

    return Y_1NN

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


# Assign all the training_batches and test_batch to dicts.
datadict1 = unpickle('C:/Users/Topsu/PycharmProjects/PRML/week3/cifar-10-batches-py/data_batch_1')
datadict2 = unpickle('C:/Users/Topsu/PycharmProjects/PRML/week3/cifar-10-batches-py/data_batch_2')
datadict3 = unpickle('C:/Users/Topsu/PycharmProjects/PRML/week3/cifar-10-batches-py/data_batch_3')
datadict4 = unpickle('C:/Users/Topsu/PycharmProjects/PRML/week3/cifar-10-batches-py/data_batch_4')
datadict5 = unpickle('C:/Users/Topsu/PycharmProjects/PRML/week3/cifar-10-batches-py/data_batch_5')
testdict = unpickle('C:/Users/Topsu/PycharmProjects/PRML/week3/cifar-10-batches-py/test_batch')

# Create X variables for data (pictures) and Y variables to corresponding labels.
X1 = datadict1["data"]
Y1 = datadict1["labels"]
X2 = datadict2["data"]
Y2 = datadict2["labels"]
X3 = datadict3["data"]
Y3 = datadict3["labels"]
X4 = datadict4["data"]
Y4 = datadict4["labels"]
X5 = datadict5["data"]
Y5 = datadict5["labels"]

# Same for test_batch
X_test = testdict["data"]
Y_test = testdict["labels"]

# Used in given illustration
labeldict = unpickle('C:/Users/Topsu/PycharmProjects/PRML/week3/cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

# Reshaping and transforming to np.array
X1 = X1.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y1 = np.array(Y1)
X2 = X2.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y2 = np.array(Y2)
X3 = X3.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y3 = np.array(Y3)
X4 = X4.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y4 = np.array(Y4)
X5 = X5.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y5 = np.array(Y5)

# Combining arrays into one big array (5*10000)
allX = np.concatenate([X1, X2, X3, X4, X5])
allY = np.concatenate([Y1, Y2, Y3, Y4, Y5])

# Reshaping and transforming to np.array
X_test = X_test.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
Y_test = np.array(Y_test)

# Randomly generated class labels for comparison
Y_random = cifar10_classifier_random(Y_test)

# Class labels calculated using the first nearest-neighbour method (1-NN)
Y_1nn = cifar10_classifier_1nn(X_test, allX, allY)

# Randomly show some pictures from the first training batch
for i in range(X1.shape[0]):
    # Show some images randomly
    if random() > 0.999:
        plt.figure(1);
        plt.clf()
        plt.imshow(X1[i])
        plt.title(f"Image {i} label={label_names[Y1[i]]} (num {Y1[i]})")
        plt.pause(1)

# Calculate and print the prediction accuracies for different methods using class_acc function
print("Prediction accuracy (correct labels) is: ", class_acc(Y1, Y1))
print("Prediction accuracy (random labels) is: ", class_acc(Y_random, Y1))
print("Prediction accuracy (1NN) is: ", class_acc(Y_1nn, Y_test))

stop = timeit.default_timer()
print("Time elapsed: ", stop - start, " seconds")