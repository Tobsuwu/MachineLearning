import pickle
import numpy as np
import matplotlib.pyplot as plt
from random import random
import timeit
from skimage.transform import resize
from scipy.stats import multivariate_normal, norm


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


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


def cifar10_color(X):
    """
    This function converts the original 32x32x3 images to 1x1x3 images.
    :param X: 50000 32x32x3 images from the test_batches.
    :return: Xp: 50000 1x1x3 resized images.
    """
    start = timeit.default_timer()
    Xp = np.zeros([len(X), 3])
    for i in range(len(X)):
        Xp[i] = resize(X[i], (1, 1))
        if i % 10000 == 0:
            print("{}/{} of training images resized.".format(i, len(X)))

    stop = timeit.default_timer()
    print("Time elapsed: ", stop - start, " seconds")
    # test = Xp[1].reshape(1, 1, 3)
    # plt.imshow(test)
    # plt.show()
    return Xp


def cifar_10_naivebayes_learn(Xp, Y):
    """
    This function calculates the means and variances for all labels and color channels.
    :param Xp: 50000 1x1x3 resized images
    :param Y: 50000 x 1 ndarray of labels with corresponding indices to Xp.
    :return: mu, sigma2, p: mu (10x3) sigma2 (10x3) p (10X1) mean, variance and class probability (prior) values for each class and colour channel
    """
    mu = np.zeros([10, 3])
    sigma2 = np.zeros([10, 3])

    mu0, mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, mu9 = [], [], [], [], [], [], [], [], [], []
    s0, s1, s2, s3, s4, s5, s6, s7, s8, s9 = [], [], [], [], [], [], [], [], [], []

    # Collecting different labels into corresponding mean and variance lists. mu0 for 0 label,
    for i in range(len(Xp)):
        if Y[i] == 0:
            mu0.append(Xp[i])
            s0.append(Xp[i])
        elif Y[i] == 1:
            mu1.append(Xp[i])
            s1.append(Xp[i])
        elif Y[i] == 2:
            mu2.append(Xp[i])
            s2.append(Xp[i])
        elif Y[i] == 3:
            mu3.append(Xp[i])
            s3.append(Xp[i])
        elif Y[i] == 4:
            mu4.append(Xp[i])
            s4.append(Xp[i])
        elif Y[i] == 5:
            mu5.append(Xp[i])
            s5.append(Xp[i])
        elif Y[i] == 6:
            mu6.append(Xp[i])
            s6.append(Xp[i])
        elif Y[i] == 7:
            mu7.append(Xp[i])
            s7.append(Xp[i])
        elif Y[i] == 8:
            mu8.append(Xp[i])
            s8.append(Xp[i])
        elif Y[i] == 9:
            mu9.append(Xp[i])
            s9.append(Xp[i])

    # Turn lists into arrays
    mu0, mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, mu9 = np.asarray(mu0), np.asarray(mu1), np.asarray(mu2), np.asarray(mu3), np.asarray(mu4), np.asarray(mu5), np.asarray(mu6), np.asarray(mu7), np.asarray(mu8), np.asarray(mu9)
    s0, s1, s2, s3, s4, s5, s6, s7, s8, s9 = np.asarray(s0), np.asarray(s1), np.asarray(s2), np.asarray(s3), np.asarray(s4), np.asarray(s5), np.asarray(s6), np.asarray(s7), np.asarray(s8), np.asarray(s9)

    # Priori probabilities. (Number of this clas)/(all samples)
    p = [[len(mu0)/len(Xp), len(mu1)/len(Xp), len(mu2)/len(Xp), len(mu3)/len(Xp), len(mu4)/len(Xp), len(mu5)/len(Xp), len(mu6)/len(Xp), len(mu7)/len(Xp), len(mu8)/len(Xp), len(mu9)/len(Xp)]]
    p = np.asarray(p).T

    # Calculate the means and variances. Prolly a neater way to do this but this works.
    for j in range(3):
        mu[0, j] = np.mean(mu0[:, j])
        mu[1, j] = np.mean(mu1[:, j])
        mu[2, j] = np.mean(mu2[:, j])
        mu[3, j] = np.mean(mu3[:, j])
        mu[4, j] = np.mean(mu4[:, j])
        mu[5, j] = np.mean(mu5[:, j])
        mu[6, j] = np.mean(mu6[:, j])
        mu[7, j] = np.mean(mu7[:, j])
        mu[8, j] = np.mean(mu8[:, j])
        mu[9, j] = np.mean(mu9[:, j])
        sigma2[0, j] = np.var(s0[:, j])
        sigma2[1, j] = np.var(s1[:, j])
        sigma2[2, j] = np.var(s2[:, j])
        sigma2[3, j] = np.var(s3[:, j])
        sigma2[4, j] = np.var(s4[:, j])
        sigma2[5, j] = np.var(s5[:, j])
        sigma2[6, j] = np.var(s6[:, j])
        sigma2[7, j] = np.var(s7[:, j])
        sigma2[8, j] = np.var(s8[:, j])
        sigma2[9, j] = np.var(s9[:, j])

    return mu, sigma2, p


def cifar_10_bayes_learn(Xp, Y):
    """
    This function calculates the means and variances for all labels and color channels.
    :param Xp: 50000 1x1x3 resized images
    :param Y: 50000 x 1 ndarray of labels with corresponding indices to Xp.
    :return: mu, sigma2, p: mu (10x3) sigma2 (10x3) p (10X1) mean, variance and class probability (prior) values for each class and colour channel
    """
    mu = np.zeros([10, 3])
    sigma = np.zeros([10, 3, 3])

    mu0, mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, mu9 = [], [], [], [], [], [], [], [], [], []
    s0, s1, s2, s3, s4, s5, s6, s7, s8, s9 = [], [], [], [], [], [], [], [], [], []

    # Collecting different labels into corresponding mean and variance lists. mu0 for 0 label,
    for i in range(len(Xp)):
        if Y[i] == 0:
            mu0.append(Xp[i])
            s0.append(Xp[i])
        elif Y[i] == 1:
            mu1.append(Xp[i])
            s1.append(Xp[i])
        elif Y[i] == 2:
            mu2.append(Xp[i])
            s2.append(Xp[i])
        elif Y[i] == 3:
            mu3.append(Xp[i])
            s3.append(Xp[i])
        elif Y[i] == 4:
            mu4.append(Xp[i])
            s4.append(Xp[i])
        elif Y[i] == 5:
            mu5.append(Xp[i])
            s5.append(Xp[i])
        elif Y[i] == 6:
            mu6.append(Xp[i])
            s6.append(Xp[i])
        elif Y[i] == 7:
            mu7.append(Xp[i])
            s7.append(Xp[i])
        elif Y[i] == 8:
            mu8.append(Xp[i])
            s8.append(Xp[i])
        elif Y[i] == 9:
            mu9.append(Xp[i])
            s9.append(Xp[i])

    # Turn lists into arrays
    mu0, mu1, mu2, mu3, mu4, mu5, mu6, mu7, mu8, mu9 = np.asarray(mu0), np.asarray(mu1), np.asarray(mu2), np.asarray(mu3), np.asarray(mu4), np.asarray(mu5), np.asarray(mu6), np.asarray(mu7), np.asarray(mu8), np.asarray(mu9)
    s0, s1, s2, s3, s4, s5, s6, s7, s8, s9 = np.asarray(s0), np.asarray(s1), np.asarray(s2), np.asarray(s3), np.asarray(s4), np.asarray(s5), np.asarray(s6), np.asarray(s7), np.asarray(s8), np.asarray(s9)

    # Priori probabilities. (Number of this clas)/(all samples)
    p = [[len(mu0)/len(Xp), len(mu1)/len(Xp), len(mu2)/len(Xp), len(mu3)/len(Xp), len(mu4)/len(Xp), len(mu5)/len(Xp), len(mu6)/len(Xp), len(mu7)/len(Xp), len(mu8)/len(Xp), len(mu9)/len(Xp)]]
    p = np.asarray(p).T

    # Calculate the means and variances. Prolly a neater way to do this but this works.
    for j in range(3):
        mu[0, j] = np.mean(mu0[:, j])
        mu[1, j] = np.mean(mu1[:, j])
        mu[2, j] = np.mean(mu2[:, j])
        mu[3, j] = np.mean(mu3[:, j])
        mu[4, j] = np.mean(mu4[:, j])
        mu[5, j] = np.mean(mu5[:, j])
        mu[6, j] = np.mean(mu6[:, j])
        mu[7, j] = np.mean(mu7[:, j])
        mu[8, j] = np.mean(mu8[:, j])
        mu[9, j] = np.mean(mu9[:, j])

    sigma[0] = np.cov(np.stack((s0[:, 0], s0[:, 1], s0[:, 2]), axis=0))
    sigma[1] = np.cov(np.stack((s1[:, 0], s1[:, 1], s1[:, 2]), axis=0))
    sigma[2] = np.cov(np.stack((s2[:, 0], s2[:, 1], s2[:, 2]), axis=0))
    sigma[3] = np.cov(np.stack((s3[:, 0], s3[:, 1], s3[:, 2]), axis=0))
    sigma[4] = np.cov(np.stack((s4[:, 0], s4[:, 1], s4[:, 2]), axis=0))
    sigma[5] = np.cov(np.stack((s5[:, 0], s5[:, 1], s5[:, 2]), axis=0))
    sigma[6] = np.cov(np.stack((s6[:, 0], s6[:, 1], s6[:, 2]), axis=0))
    sigma[7] = np.cov(np.stack((s7[:, 0], s7[:, 1], s7[:, 2]), axis=0))
    sigma[8] = np.cov(np.stack((s8[:, 0], s8[:, 1], s8[:, 2]), axis=0))
    sigma[9] = np.cov(np.stack((s9[:, 0], s9[:, 1], s9[:, 2]), axis=0))

    return mu, sigma, p


def cifar10_classifier_naivebayes(x, mu, sigma2, p):
    """
    This function calculates, classifies and returns the naive Bayesian optimal class c for sample x
    :param x: Testdata sample (image) from the test data set (32x32x3)
    :param mu: mean values for each class and color channel (10x3)
    :param sigma2: variance values for each class and color channel (10x3)
    :param p: prior probability of each class (10x1)
    :return: c: Naive Bayesian optimal class for x
    """
    x = resize(x, (1, 1))  # Resize the 32x32x3 image to 1x1x3 image. Mean of every clor channel
    max_classP = 0  # Highest class probability
    c = 0  # Class corresponding to the highest probability

    # Denominator is same for all the samples, so we can leave that out of the equation.
    # Note that P is no longer a probability nut we don't care about that

    for i in range(len(mu)):
        P = norm.pdf(x.item(0), mu.item((i, 0)), np.sqrt(sigma2.item((i, 0)))) *\
            norm.pdf(x.item(1), mu.item((i, 1)), np.sqrt(sigma2.item((i, 1)))) *\
            norm.pdf(x.item(2), mu.item((i, 2)), np.sqrt(sigma2.item((i, 2)))) * p.item(i)
        if P > max_classP:
            max_classP = P
            c = i

    return c


def cifar10_classifier_bayes(x, mu, sigma, p):
    """
    This function calculates, classifies and returns the Bayesian optimal class c for sample x
    :param x: Testdata sample (image) from the test data set (32x32x3)
    :param mu: mean values for each class and color channel (10x3)
    :param sigma: covariance values for each class and color channel (10x3x3)
    :param p: prior probability of each class (10x1)
    :return: c: Bayesian optimal class for x
    """
    x = resize(x, (1, 1))  # Resize the 32x32x3 image to 1x1x3 image. Mean of every clor channel
    np.reshape(x, (1, 3))
    max_classP = 0  # Highest class probability
    c = 0  # Class corresponding to the highest probability

    for i in range(len(mu)):
        P = multivariate_normal.pdf(x, mu[i], sigma[i])*p.item(i)
        if P > max_classP:
            max_classP = P
            c = i

    return c

def main():

    # Assign all the training_batches and test_batch to dicts.
    datadict1 = unpickle('C:/Users/Topsu/PycharmProjects/PRML/week4/cifar-10-batches-py/data_batch_1')
    datadict2 = unpickle('C:/Users/Topsu/PycharmProjects/PRML/week4/cifar-10-batches-py/data_batch_2')
    datadict3 = unpickle('C:/Users/Topsu/PycharmProjects/PRML/week4/cifar-10-batches-py/data_batch_3')
    datadict4 = unpickle('C:/Users/Topsu/PycharmProjects/PRML/week4/cifar-10-batches-py/data_batch_4')
    datadict5 = unpickle('C:/Users/Topsu/PycharmProjects/PRML/week4/cifar-10-batches-py/data_batch_5')
    testdict = unpickle('C:/Users/Topsu/PycharmProjects/PRML/week4/cifar-10-batches-py/test_batch')

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
    labeldict = unpickle('C:/Users/Topsu/PycharmProjects/PRML/week4/cifar-10-batches-py/batches.meta')
    label_names = labeldict["label_names"]

    # Reshaping and transforming to np.array
    X1 = X1.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
    Y1 = np.array(Y1)
    X2 = X2.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
    Y2 = np.array(Y2)
    X3 = X3.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
    Y3 = np.array(Y3)
    X4 = X4.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
    Y4 = np.array(Y4)
    X5 = X5.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
    Y5 = np.array(Y5)

    # Combining arrays into one big array (5*10000)
    allX = np.concatenate([X1, X2, X3, X4, X5])
    allY = np.concatenate([Y1, Y2, Y3, Y4, Y5])

    # Reshaping and transforming to np.array
    X_test = X_test.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
    Y_test = np.array(Y_test)

    Xp = cifar10_color(allX)

    naive_bayesian(Xp, allY, X_test, Y_test)
    bayesian(Xp, allY, X_test, Y_test)


def naive_bayesian(X, Y, X_test, Y_test):

    """ Naive Bayesian. First problem of the exercise. """

    mu, sigma2, p = cifar_10_naivebayes_learn(X, Y)

    print()
    start = timeit.default_timer()
    Y_naivebayes = np.zeros_like(Y_test)
    for i in range(len(Y_naivebayes)):
        Y_naivebayes[i] = cifar10_classifier_naivebayes(X_test[i], mu, sigma2, p)
        if i % 1000 == 0:
            print("{}/{} of test images classified.".format(i, len(X_test)))

    stop = timeit.default_timer()
    print("Time elapsed: ", stop - start, " seconds")
    print()
    print("Prediction accuracy (NaiveBayes) is: ", class_acc(Y_naivebayes, Y_test))


def bayesian(X, Y, X_test, Y_test):

    """ 3D Bayesian. Second problem of the exercise. """

    mu, sigma, p = cifar_10_bayes_learn(X, Y)

    print()
    start = timeit.default_timer()
    Y_bayes = np.zeros_like(Y_test)
    for i in range(len(Y_bayes)):
        Y_bayes[i] = cifar10_classifier_bayes(X_test[i], mu, sigma, p)
        if i % 1000 == 0:
            print("{}/{} of test images classified.".format(i, len(X_test)))

    stop = timeit.default_timer()
    print("Time elapsed: ", stop - start, " seconds")
    print()
    print("Prediction accuracy (Bayes) is: ", class_acc(Y_bayes, Y_test))

main()
