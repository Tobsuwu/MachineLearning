import numpy as np
import random
import timeit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt


def load_cifar10():
    """
    This function loads the cifar10 pictures from the tensorflow.keras.dataset library and returns the train and test
    images (pixel values scaled down from 0 to 1) with corresponding "one-hot" labels.
    :return:
    """

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    train_images, test_images = train_images/255, test_images/255
    train_labels, test_labels = to_categorical(train_labels), to_categorical(test_labels)
    # Use part of the training images as validation_images. Good practice according to the internet.
    validation_images, validation_labels = train_images[0:10000], train_labels[0:10000]
    train_images, train_labels = train_images[10000:], train_labels[10000:]

    return train_images, train_labels, test_images, test_labels, validation_images, validation_labels


def main():
    # Tensorflow-gpu used to speed up calculations (NVIDIA GeForce GTX 1070)
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    train_images, train_labels, test_images, test_labels, validation_images, validation_labels = load_cifar10()

    start = timeit.default_timer()

    # Create the model. Lots of parameters played with. 'categorical_crossentropy' gave better accuracy
    # compared to 'mse'. Added convolutional layers to improve accuracy. Oversampling is a problem. Dropout was used to
    # help with this, but the improvement was small.

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Flatten())
    #  Trying to improve the model by fighting oversampling.
    model.add(Dropout(0.2))
    model.add(Dense(256, activation='sigmoid'))
    model.add(Dense(10, activation='sigmoid'))

    model.compile(optimizer=SGD(learning_rate=0.075), loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_images, train_labels, epochs=20, verbose=1, validation_data=(validation_images, validation_labels))
    stop = timeit.default_timer()
    print("Time elapsed: {t} seconds.\n".format(t=stop-start))

    plt.plot(history.history['accuracy'], label='accuracy', color='r')
    plt.plot(history.history['val_accuracy'], label='val_accuracy', color='m')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend()
    plt.title('Classification accuracy of training data and validation data for the model')
    plt.show()

    # Test the model by calculating the classification accuracy for the test data and plotting some random images from
    # the test data.

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
    print("Classification accuracy for test data: ", test_acc)

    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    plt.figure()
    for i in range(9):
        rand_num = random.randint(0, 9999)
        im = np.array(test_images[rand_num])
        rand_im = im.reshape(1, 32, 32, 3)
        predicted_label = class_names[model.predict(rand_im).argmax()]
        correct_label = class_names[test_labels[rand_num].argmax()]
        plt.subplot(330 + 1 + i)
        plt.gca().set_title("Correct label: {}. Predicted label {}.".format(correct_label, predicted_label))
        plt.imshow(test_images[rand_num])

    plt.suptitle("Classification of 9 random images from the test data")
    plt.show()


main()
