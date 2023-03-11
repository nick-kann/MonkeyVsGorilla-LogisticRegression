import os
import numpy as np
from PIL import Image
import copy
import matplotlib.pyplot as plt


def main():
    path = os.path.dirname(os.path.realpath(__file__))
    classes = ["gorilla", "monkey"]
    epsilon = 1e-5

    # y = 1 for "monkey"
    # y = 0 for "gorilla"
    train_x_orig, train_y_orig = load_dataset(os.path.join(path, "MonkeyVsGorillaDataset"), "train", classes)

    test_x_orig, test_y_orig = load_dataset(os.path.join(path, "MonkeyVsGorillaDataset"), "test", classes)

    # Shuffling the sets so that the model doesn't learn in a certain order and bias towards beef carpaccio
    train_x_shuffle, train_y_shuffle = shuffle_set(train_x_orig, train_y_orig)
    test_x_shuffle, test_y_shuffle = shuffle_set(test_x_orig, test_y_orig)

    # Reshaping the label vectors to make each example be represented as a column vector
    # (m,) will be reshaped to (1, m)
    # This is done to ensure that the matrix operations between the input and label matrices are performed correctly
    train_y_shuffle = train_y_shuffle.reshape(1, train_y_shuffle.shape[0])
    test_y_shuffle = test_y_shuffle.reshape(1, test_y_shuffle.shape[0])

    m_train = train_x_shuffle.shape[0]
    m_test = test_x_shuffle.shape[0]
    num_px = train_x_shuffle.shape[1]

    # Flattens and transposes the input sets so that each column vector is a different example of the flattened image
    train_x_flatten = train_x_shuffle.reshape(train_x_shuffle.shape[0], -1).T
    test_x_flatten = test_x_shuffle.reshape(test_x_shuffle.shape[0], -1).T

    # Preprocessing the dataset

    # Standardizing the dataset
    train_x = train_x_flatten / 255
    test_x = test_x_flatten / 255

    # Creating the model
    logistic_regression_model = model(train_x, train_y_shuffle, test_x, test_y_shuffle, num_iterations=3000,
                                      learning_rate=0.00005, print_cost=True)

    # Creating a list of indexes of test examples to show
    indexes = [0, 1]

    show_prediction_example(logistic_regression_model, test_x, indexes)

    # Plotting the cost function every hundred iterations
    costs = np.squeeze(logistic_regression_model['costs'])
    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('Iterations (hundreds)')
    plt.title("Learning rate = " + str(logistic_regression_model["learning_rate"]))
    plt.show()

    # Add your own test image to the same directory and change the test image to the file name
    # to see the model's prediction of your image
    test_image = "none"

    if test_image != "none":
        fname = os.path.join(path, test_image)
        image = np.array(Image.open(fname).resize((256, 256)))
        plt.imshow(image)
        image = image.reshape((256 * 256 * 3, 1))
        image = image / 255
        y_prediction = int(np.squeeze(predict(logistic_regression_model["w"], logistic_regression_model["b"], image)))
        class_prediction = "\"monkey\"" if y_prediction == 1 else "\"beef carpaccio\""
        plt.title(f"y = {y_prediction}, the model predicted that it is a {class_prediction} picture.")
        plt.axis('off')
        plt.show()


def load_dataset(path, dataset_split, classes):
    x = []
    y = []
    # 0 for gorilla
    # 1 for monkey
    for label, cls in enumerate(classes):
        class_path = os.path.join(path, dataset_split, cls)
        for img_path in os.listdir(class_path):
            img = Image.open(os.path.join(class_path, img_path)).resize((256, 256))
            x.append(np.array(img))
            y.append(label)

    return np.array(x), np.array(y)


def shuffle_set(set_x, set_y):
    n_samples = set_x.shape[0]
    permutation = np.random.permutation(n_samples)
    return set_x[permutation], set_y[permutation]


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


# Creates a column vector with shape (dim, 1) to represent the weights (w) and initialized the bias (b) to zero
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1), dtype=float)
    b = float(0)
    return w, b


def propagate(w, b, X, Y, epsilon=1e-5):
    # m - the number of examples
    m = X.shape[1]

    # Forward propagation
    a = sigmoid(np.dot(w.T, X) + b)
    cost = (-1 / m) * (np.dot(Y, np.log(a + epsilon).T) + np.dot(1 - Y, np.log(1 - a + epsilon).T))

    # Backward propagation
    dw = (1 / m) * np.dot(X, (a - Y).T)
    db = (1 / m) * np.sum(a - Y)

    cost = np.squeeze(np.array(cost))

    grads = {"dw": dw,
             "db": db}

    return grads, cost


def optimize(w, b, X, Y, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

            if print_cost:
                print("Cost after iteration " + str(i) + ": " + str(cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


def predict(w, b, X):
    m = X.shape[1]
    yhat = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            yhat[0, i] = 1
        else:
            yhat[0, i] = 0

    return yhat


def model(train_x, train_y, test_x, test_y, num_iterations=2000, learning_rate=0.5, print_cost=True):
    w, b = initialize_with_zeros(train_x.shape[0])
    params, grads, costs = optimize(w, b, train_x, train_y, num_iterations, learning_rate, print_cost)
    w = params["w"]
    b = params["b"]
    Y_prediction_train = predict(w, b, train_x)
    Y_prediction_test = predict(w, b, test_x)

    if print_cost:
        print("Train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_y)) * 100))
        print("Test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_y)) * 100))

    model_dict = {"costs": costs,
                  "Y_prediction_test": Y_prediction_test,
                  "Y_prediction_train": Y_prediction_train,
                  "w": w,
                  "b": b,
                  "learning_rate": learning_rate,
                  "num_iterations": num_iterations}

    return model_dict


def show_prediction_example(model, test_x, index):
    for i in index:
        plt.imshow(test_x[:, i].reshape((256, 256, 3)))
        y_prediction = int(model['Y_prediction_test'][0, i])
        class_prediction = "\"monkey\"" if y_prediction == 1 else "\"beef carpaccio\""
        plt.title(f"y = {y_prediction}, the model predicted that it is a {class_prediction} picture.")
        plt.axis('off')
        plt.show()


if __name__ == '__main__':
    main()
