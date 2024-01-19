""" This program calculates and plots a linear 2D model from points given by the user.
Left click adds a point, middle button removes previous point and right click stops collecting points. """

# Import necessary packages
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import MouseButton

# Linear solver
def mylinfit(x, y):
    """
    This program returns the coefficients used in the linear mode. Derived by hand in homework 1.
    :param x: x values of clicked points
    :param y: y values of clicked points
    :param N: number of points
    :return: coefficients for the linear model
    """
    N = len(x)
    a = np.sum(y*x)/np.sum(x**2) - (np.sum(y)*np.sum(x**2)*np.sum(x) - np.sum(y*x)*(np.sum(x))**2) / (np.sum(x**2)*(N*np.sum(x**2)-(np.sum(x))**2))
    b = (np.sum(y)*np.sum(x**2) - np.sum(y*x)*np.sum(x)) / (N*np.sum(x**2) - (np.sum(x))**2)
    return a, b

# Main
def main():
    """
    Main function that plots the graphs and calls myliinfit.
    :return:
    """

    plt.plot(np.arange(100), label='Dummy line for adjusting the default window size')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Select N>2 number of points with left clicks. Push middle button to cancel previous point and '
              'right click to stop collecting', fontdict=None, loc='center', pad=None)
    cords = plt.ginput(n=-1, timeout=-1, show_clicks=True, mouse_add=MouseButton.LEFT, mouse_pop=MouseButton.MIDDLE,
                       mouse_stop=MouseButton.RIGHT)
    plt.close()

    conv_tup = np.array(cords)  # Convert the tuple output of plt.ginput to array
    x = conv_tup[:, 0]
    y = conv_tup[:, 1]

    a, b = mylinfit(x, y)  # Get the coefficients for the linear model and print them
    print('a (slope): ', a)
    print('b (y-intercept): ', b)

    # Plot the selected points
    plt.plot(x, y, 'kx', label='Selected points')

    # Plot the linear model
    xp = np.arange(min(x), max(x), 0.1)
    plt.plot(xp, a*xp+b, 'r-', label='Fitted linear model')

    # Plot 1D-polyfit for comparison
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(xp, p(xp), 'g--', label='1D-Polyfit')
    plt.title('Linear 2D linefit for given points')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

main()