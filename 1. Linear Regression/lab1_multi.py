import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt


def compute_hypothesis(X, theta):
    return X @ theta

def compute_cost(X, y, theta):
    m = X.shape[0]  # количество примеров в выборке
    return 1 / (2 * m) * sum((compute_hypothesis(X, theta) - y) ** 2)

def gradient_descend(X, y, theta, alpha, num_iter):
    history = list()
    m = X.shape[0]  # количество примеров в выборке
    n = X.shape[1]  # количество признаков с фиктивным
    for _ in range(num_iter):
        theta = theta - alpha * (compute_hypothesis(X, theta) - y).dot(X[:,]) / m 
        history.append(compute_cost(X, y, theta))
    return history, theta

def scale_features(X):
    return (X - np.mean(X)) / np.std(X)

def normal_equation(X, y):
    return np.linalg.pinv((X.T @ X)) @ X.T @ y

def load_data(data_file_path):
    with open(data_file_path) as input_file:
        X = list()
        y = list()
        for line in input_file:
            *row, label = map(float, line.split(','))
            X.append([1] + row)
            y.append(label)
        return np.array(X, float), np.array(y, float)



def main():
    X, y = load_data('lab1data2.txt')

    print(compute_hypothesis(X, [0,0,0]).shape, X[:, 1].shape)

    history, theta = gradient_descend(X, y, np.array([0, 0, 0], float), 0.01, 1500)

    plt.title('График изменения функции стоимости от номера итерации до нормализации')
    plt.plot(range(len(history)), history)
    plt.show()

    X = scale_features(X)

    history, theta = gradient_descend(X, y, np.array([0,0,0], float), 0.1, 15000)

    plt.title('График изменения функции стоимости от номера итерации после нормализации')
    plt.plot(range(len(history)), history)
    plt.show()

    theta_solution = normal_equation(X, y)
    print(f'theta, посчитанные через градиентный спуск: {theta}, через нормальное уравнение: {theta_solution}')

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:, 1], X[:, 2], y)
    surf = ax.plot_trisurf(X[:, 1], X[:, 2], compute_hypothesis(X, theta),
        cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


if __name__ == '__main__':
    main()
