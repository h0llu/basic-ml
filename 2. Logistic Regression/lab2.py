import numpy as np
from scipy.optimize import minimize
from matplotlib import pyplot as plt

# В этой лабораторной работе вам предстоит обучить модель логистической регрессии.
# Представьте, что вы сотрудник приемной комиссии и хотите оценить шансы студента на прохождение на бюджет.
# У вас есть исторические данные о сдачах студентами двух входных экзаменов по 100-балльной шкале.
# Кроме того, данные размечены. То есть студенты, прошедшие отбор помечены меткой 1, остальные 0.

# Ваша задача - заполнить код в функциях:
# logistic - вычисляет логистическую функцию от аргумента,
# compute_cost - вычисляет функцию стоимости,
# compute_cost_grad - вычисляет градиент функции стоимости.

# В данной задаче будет использоваться градиентный метод оптимизации, определенный в библиотеке scipy, так что
# не нужно реализовывать градиентный спуск.

# По ходу работы смотрите в консоль. Там будут выводиться результаты написанных вами функций и ожидаемые результаты.
# Так вы сможете проверить себя.

# Комментарии к графикам:
# Первый график - исходные данные
# Второй - исходные данные с результатом работы модели - границей решений.
# Черным цветом обозначено поле, попадание точки в которое означает ее
# принадлежность к 1 классу (студент проходит на бюджет), белым - принадлежность к 0 классу (не проходит на бюджет).


def logistic(z):
    # Функция принимает аргумент z - скаляр, вектор или матрицу в виде объекта numpy.array()
    # Должна возвращать скяляр, вектор или матрицу (в зависимости от размерности z)
    # результатов вычисления логистической функции от элементов z
    return 1 / (1 + np.exp(-z))

def h(X, theta):
    return logistic(np.dot(X, theta))

def compute_cost(X, y, theta):

    # Функция принимает матрицу данных X, вектор целевых переменных y и вектор параметров theta.
    # Должна возвратить число - результат вычисления функции стоимости в точке theta.

    # m - количество примеров в выборке, n - количество признаков у каждого примера
    m = X.shape[0]
    return 1/m * sum(-y*np.log(h(X, theta)) - (1-y)*np.log(1-h(X, theta)))


def compute_cost_grad(X, y, theta):
    # Функция принимает матрицу данных X, вектор целевых переменных y и вектор параметров theta.
    # Должна возвратить вектор координат градиента функции стоимости в точке theta.
    # Вектор можно заполнять в виде списка python. При возврате он преобразуется в массив numpy.

    # m - количество примеров в выборке, n - количество признаков у каждого примера
    m = X.shape[0]
    return 1/m * (h(X, theta) - y).dot(X[:,])

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
    X, y = load_data('lab2data1.txt')

    print(f'logistic(0) = {logistic(np.array(0))} (должно быть 0.5)\n'
        f'logistic(-10) = {logistic(np.array(-10))} (должно быть ~0)\n'
        f'logistic(10) = {logistic(np.array(10))} (должно быть ~1)')

    plt.title('Исходные данные')
    plt.scatter(X[y == 0][:, 1], X[y == 0][:, 2], c='r', marker='o')
    plt.scatter(X[y == 1][:, 1], X[y == 1][:, 2], c='g', marker='x')
    plt.show()

    np.seterr(all = 'ignore')
    # неправильно (???)
    # init_theta = np.zeros((X.shape[1], 1))
    print(f'{init_theta=}')
    init_theta = np.zeros(X.shape[1])
    cost0 = compute_cost(X, y, init_theta)
    print(f'Функция стоимости при начальном theta = {cost0} (должно быть ~0.693)')

    opt_theta_obj = minimize(lambda th: compute_cost(X, y, th), init_theta,
                            method='CG',
                            jac=lambda th: compute_cost_grad(X, y, th),
                            options={'gtol': 1e-5, 'maxiter': 200, 'disp': False})

    print('Минимизация функции стоимости ' + ('прошла успешно.' if opt_theta_obj.success else 'не удалась.'))
    opt_theta = opt_theta_obj.x
    print(f'{opt_theta=}')

    opt_cost = compute_cost(X, y, opt_theta)
    print(f'Функция стоимости при оптимальном theta = {opt_cost} (должно быть ~0.203)')

    hm = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            hm[i, j] = round(logistic(opt_theta @ np.array([1, j, i])))
    c = plt.pcolor(range(101), range(101), hm, cmap='Greys')
    plt.colorbar(c)

    plt.title('Построенная граница решений')
    plt.scatter(X[y == 0][:, 1], X[y == 0][:, 2], c='r', marker='o')
    plt.scatter(X[y == 1][:, 1], X[y == 1][:, 2], c='g', marker='x')
    plt.show()


if __name__ == '__main__':
    main()
