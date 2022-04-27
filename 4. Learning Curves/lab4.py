import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize


# В данной лабораторной работе вам предлагается несколько моделей линейной регрессии и пронаблюдать эффект недообучения,
# переобучения, а так же пронаблюдать соответствующие этим случаям кривые обучения.

# Для выполнения данной лабораторной работы, необходимо заполнить код в следующих фукнциях:
# scale_features - стандартизирует признаки,
# compute_cost - вычисляет функцию стоимости,
# compute_cost_grad - вычисляет градиент фукнции стоимости,
# compute_learning_curves - вычисляет значения кривых обучения.

# Кроме заполнения этих функций настоятельно рекомендуется ознакомится с содержимым остальных функций.
# Комментарии к графикам:
# 1. Исходные данные
# 2. Обученная модель линейной регрессии.
#    Здесь можно наблюдать эффект недообучения модели из-за недостаточной сложности.
# 3. Кривые обучения для недообученной модели, изображенной на графике 2.
#    Можно заметить, что ошибка на всем обучающем наборе и на всем валидационном наборе велика - признак недообучения.
# 4. Обученная модель с полиномиальными признаками. На графике можно наблюдать, как функция проходит почти через
#    все точки обучающего набора (отмечены красными крестами).
# 5. Кривые обучения, соответствующие модели с полиномиальными признаками. Можно наблюдать большой разрыв между ошибкой
#    на всем обучающем наборе и на всем валидационном - признак переобучения.


def compute_hypothesis(X, theta):
    # Функция для вычисления гипотезы линейной регрессии.
    # Принимает на вход X - матрицу данных, theta - вектор параметров.
    # Возвращает вектор значений функции гипотезы на всех примерах из матрицы X.

    return X @ theta

def train_linear_regression(X_train, y_train, init_theta, lamb):
    # Функция для обучения алгоритма линейной регрессии.
    # Принимает на вход X_train - матрицу данных, y_train - вектор значений целевой переменной,
    # init_theta - вектор начальногог приблежения параметров для старта градиентного спуска,
    # lamb - параметр регуляризации.
    # Возвращает кортеж из двух элементов: флаг успеха минимизации (True или False) и оптимальный вектор параметров.

    opt_theta_obj = minimize(lambda th: compute_cost(X_train, y_train, th, lamb), init_theta,
                             method='BFGS',
                             jac=lambda th: compute_cost_grad(X_train, y_train, th, lamb),
                             options={'gtol': 1e-5, 'maxiter': 200, 'disp': False})
    return opt_theta_obj.success, opt_theta_obj.x

def show_learning_curves(curves):
    # Функция для отрисовки кривых обучения.
    # Принимает на вход список котрежей. Индекс кортежа - координата по оси абсцисс, элементы кортежа - значения
    # кривой обучения на обучающем наборе и валидационном наборе соответственно.

    tr_curve = list(map(lambda x: x[0], curves))
    val_curve = list(map(lambda x: x[1], curves))

    plt.plot(range(1, len(tr_curve) + 1), tr_curve)
    plt.plot(range(1, len(val_curve) + 1), val_curve)

    plt.xlim(2, len(curves) + 1)
    plt.ylim(0, 150)

    plt.title('Кривые обучения')

    plt.ylabel('Ошибка')
    plt.xlabel('Количество примеров в обучающем наборе')

    plt.legend(['Обучающий набор', 'Валидационный набор'])

    plt.show()

def map_features(X, p):
    # Функция для добавления степеней в набор признаков примера.
    # Принимает на вход X - матрицу данных, p - степень рещультирующего полинома.
    # Возвращает новую матрицу данных, в которую добавлены степени первого (X[:, 1]) признака матрицы данных
    # до p-ой включительно.

    for i in range(2, p + 1):
        X = np.append(X, X[:, 1].reshape(X.shape[0], 1) ** i, axis=1)
    return X

def load_data(data_file_path):
    # Функция для загрузки данных.
    # Принимает на вход путь к файлу.
    # На выход возвращает кортеж из двух элементов:
    # матрицу данных с фиктивным признаком и вектор значений целевой переменной.

    with open(data_file_path) as input_file:
        X = list()
        y = list()
        for line in input_file:
            *row, label = map(float, line.split(','))
            X.append([1] + row)
            y.append(label)
        return np.array(X, float), np.array(y, float)

def scale_features(X):
    # Функция для стандартизации признаков.
    # Принимает на вход матрицу данных X.
    # На выход возвращает кортеж из двух элементов: новую матрицу данных, со стандартизированными признаками и
    # кортеж из двух элементов: вектор средних значений признаков и вектор средних квадратических отклонений.

    result = (X[:, 1:] - np.mean(X[:, 1:], axis=0)) / np.std(X[:, 1:], axis=0)
    mean = X[:, 1:].mean(axis=0)
    std = X[:, 1:].std(axis=0)

    return np.insert(result, 0, X[:, 0], axis=1), (mean, std)

def compute_cost(X, y, theta, lamb):
    # Функция для расчета стоимости (ошибки) модели на наборе данных.
    # Принимает на вход X - матрицу данных, y - вектор значений целевой переменной, theta - вектор параметров,
    # lamb - параметр регуляризации.
    # Возвращает значение функции стоимости на данном наборе данных.

    m = X.shape[0]

    return 1/(2*m) * np.sum((compute_hypothesis(X, theta) - y) ** 2) + lamb/m * np.sum(theta**2)

def compute_cost_grad(X, y, theta, lamb):
    # Функция для расчета градиента функции стоимости в заданной точке.
    # Принимает на вход X - матрицу данных, y - вектор значений целевой переменной,
    # theta - точка, в которой расчитывается градиент, lamb - параметр регуляризации.
    # Возвращает градиент функции стоимости в заданной точке.

    m = X.shape[0]

    return (compute_hypothesis(X, theta) - y).dot(X[:])/m + 2*lamb/m * theta

def compute_learning_curves(X_train, y_train, X_val, y_val, lamb):
    # Функция для расчета кривых обучения.
    # Принимает на вход X_train - матрицу данных обучющего набора,
    # y_train - вектор целевой переменной обучающего набора, X_val - матрицу данных валидационного набора,
    # y_val - вектор целевой переменной валидационного набора, lamb - параметр решуляризации.

    m = X_train.shape[0]
    result = list()  # список значений кривых обучения

    for i in range(1, m):
        theta = train_linear_regression(X_train[:i], y_train[:i], np.zeros(X_train.shape[1]), lamb)[1]

        train_error = compute_cost(X_train[:i], y_train[:i], theta, 0)
        val_error = compute_cost(X_val, y_val, theta, 0)

        result.append((train_error, val_error))
    
    return result


def main():
    # Загрузка данных
    X_train, y_train = load_data('lab4data1.txt')
    X_val, y_val = load_data('lab4data2.txt')
    X_test, y_test = load_data('lab4data3.txt')

    plt.title('Обучающий набор')
    plt.scatter(X_train[:, 1], y_train, c='r', marker='x')
    plt.show()

    init_theta = np.array([1, 1])
    init_lamb = 1.0

    print('Значение функции стоимости при theta = [1, 1], lamb = 1.0 (должно быть ~303.993): ',
        compute_cost(X_train, y_train, init_theta, init_lamb))

    print('Значение градиента функции стоимости при theta = [1, 1], lamb = 1.0 (должно быть ~[-15.30, 598.25]): ',
        compute_cost_grad(X_train, y_train, init_theta, init_lamb))

    print()
    print('Обучение модели линейной регрессии..')

    success, opt_theta = train_linear_regression(X_train, y_train, init_theta, init_lamb)
    print('Минимизация функции стоимости ' + ('прошла успешно.' if success else 'не удалась.'))

    plt.title('Обученая модель при lamb = 1.0')
    plt.scatter(X_train[:, 1], y_train, c='r', marker='x')
    plt.plot(X_train[:, 1], compute_hypothesis(X_train, opt_theta))
    plt.show()

    lin_learning_curves = compute_learning_curves(X_train, y_train, X_val, y_val, 0.0)
    show_learning_curves(lin_learning_curves)

    poly_pow = 6  # степень полинома для полиномиальной регрессии, можно варьировать
    poly_lamb = 0.4  # параметр регуляризации для полиномиальной регрессии, можно варьировать

    X_train_poly, (mean, std) = scale_features(map_features(X_train, poly_pow))
    X_val_poly = scale_features(map_features(X_val, poly_pow))[0]

    X_plot_poly = np.array([[i ** p for p in range(poly_pow + 1)] for i in range(-50, 40)], dtype=float)
    X_plot_poly[:, 1:] = (X_plot_poly[:, 1:] - mean) / std

    print()
    print('Обучение модели полномиальной регрессии..')

    success, opt_theta = train_linear_regression(X_train_poly, y_train, np.zeros(X_train_poly.shape[1]), poly_lamb)
    print('Минимизация функции стоимости ' + ('прошла успешно.' if success else 'не удалась.'))

    plt.title('Обученая модель при lamb = ' + str(poly_lamb))
    plt.scatter(X_train_poly[:, 1], y_train, c='r', marker='x')
    plt.scatter(X_val_poly[:, 1], y_val, c='b', marker='x')
    plt.plot(X_plot_poly[:, 1], compute_hypothesis(X_plot_poly, opt_theta))
    plt.show()

    poly_learning_curves = compute_learning_curves(X_train_poly, y_train, X_val_poly, y_val, poly_lamb)
    show_learning_curves(poly_learning_curves)

    X_test = map_features(X_test, poly_pow)
    X_test[:, 1:] = (X_test[:, 1:] - mean) / std

    print()
    print('Оценка качества модели на тестовом наборе:', compute_cost(X_test, y_test, opt_theta, 0.0))

if __name__ == '__main__':
    main()
