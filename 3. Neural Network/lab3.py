import idx2numpy
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

# Вам предлагается обучить трехслойную нейронную сеть прямого распространения с сигмоидальными функциями активации на
# наборе данных MNIST - множестве изображений рукописных цифр и соответствующих метках 0-9.

# Ваша задача заполнить код в следующих функциях:
# logistic - вычисляет логистическую функцию от аргумента,
# compute_hypothesis - вычисляет выходы сети для заданного набора данных,
# compute_cost - вычисляет функцию стоимости,
# compute_cost_grad - вычисляет градиент функции стоимости с помощью алгоритма обратного распространения ошибки.

# По ходу работы смотрите в консоль. Там будут выводиться результаты написанных вами функций и ожидаемые результаты.
# Так вы сможете проверить себя.

# Ниже заданы "константы", с которыми можно "поиграть" - поварьировать их и посмотреть, что будет.
HIDDEN_LAYER_SIZE = 25  # Количество нейронов в скрытом слое.

# Всего у вас есть 60 000 изображений, поэтому сумма следующих двух констант не должна быть больше 60 000.
SAMPLES_TO_TRAIN = 1000  # Количество примеров в обучающей выборка, на ней будет производить подбор параметров модели.
SAMPLES_TO_TEST = 100  # Количество примеров в тестовой выборке, на ней будет оцениваться качество модели.


def one_hot_encode(y):
    # Функция кодирует вектор чисел с метками классов в матрицу, каждая строка которой является унитарным кодом
    # соответствующей метки.
    return np.array(np.arange(0, 10, 1) == y, dtype=np.int32)


def get_random_matrix(l_in, l_out):
    # Функция для генерации матрицы случайных параметров.
    return np.random.uniform(-0.5, 0.5, (l_out, l_in + 1))


def compute_cost_grad_approx(X, y, theta, lamb):
    # Фукнция для численного расчета градиента функции стоимости. Такой метод полезно использовать при проверке
    # правильности аналитического расчета.

    eps = 1e-5
    grad = np.zeros(theta.size)
    perturb = np.zeros(theta.size)
    for i in range(theta.shape[0]):
        if i % 1000 == 0:
            print(f"{round(i / theta.shape[0] * 100, 2)}%")
        perturb[i] = eps
        grad[i] = (
            compute_cost(X, y, theta + perturb, lamb)
            - compute_cost(X, y, theta - perturb, lamb)
        ) / (2 * eps)
        perturb[i] = 0
    return grad


def logistic(z):
    # Функция принимает аргумент z - скаляр, вектор или матрицу в виде объекта numpy.array()
    # Должна возвращать скяляр, вектор или матрицу (в зависимости от размерности z)
    # результатов вычисления логистической функции от элементов z

    return 1 / (1 + np.exp(-z))


def logistic_grad(z):
    # Функция принимает аргумент z - скаляр, вектор или матрицу в виде объекта numpy.array()
    # Должна возвращать скяляр, вектор или матрицу (в зависимости от размерности z)
    # результатов вычисления производной логистической функции от элементов z

    x = logistic(z)
    return x * (1 - x)


def compute_hypothesis(X, theta):
    # Функция принимает матрицу данных X без фиктивного признака и вектор параметров theta.
    # Вектор theta является "разворотом" матриц Theta1 и Theta2 в плоский вектор.
    # Функция должна возвращать матрицу выходов сети для каждого примера из обучающего набора,
    # а также матрицу входов нейронов скрытого слоя и матрицу выходов скрытого слоя (это понадобится для реализации
    # алгоритма обратного распространения ошибки).

    # Восстановление матриц Theta1 и Theta2 из ветора theta:
    Theta1 = theta[: HIDDEN_LAYER_SIZE * (X.shape[1] + 1)].reshape(
        HIDDEN_LAYER_SIZE, X.shape[1] + 1
    )
    Theta2 = theta[HIDDEN_LAYER_SIZE * (X.shape[1] + 1) :].reshape(
        10, HIDDEN_LAYER_SIZE + 1
    )

    Z3 = None  # Будущая матрица входов выходного слоя
    A3 = None  # Будущая матрица выходов, нужно заполнить
    Z2 = None  # Будущая матрица входов скрытого слоя, нужно заполнить
    A2 = None  # Будущая матрица выходов скрытого слоя, нужно заполнить

    Z2 = np.insert(X, 0, [1] * X.shape[0], axis=1) @ Theta1.T
    A2 = logistic(Z2)
    Z3 = np.insert(A2, 0, [1] * A2.shape[0], axis=1) @ Theta2.T
    A3 = logistic(Z3)
    return A3, Z2, A2


def compute_cost(X, y, theta, lamb):
    # Функция принимает матрицу данных X без фиктивного признака, вектор верных классов y,
    # вектор параметров theta и параметр регуляризации lamb.
    # Вектор theta является "разворотом" мартиц Theta1 и Theta2 в плоский вектор.
    # Должна вернуть значение функции стоимости в точке theta.

    (
        m,
        n,
    ) = (
        X.shape
    )  # m - количество примеров в выборке, n - количество признаков у каждого примера

    y = one_hot_encode(np.array([[elem] for elem in y]))
    H = compute_hypothesis(X, theta)[0]

    cost = -(1 / m) * sum(sum(y * np.log(H) + (1 - y) * np.log(1 - H))) + lamb / (
        2 * m
    ) * sum(theta**2)
    return cost


def compute_cost_grad(X, y, theta, lamb):
    # Функция принимает матрицу данных X без фиктивного признака, вектор верных классов y,
    # вектор параметров theta и параметр регуляризации lamb.
    # Вектор theta является "разворотом" мартиц Theta1 и Theta2 в плоский вектор.
    # Функция должна реализовать алгоритм обратного распространения ошибки и вернуть матрицы градиентов весов связей.

    (
        m,
        n,
    ) = (
        X.shape
    )  # m - количество примеров в выборке, n - количество признаков у каждого примера

    # Восстановление матриц Theta1 и Theta2 из ветора theta:
    Theta1 = theta[: HIDDEN_LAYER_SIZE * (X.shape[1] + 1)].reshape(
        HIDDEN_LAYER_SIZE, X.shape[1] + 1
    )
    Theta2 = theta[HIDDEN_LAYER_SIZE * (X.shape[1] + 1) :].reshape(
        10, HIDDEN_LAYER_SIZE + 1
    )

    Theta1_grad = None  # Будущая матрица градиентов весов между входным и скрытым слоем, ее нужно заполнить
    Theta2_grad = None  # Будущая матрица градиентов весов между скрытым и выходным слоем, ее нужно заполнить

    A3, Z2, A2 = compute_hypothesis(X, theta)

    # добавить фиктивные признаки в соответствующие матрицы
    X = np.insert(X, 0, [1] * X.shape[0], axis=1)
    A2 = np.insert(A2, 0, [1] * A2.shape[0], axis=1)

    # подсчёт дельт
    delta_output = A3 - one_hot_encode(np.array([[elem] for elem in y]))
    delta_hidden = delta_output @ np.delete(Theta2, 0, 1) * logistic_grad(Z2)

    # матрицы градиентов
    Theta1_grad = 1 / m * (delta_hidden.T @ X) + lamb / m * Theta1
    Theta2_grad = 1 / m * (delta_output.T @ A2) + lamb / m * Theta2

    return np.concatenate((Theta1_grad, Theta2_grad), axis=None)


def load_data_idx(file_path):
    with open(file_path, "rb") as f:
        return idx2numpy.convert_from_file(f)


def main():
    # Загрузка данных
    images = load_data_idx("images.idx")
    features = (
        images.reshape((images.shape[0], images.shape[1] * images.shape[2])) / 128 - 1.0
    )

    labels = load_data_idx("labels.idx")

    X = features[:SAMPLES_TO_TRAIN]
    y = labels[:SAMPLES_TO_TRAIN]

    X_test = features[SAMPLES_TO_TRAIN : SAMPLES_TO_TRAIN + SAMPLES_TO_TEST]
    y_test = labels[SAMPLES_TO_TRAIN : SAMPLES_TO_TRAIN + SAMPLES_TO_TEST]

    print(
        f"logistic(0) = {logistic(np.array(0))} (должно быть 0.5)\n"
        f"logistic(-10) = {logistic(np.array(-10))} (должно быть ~0)\n"
        f"logistic(10) = {logistic(np.array(10))} (должно быть ~1)"
    )
    print()
    print(
        f"logistic_grad(0) = {logistic_grad(np.array(0))} (должно быть ~0.25)\n"
        f"logistic_grad(-10) = {logistic_grad(np.array(-10))} (должно быть ~0)\n"
        f"logistic_grad(10) = {logistic_grad(np.array(10))} (должно быть ~0)"
    )

    im_size = int(features.shape[1] ** 0.5)
    hm = np.zeros((im_size * 10, im_size * 10))
    for i in range(100):
        im_x = i % 10 * im_size
        im_y = i // 10 * im_size
        for j in range(features.shape[1]):
            px_x = im_x + j % im_size
            px_y = im_y + j // im_size
            hm[10 * im_size - 1 - px_x, px_y] = images[i, j % im_size, j // im_size]
    plt.pcolor(hm, cmap="Greys")
    plt.show()

    lamb = 0.5  # можно поварьировать параметр регуляризации и посмотреть, что будет

    init_Theta1 = np.random.uniform(-0.12, 0.12, (HIDDEN_LAYER_SIZE, X.shape[1] + 1))
    init_Theta2 = np.random.uniform(-0.12, 0.12, (10, HIDDEN_LAYER_SIZE + 1))

    init_theta = np.concatenate((init_Theta1, init_Theta2), axis=None)

    back_prop_grad = compute_cost_grad(X[:10], y[:10], init_theta, 0)

    print("Запуск численного расчета градиента (может занять какое-то время)")
    num_grad = compute_cost_grad_approx(X[:10], y[:10], init_theta, 0)

    print("Градиент, посчитанный аналитически: ")
    print(back_prop_grad)

    print("Градиент, посчитанный численно (должен примерно совпадать с предыдущим): ")
    print(num_grad)

    print()
    print("Относительное отклоение градиентов (должно быть < 1e-7): ")
    print(
        (
            np.linalg.norm(back_prop_grad - num_grad)
            / np.linalg.norm(back_prop_grad + num_grad)
        )
    )

    print()
    print("Запуск оптимизации параметров сети (может занять какое-то время)")
    opt_theta_obj = minimize(
        lambda th: compute_cost(X, y, th, lamb),
        init_theta,
        method="CG",
        jac=lambda th: compute_cost_grad(X, y, th, lamb),
        options={"gtol": 1e-3, "maxiter": 3000, "disp": False},
    )

    print(
        "Минимизация функции стоимости "
        + ("прошла успешно." if opt_theta_obj.success else "не удалась.")
    )

    opt_theta = opt_theta_obj.x
    print("Функция стоимости при оптимальных параметрах (должна быть < 0.5): ")
    print(compute_cost(X, y, opt_theta, lamb))

    hypos_train = compute_hypothesis(X, opt_theta)[0]

    true_preds_train_num = 0
    for k in range(hypos_train.shape[0]):
        max_el = np.max(hypos_train[k, :])
        true_preds_train_num += list(
            np.array(hypos_train[k, :] == max_el, dtype=np.int32)
        ) == list(one_hot_encode(y[k]))
    print(
        "Доля верно распознаных цифр в обучающем наборе:",
        true_preds_train_num / hypos_train.shape[0] * 100,
        "%",
    )

    hypos_test = compute_hypothesis(X_test, opt_theta)[0]

    true_preds_test_num = 0
    for k in range(hypos_test.shape[0]):
        max_el = np.max(hypos_test[k, :])
        true_preds_test_num += list(
            np.array(hypos_test[k, :] == max_el, dtype=np.int32)
        ) == list(one_hot_encode(y_test[k]))
    print(
        "Доля верно распознаных цифр в тестовом наборе:",
        true_preds_test_num / hypos_test.shape[0] * 100,
        "%",
    )

    if HIDDEN_LAYER_SIZE**0.5 % 1.0 == 0:
        Theta1 = opt_theta[: HIDDEN_LAYER_SIZE * (X.shape[1] + 1)].reshape(
            HIDDEN_LAYER_SIZE, X.shape[1] + 1
        )
        Theta1 = Theta1[:, 1:]
        Theta1 = (Theta1 - Theta1.mean(axis=0)) / (Theta1.std(axis=0))
        Theta1 = Theta1.reshape(HIDDEN_LAYER_SIZE, 28, 28)
        im_size = int(features.shape[1] ** 0.5)
        size_imgs = int(HIDDEN_LAYER_SIZE**0.5)
        hm = np.zeros((im_size * size_imgs, im_size * size_imgs))
        for i in range(HIDDEN_LAYER_SIZE):
            im_x = i % size_imgs * im_size
            im_y = i // size_imgs * im_size
            for j in range(features.shape[1]):
                px_x = im_x + j % im_size
                px_y = im_y + j // im_size
                hm[size_imgs * im_size - 1 - px_x, px_y] = Theta1[
                    i, j % im_size, j // im_size
                ]
        plt.pcolor(hm, cmap="Greys")
        plt.show()


if __name__ == "__main__":
    main()
