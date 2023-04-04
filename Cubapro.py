import numpy as np
import time
from numba import cuda, float32
import matplotlib.pyplot as plt

@cuda.jit
def matrix_multiply_kernel(matrix1, matrix2, result):
    row, col = cuda.grid(2)
    if row < result.shape[0] and col < result.shape[1]:
        sum = 0.0
        for i in range(matrix1.shape[1]):
            sum += matrix1[row][i] * matrix2[i][col]
        result[row][col] = sum

def multiply_matrices(matrix1, matrix2, n1, m1, n2, m2, num_threads):
    result = np.zeros((n1, m2), dtype=np.float32)

    # Вычисление размера блока и сетки потоков
    block_size = (16, 16)
    grid_size = (int(np.ceil(n1 / block_size[0])), int(np.ceil(m2 / block_size[1])))

    # Копирование матриц на GPU
    matrix1_gpu = cuda.to_device(matrix1)
    matrix2_gpu = cuda.to_device(matrix2)
    result_gpu = cuda.to_device(result)

    # Запуск ядра на GPU
    for i in range(1000):
        matrix_multiply_kernel[grid_size, block_size](matrix1_gpu, matrix2_gpu, result_gpu)

    # Получение результата
    result = result_gpu.copy_to_host()

    return result


if __name__ == '__main__':
    n1, m1, n2, m2 = 500, 500, 500, 500
    matrix1 = np.random.rand(n1, m1).astype(np.float32)
    matrix2 = np.random.rand(n2, m2).astype(np.float32)
    threads_num = [1, 2, 4, 8, 16]  # список количества потоков для тестирования
    results = []  # список результатов для каждого теста
    times = []  # список времени выполнения каждого теста
    for i in threads_num:
        start_time = time.time()
        res = multiply_matrices(matrix1, matrix2, n1, m1, n2, m2, i)
        end_time = time.time()
        time_taken = end_time - start_time
        results.append(res)
        times.append(time_taken)
        print(f"{i} threads took {time_taken:.5f} seconds")

    # график зависимости времени от количества потоков
    plt.plot(threads_num, times)
    plt.title('Зависимость времени от количества потоков')
    plt.xlabel('Количество потоков')
    plt.ylabel('Время выполнения (сек)')
    plt.savefig(f"plot.png")
    plt.show()
