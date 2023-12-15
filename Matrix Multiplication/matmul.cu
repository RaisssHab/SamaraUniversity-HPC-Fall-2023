#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <fstream>
#include <iomanip>

// макрос, который используется с функциями CUDA во избежание повторения кода проверки успешности результата функции
#define SAFE_CALL(CallInstruction, error_str) { \
    cudaError_t cuerr = CallInstruction; \
    if (cuerr != cudaSuccess) { \
        fprintf(stderr, error_str, cudaGetErrorString(cuerr)); \
        return 1; \
    } \
}

// макрос index-to-c, преобразующий номер строки и столбца элемента матрицы в индекс массива, который им соответствует (разложение по столбцам) 
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

/* последовательное умножение матриц - для i,j-го элемента выполняется как скалярное произведение i-й строки матрицы A на j-ю строку матрицы B

    A, B - перемножаемые матрицы
    C - результирующая матрица
    nr_cols_A, nr_rows_C, nr_cols_C - количество столбцов/строк в соответствующей матрице
*/
void sequential_mult(const double* A, const double* B, double* C, const int N, const int M, const int K)
{
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            C[IDX2C(i, j, N)] = 0;
        }
    }

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < K; ++j) {
            int ij_idx = IDX2C(i, j, N);
            for (int k = 0; k < M; ++k) {
                const double& c1 = A[IDX2C(i, k, N)];
                const double& c2 = B[IDX2C(k, j, M)];
                C[ij_idx] += c1 * c2;
            }
        }
    }

    //int ij_idx = 0;
    /*
    for (int j = 0; j < K; ++j) {
        //int ij_idx = j * N;
        for (int k = 0; k < M; ++k) {
            for (int i = 0; i < N; ++i) { //, ++ij_idx) {
            int ij_idx = IDX2C(i, j, N);
            
                const double& c1 = A[IDX2C(i, k, N)];
                const double& c2 = B[IDX2C(k, j, M)];
                C[ij_idx] += c1 * c2;
            }
        }
    }
    */
}

/*
    Функция ядра, используемая для вычислений на GPU.
    Параметры аналогичны параметрам для функции последовательного умножения матриц.
    Недостаток: размерность грида должна позволять разместить в грид всю матрицу целиком.
*/
__global__ void addKernel(double* C, double* A, double* B, const int N, const int M, const int K)
{
    // введем переменные i, j  которые характеризуют позицию нити в гриде
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // поскольку размерность грида, выраженная в нитях, необязательно совпадет с размерностью результирующей матрицы, следует проверять наличиие выхода за пределы матрицы
    if (i >= N || j >= K) {
        return;
    }

    // выполняем третий по вложенности цикл при последовательном произведении - скалярное произведение
    int ij_idx = IDX2C(i, j, N); 
    C[ij_idx] = 0;
    for (int k = 0; k < M; ++k) {
        const double& c1 = A[IDX2C(i, k, N)];
        const double& c2 = B[IDX2C(k, j, M)];
        C[ij_idx] += c1 * c2;
    }
}

int main(int argc, char* argv[])
{
    /* 
    чтение входных данных формата:
    N M K
    *матрица А размерности NxM, представленная в виде разложения по строкам*
    *матрица B размерности MxK, представленная в виде разложения по строкам*
    */
    std::ifstream in("input_data.txt");
    if (!in.is_open()) {
        std::cout << "file is unavailable";
        return 1;
    }
    int N, M, K;

    for (int* param : { &N, &M, &K }) {
        in >> (*param);
        if (!in || *param <= 0) {
            std::cout << "one of the provided values is incorrect (or is <= 0), check the file";
            return 1;
        }
    }
    
    // размерности матриц
    int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B, nr_rows_C, nr_cols_C;
    nr_rows_A = nr_rows_C = N;
    nr_cols_A = nr_rows_B = M;
    nr_cols_B = nr_cols_C = K;

    // выделение памяти для матриц, h_C будет хранить в себе результат вычисления центральным процессором, а h_C1 - графическим
    double* h_A = (double*)malloc(nr_rows_A * nr_cols_A * sizeof(double));
    double* h_B = (double*)malloc(nr_rows_B * nr_cols_B * sizeof(double));
    double* h_C = (double*)malloc(nr_rows_C * nr_cols_C * sizeof(double));
    double* h_C1 = (double*)malloc(nr_rows_C * nr_cols_C * sizeof(double));

    // чтение матриц
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            in >> h_A[IDX2C(i, j, N)];
            if (!in) {
                std::cout << "matrix values are incorrect, check the file\n";
                return 1;
            }
        }
    }
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            in >> h_B[IDX2C(i, j, M)];
            if (!in) {
                std::cout << "matrix values are incorrect, check the file\n";
                return 1;
            }
        }
    }

    // выделение памяти под матрицы на графическом процессоре
    double* d_A, *d_B, *d_C;

    cudaMalloc(&d_A, nr_rows_A * nr_cols_A * sizeof(double));
    cudaMalloc(&d_B, nr_rows_B * nr_cols_B * sizeof(double));
    cudaMalloc(&d_C, nr_rows_C * nr_cols_C * sizeof(double));

    // выводим фрагменты матриц

    printf("first 5x5 block of host A matrix:\n");
    for (int i = 0; i < std::min(nr_rows_A, 5); ++i) {
        for (int j = 0; j < std::min(nr_cols_A, 5); ++j) {
            int ij = IDX2C(i, j, nr_rows_A);
            std::cout << std::setw(6) << std::fixed << std::setprecision(3) << h_A[ij] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    printf("first 5x5 block of host B matrix:\n");
    for (int i = 0; i < std::min(nr_rows_B, 5); ++i) {
        for (int j = 0; j < std::min(nr_cols_B, 5); ++j) {
            int ij = IDX2C(i, j, nr_rows_B);
            std::cout << std::setw(6) << std::fixed << std::setprecision(3) << h_B[ij] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // копирование матриц на GPU

    // время копирования с CPU на GPU
    double CPU_GPU_transfer_time = 0;
    double clock_start, clock_end;
    // установка точки старта
    clock_start = clock();
    cudaMemcpy(d_A, h_A, nr_rows_A * nr_cols_A * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, nr_rows_B * nr_cols_B * sizeof(double), cudaMemcpyHostToDevice);
    // установка точки завершения
    clock_end = clock();
    CPU_GPU_transfer_time = (clock_end - clock_start) / CLOCKS_PER_SEC;

    // Синхронизация устройств
    SAFE_CALL(cudaDeviceSynchronize(),
        "Cannot synchronize CUDA kernel: %s\n");

    // вычисляем результирующую матрицу последовательно 12 раз, усредняем результат
    double sequential_mult_time = 0;
    //for (int i = 0; i < 12; ++i) {
    for (int i = 0; i < 1; ++i) {
        clock_start = clock();
        sequential_mult(h_A, h_B, h_C, N, M, K);
        clock_end = clock();
        sequential_mult_time += (float)(clock_end - clock_start) / CLOCKS_PER_SEC;
    }
    //sequential_mult_time /= 12;

    // Создание обработчиков событий
    cudaEvent_t start, stop;
    float gpuTime = 0;

    SAFE_CALL(cudaEventCreate(&start),
        "Cannot create CUDA start event: %s\n");

    SAFE_CALL(cudaEventCreate(&stop),
        "Cannot create CUDA end event: %s\n");

    // устанавливаем размеры блока - 32х32 - больше нельзя
    dim3 block_size(32, 32, 1);
    // устанавливаем размеры грида - количество блоков устанавливаем таким, чтобы размер грида в нитях позволял вместить в себя матрицу целиком
    dim3 grid_size((nr_rows_C + 31) / 32, (nr_cols_C + 31) / 32, 1);

    // выполняем вычисления на GPU 12 раз, усредняем
    double GPU_mult_time = 0;
    for (int i = 0; i < 12; ++i) {
        // установка точки старта
        SAFE_CALL(cudaEventRecord(start, 0),
            "Cannot record CUDA event: %s\n");
        // запуск ядра
        addKernel << <grid_size, block_size >> > (d_C, d_A, d_B, N, M, K);

        cudaError_t cuerr = cudaGetLastError();
        if (cuerr != cudaSuccess)
        {
            fprintf(stderr, "Cannot launch CUDA kernel: %s\n",
                cudaGetErrorString(cuerr));
            return 0;
        }

        // синхронизация устройств
        SAFE_CALL(cudaDeviceSynchronize(),
            "Cannot synchronize CUDA kernel: %s\n");
        // установка точки окончания
        SAFE_CALL(cudaEventRecord(stop, 0),
            "Cannot record CUDA event: %s\n");

        SAFE_CALL(cudaDeviceSynchronize(),
            "Cannot synchronize CUDA kernel: %s\n");
        // расчет времени
        SAFE_CALL(cudaEventElapsedTime(&gpuTime, start, stop), "Cannot measure time: %s\n");
        GPU_mult_time += gpuTime / 1000;
    }

    GPU_mult_time /= 12;

    // копирование результата на хост
    SAFE_CALL(cudaMemcpy(h_C1, d_C, nr_rows_C * nr_cols_C * sizeof(double), cudaMemcpyDeviceToHost),
        "Cannot copy C array from device to host: %s\n");

    // оценивает эквивалентность результатов вычисления матрицы на центральном и графическом процессорах
    bool equiv_res = true;
    for (int i = 0; i < nr_rows_C * nr_cols_C; ++i) {
        if (h_C[i] != h_C1[i]) {
            equiv_res = false;
            break;
        }
    }

    if (equiv_res) {
        printf("SUCCESS: results on host (sequential) and on device (parallel) are the same\n\n");
    }
    else {
        printf("FAILURE: results on host (sequential) and on device (parallel) are NOT the same\n\n");
    }

    // выводим фрагменты результирующих матриц на хосте и устройстве
    printf("first 5x5 block of host matrix:\n");
    for (int i = 0; i < std::min(nr_rows_C, 5); ++i) {
        for (int j = 0; j < std::min(nr_cols_C, 5); ++j) {
            int ij = IDX2C(i, j, nr_rows_C);
            std::cout << std::setw(6) << std::fixed << std::setprecision(3) << h_C[ij] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    printf("first 5x5 block of device matrix:\n");
    for (int i = 0; i < std::min(nr_rows_C, 5); ++i) {
        for (int j = 0; j < std::min(nr_cols_C, 5); ++j) {
            int ij = IDX2C(i, j, nr_rows_C);
            std::cout << std::setw(6) << std::fixed << std::setprecision(3) << h_C1[ij] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    // освобождаем память на GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // освобождаем память на CPU
    free(h_A);
    free(h_B);
    free(h_C);

    // вывод информации о времени выполнения
    printf("CPU-GPU transfer time: %.9f seconds\n", CPU_GPU_transfer_time);
    printf("parallel GPU+transfer time: %.9f seconds\n", CPU_GPU_transfer_time + GPU_mult_time);
    printf("parallel time (without transfer): %.9f seconds\n", GPU_mult_time);
    printf("sequential time (without generation): %.9f seconds\n", sequential_mult_time);

    printf("CPU acceleration (considering trasferring): %.9f times\n", (CPU_GPU_transfer_time + GPU_mult_time) / sequential_mult_time);
    printf("GPU acceleration (considering trasferring): %.9f times\n",  sequential_mult_time / (CPU_GPU_transfer_time + GPU_mult_time));
    printf("CPU acceleration (considering computing performance only): %.9f times\n", GPU_mult_time / sequential_mult_time);
    printf("GPU-acceleration (considering computing performance only): %.9f times\n\n", sequential_mult_time / GPU_mult_time);
    return 0;
}