#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <sstream>
#include <fstream>

/*
    Функция заполнения матрицы значенияи с выбором типа генерации - на центральном или графическом процессоре.
*/
void fill_rand(double* A, int nr_rows_A, int nr_cols_A, bool isCPU = false) {
    // инициализация генератора псевдо-случайных чисел
    curandGenerator_t prng;
    if (!isCPU) {
        // генерация на CPU
        curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    }
    else {
        // генерация на GPU
        curandCreateGeneratorHost(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    }
    // установка seed 
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    // заполнение случайными числами - равномерное распределение
    curandGenerateUniformDouble(prng, A, nr_rows_A * nr_cols_A);
    curandDestroyGenerator(prng);
}

// генерация на GPU
void GPU_fill_rand(double* A, int nr_rows_A, int nr_cols_A) {
    fill_rand(A, nr_rows_A, nr_cols_A);
}

// генерация на CPU
void CPU_fill_rand(double* A, int nr_rows_A, int nr_cols_A) {
    fill_rand(A, nr_rows_A, nr_cols_A, true);
}

// макрос index-to-c, преобразующий номер строки и столбца элемента матрицы в индекс массива, который им соответствует (разложение по столбцам) 
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

int main(int argc, char* argv[])
{
    // вспомогательная информация о программе
    if (argc == 1) {
        std::cout << "Usage: program_name N M K\n";
        std::cout << "N, M, K characterize the dimensions of the matrices: the first matrix is NxM matrix, the second matrix is MxK matrix, therefore the resulting matrix is NxK matrix.";
        return 0;
    }
    // проверка, что даны три аргумента
    if (argc != 4) {
        std::cout << "please provide the params: N, M, K";
        return -1;
    }
    // чтение в и из строкового потока параметров N, M, K
    std::stringstream ss;
    for (int i = 1; i < 4; ++i) {
        ss << argv[i] << " ";
    }
    int N, M, K;
    for (int* param : { &N, &M, &K }) {
        std::string s;
        ss >> (*param);
        if (!ss || *param <= 0) {
            std::cout << "one of the provided values is incorrect (or is <= 0), try again";
            return 1;
        }
    }
    // выделение памяти
    int nr_rows_A, nr_cols_A, nr_rows_B, nr_cols_B;
    nr_rows_A = N;
    nr_cols_A = nr_rows_B = M;
    nr_cols_B = K;

    double* h_A = (double*)malloc(nr_rows_A * nr_cols_A * sizeof(double));
    double* h_B = (double*)malloc(nr_rows_B * nr_cols_B * sizeof(double));

    // заполнение случайными числами
    CPU_fill_rand(h_A, nr_rows_A, nr_cols_A);
    CPU_fill_rand(h_B, nr_rows_B, nr_cols_B);

    /* вывод данных в формате
        N M K
        *матрица А размерности NxM, представленная в виде разложения по строкам*
        *матрица B размерности MxK, представленная в виде разложения по строкам* 
    */
    std::ofstream out("input_data.txt");
    out << N << " " << M << " " << K << "\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < M; ++j) {
            out << h_A[IDX2C(i, j, N)] << " ";
        }
        out << "\n";
    }
    out << "\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            out << h_B[IDX2C(i, j, M)] << " ";
        }
        out << "\n";
    }
}
