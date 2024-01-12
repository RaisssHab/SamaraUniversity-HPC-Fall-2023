#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>

// макрос, который используется с функциями CUDA во избежание повторения кода проверки успешности результата функции
#define SAFE_CALL(CallInstruction, error_str, return_type) { \
    cudaError_t cuerr = CallInstruction; \
    if (cuerr != cudaSuccess) { \
        fprintf(stderr, error_str, cudaGetErrorString(cuerr)); \
        return return_type; \
    } \
}

/*
    Функция заполнения массива значенияи с выбором типа генерации - на центральном или графическом процессоре.
*/
void fill_rand(double *input, int size, bool isCPU = false) {
    // инициализация генератора псевдо-случайных чисел
    curandGenerator_t prng;
    if (!isCPU) {
        // генерация на GPU
        curandCreateGenerator(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    }
    else {
        // генерация на CPU
        curandCreateGeneratorHost(&prng, CURAND_RNG_PSEUDO_DEFAULT);
    }
    // установка seed 
    curandSetPseudoRandomGeneratorSeed(prng, (unsigned long long) clock());

    // заполнение случайными числами - равномерное распределение
    curandGenerateUniformDouble(prng, input, size);
    curandDestroyGenerator(prng);
}

// генерация на GPU
void GPU_fill_rand(double* input, int size) {
    fill_rand(input, size);
}

// генерация на CPU
void CPU_fill_rand(double* input, int size) {
    fill_rand(input, size, true);
}

// вычисление суммы массива на CPU
double vectorSumCPU(double *input, int size) {
    double sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += input[i];
    }
    return sum;
}

// ядро для вычисления суммы с глобальной памятью
__global__ void vectorSumGlobal(double *input, double *output, int size)
{
    
    int blockStart = blockIdx.x * blockDim.x;
    int thread_x = blockIdx.x * blockDim.x + threadIdx.x;
    int i = threadIdx.x;

    if (thread_x < size) {
        output[thread_x] = input[thread_x];
    }

    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * i;
        if ((index + s < blockDim.x) && (blockStart + index + s < size)) {
            output[blockStart + index] += output[blockStart + index + s];
        }
        __syncthreads();
    }

    if (i == 0) {
        output[blockIdx.x] = output[blockStart];
    }
}

// функция для вычисления суммы с глобальной памятью
void launchVectorSumGlobal(double* input, int size) {
    while (size > 1) {
        int gridSize = (size + 1023) / 1024;
        vectorSumGlobal << < gridSize, 1024 >> > (input, input, size);
        cudaError_t cuerr = cudaGetLastError();
        if (cuerr != cudaSuccess)
        {
            fprintf(stderr, "Cannot launch CUDA kernel: %s\n",
                cudaGetErrorString(cuerr));
            return;
        }

        // синхронизация устройств
        SAFE_CALL(cudaDeviceSynchronize(),
            "Cannot synchronize CUDA kernel: %s\n");
        size = gridSize;
    }
}

// ядро для вычисления суммы с shared памятью, с конфликтами
__global__ void vectorSumShared(double* input, double* output, int size) {
    extern __shared__ double sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        sdata[tid] = input[i];
    }
    else {
        sdata[tid] = 0;
        return;
    }
    __syncthreads();
    
    // do reduction in shared mem
    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid;
        if (index + s < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) output[blockIdx.x] = sdata[0];
    //output[i] = sdata[tid];
}

// функция для вычисления суммы с shared памятью, с конфликтами
void launchVectorSumShared(double* input, int size) {
    while (size > 1) {
        int gridSize = (size + 1023) / 1024;
        vectorSumShared << < gridSize, 1024, 1024 * sizeof(double) >> > (input, input, size);
        cudaError_t cuerr = cudaGetLastError();
        if (cuerr != cudaSuccess)
        {
            fprintf(stderr, "Cannot launch CUDA kernel: %s\n",
                cudaGetErrorString(cuerr));
            return;
        }

        // синхронизация устройств
        SAFE_CALL(cudaDeviceSynchronize(),
            "Cannot synchronize CUDA kernel: %s\n");
        size = gridSize;
    }
}

// функция для вычисления суммы с shared памятью, с конфликтами, порционно, с обменом с CPU в процессе
void launchVectorSumSharedPortional(double* input, int size) {
    int portion_size = 1000000;
    double res = 0;
    double intermediate_res = 0;
    for (int i = 0; i < size; i += portion_size) {
        double* intermediate_input = input + i;
        int intermediate_size = ((size - i) > portion_size) ? portion_size : (size - i);
        //std::cout << intermediate_size << "\n";
        launchVectorSumShared(intermediate_input, intermediate_size);
        SAFE_CALL(cudaMemcpy(&intermediate_res, intermediate_input, sizeof(double), cudaMemcpyDeviceToHost),
            "Cannot copy C array from device to host: %s\n");
        res += intermediate_res;
    }
    SAFE_CALL(cudaMemcpy(input, &res, sizeof(double), cudaMemcpyHostToDevice),
        "Cannot copy C array from host to device: %s\n");
}

// функция для вычисления суммы с shared памятью, с конфликтами, порционно, без обмена с CPU в процессе
void launchVectorSumSharedPortionalNoCPU(double* input, int size) {
    int portion_size = 1000000;
    int chunks = (size + portion_size - 1) / portion_size;

    int cur_chunk = 0;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        int i = chunk * portion_size;
        double* intermediate_input = input + i;
        int intermediate_size = ((size - i) > portion_size) ? portion_size : (size - i);
        //std::cout << intermediate_size << "\n";
        launchVectorSumShared(intermediate_input, intermediate_size);
        SAFE_CALL(cudaMemcpy(input + chunk, intermediate_input, sizeof(double), cudaMemcpyDeviceToDevice),
            "Cannot copy C array from device to host: %s\n");
    }
    if (chunks > 1) {
        launchVectorSumSharedPortionalNoCPU(input, chunks);
    }
}

int main(int argc, char** argv)
{
    int n;
    std::stringstream ss;
    for (int i = 1; i < argc; ++i) {
        ss << argv[i];
    }
    ss >> n;
    if (!ss) {
        std::cout << "incorrect input when reading size\n";
        return 1;
    }
    if (n < 1 || n > 1000000) {
        std::cout << "incorrect input: n must be in range (1, 1000000)\n";
        return 1;
    }
    
    double* input = (double*)malloc(n * sizeof(double));
    double* d_input;
    cudaMalloc(&d_input, n * sizeof(double));

    // заполнение случайными числами
    CPU_fill_rand(input, n);

    double clock_start, clock_end;
    // вычисляем результирующую матрицу последовательно 12 раз, усредняем результат
    double CPU_time = 0;
    double CPU_res = 0;
    for (int i = 0; i < 12; ++i) {
        clock_start = clock();
        CPU_res = vectorSumCPU(input, n);
        clock_end = clock();
        CPU_time += (double)(clock_end - clock_start) / CLOCKS_PER_SEC;
    }
    CPU_time /= 12;
    
    auto GPU_kernels = { launchVectorSumGlobal, launchVectorSumShared, launchVectorSumSharedPortional, launchVectorSumSharedPortionalNoCPU }; //, launchVectorSumShared };


    for (auto GPU_kernel : GPU_kernels) {
        if (GPU_kernel == launchVectorSumGlobal) {
            printf("GPU, Global Memory\n\n");
        }
        else if (GPU_kernel == launchVectorSumShared) {
            printf("GPU, Shared Memory, basic\n\n");
        }
        else if (GPU_kernel == launchVectorSumSharedPortional) {
            printf("GPU, Shared Memory, portional\n\n");
        }
        else if (GPU_kernel == launchVectorSumSharedPortionalNoCPU) {
            printf("GPU, Shared Memory, portional, no CPU\n\n");
        }

        // время копирования с CPU на GPU
        double CPU_GPU_transfer_time = 0;
       
        // установка точки старта
        clock_start = clock();
        cudaMemcpy(d_input, input, n * sizeof(double), cudaMemcpyHostToDevice);

        // установка точки завершения
        clock_end = clock();
        CPU_GPU_transfer_time = (clock_end - clock_start) / CLOCKS_PER_SEC;

        // Синхронизация устройств
        SAFE_CALL(cudaDeviceSynchronize(),
            "Cannot synchronize CUDA kernel: %s\n", 1);

        // Создание обработчиков событий
        cudaEvent_t start, stop;
        float gpuTime = 0;

        SAFE_CALL(cudaEventCreate(&start),
            "Cannot create CUDA start event: %s\n", 1);

        SAFE_CALL(cudaEventCreate(&stop),
            "Cannot create CUDA end event: %s\n", 1);

        // выполняем вычисления на GPU 12 раз, усредняем
        double GPU_time = 0;
        for (int i = 0; i < 12; ++i) {
            cudaMemcpy(d_input, input, n * sizeof(double), cudaMemcpyHostToDevice);
            // установка точки старта
            SAFE_CALL(cudaEventRecord(start, 0),
                "Cannot record CUDA event: %s\n", 1);

            GPU_kernel(d_input, n);

            cudaError_t cuerr = cudaGetLastError();
            if (cuerr != cudaSuccess)
            {
                return 1;
            }

            // установка точки окончания
            SAFE_CALL(cudaEventRecord(stop, 0),
                "Cannot record CUDA event: %s\n", 1);

            SAFE_CALL(cudaDeviceSynchronize(),
                "Cannot synchronize CUDA kernel: %s\n", 1);
            // расчет времени
            SAFE_CALL(cudaEventElapsedTime(&gpuTime, start, stop), "Cannot measure time: %s\n", 1);
            GPU_time += gpuTime / 1000;
        }

        GPU_time /= 12;

        double GPU_res = 0;
        // копирование результата на хост
        SAFE_CALL(cudaMemcpy(&GPU_res, d_input, sizeof(double), cudaMemcpyDeviceToHost),
            "Cannot copy C array from device to host: %s\n");

        /*
        double* GPU_arr = new double[n];
        cudaMemcpy(GPU_arr, d_input, n * sizeof(double), cudaMemcpyDeviceToHost);
        std::cout << "source array:\n";
        for (int i = 0; i < n; ++i) {
            std::cout << input[i] << " ";
        }
        std::cout << "\n\n";
        std::cout << "processed array:\n";
        for (int i = 0; i < n; ++i) {
            std::cout << GPU_arr[i] << " ";
        }
        std::cout << "\n\n";
        delete GPU_arr;
        */
        // оценивает эквивалентность результатов вычисления матрицы на центральном и графическом процессорах
        bool equiv_res = (GPU_res == CPU_res);

        if (equiv_res) {
            printf("SUCCESS: results on host (CPU) and on device (GPU) are the same\n\n");
        }
        else {
            printf("FAILURE: results on host (CPU) and on device (GPU) are NOT the same\n\n");
        }

        // выводим фрагменты результирующих матриц на хосте и устройстве
        printf("host result: %.9f\n", CPU_res);
        printf("device result: %.9f\n", GPU_res);

        // вывод информации о времени выполнения
        printf("CPU-GPU transfer time: %.9f seconds\n", CPU_GPU_transfer_time);
        printf("parallel GPU+transfer time: %.9f seconds\n", CPU_GPU_transfer_time + GPU_time);
        printf("parallel time (without transfer): %.9f seconds\n", GPU_time);
        printf("sequential time (without generation): %.9f seconds\n", CPU_time);

        printf("CPU acceleration (considering trasferring): %.9f times\n", (CPU_GPU_transfer_time + GPU_time) / CPU_time);
        printf("GPU acceleration (considering trasferring): %.9f times\n", CPU_time / (CPU_GPU_transfer_time + GPU_time));
        printf("CPU acceleration (considering computing performance only): %.9f times\n", GPU_time / CPU_time);
        printf("GPU-acceleration (considering computing performance only): %.9f times\n\n", CPU_time / GPU_time);
    }

    // освобождаем память на GPU
    cudaFree(d_input);

    // освобождаем память на CPU
    free(input);
    return 0;
}