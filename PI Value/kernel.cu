#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand.h>
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

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
void fill_rand(double* input, int size, bool isCPU = false) {
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

// вычисляем на ЦП с явным if вместо sum += (v < 1);
double CPU_calc_with_if(double* xs, double* ys, int size) {
    double sum = 0;
    for (int i = 0; i < size; ++i) {
        double v = xs[i] * xs[i] + ys[i] * ys[i];
        if (v < 1) {
            ++sum;
        }
    }
    return sum * 4. / size;
}

// вычисляем на ЦП с sum += (v < 1);
double CPU_calc_without_if(double* xs, double* ys, int size) {
    double sum = 0;
    for (int i = 0; i < size; ++i) {
        double v = xs[i] * xs[i] + ys[i] * ys[i];
        sum += (v < 1);
    }
    return sum * 4. / size;
}

// реализация ядра для редукции с shared-памятью из прошлой ЛР
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

// реализация редукции с shared-памятью из прошлой ЛР
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

// реализация ядра для получения точек в единичном радиусе, с явным if
__global__ void kernelWithIf(double* xs, double* ys, int size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) {
        return;
    }
    double x = xs[i];
    double y = ys[i];
    double v = x * x + y * y;
    if (v < 1) {
        xs[i] = 1;
    }
    else {
        xs[i] = 0;
    }
}

// реализация функции с использованием GPU для получения точек в единичном радиусе, с явным if
void GPU_calc_with_if(double* xs, double* ys, int size)
{
    int gridSize = (size + 1023) / 1024;
    kernelWithIf <<< gridSize, 1024>>> (xs, ys, size);
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

    launchVectorSumShared(xs, size);
}

// реализация ядра для получения точек в единичном радиусе, с xs[i] = (v < 1)
__global__ void kernelWithoutIf(double* xs, double* ys, int size) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) {
        return;
    }
    double x = xs[i];
    double y = ys[i];
    double v = x * x + y * y;
    xs[i] = (v < 1);
}

// реализация функции с использованием GPU для получения точек в единичном радиусе, с xs[i] = (v < 1)
void GPU_calc_without_if(double* xs, double* ys, int size) {
    int gridSize = (size + 1023) / 1024;
    kernelWithoutIf << < gridSize, 1024 >> > (xs, ys, size);
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

    launchVectorSumShared(xs, size);
}

// реализация ядра с интеграцией получения точек в единичном радиусе с первым шагом редукции
__global__ void PI_value_reduction_integrated_first_step(double* xs, double* ys, double* output, int size) {
    extern __shared__ double sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        double x = xs[i];
        double y = ys[i];
        double v = x * x + y * y;
        sdata[tid] = (v < 1);
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

// реализация функции с использованием GPU с интеграцией получения точек в единичном радиусе с первым шагом редукции
void launch_PI_value_reduction_integrated(double* xs, double* ys, int size) {
    int gridSize = (size + 1023) / 1024;
    PI_value_reduction_integrated_first_step << < gridSize, 1024, 1024 * sizeof(double) >> > (xs, ys, xs, size);
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
    
    launchVectorSumShared(xs, size);
}

// подсчет времени CPU
template<typename T>
void get_CPU_time(T call, double& CPU_res, double& CPU_time) {
    double clock_start, clock_end;

    CPU_time = 0;
    CPU_res = 0;
    for (int i = 0; i < 12; ++i) {
        clock_start = clock();
        CPU_res = call();
        clock_end = clock();
        CPU_time += (double)(clock_end - clock_start) / CLOCKS_PER_SEC;
    }
    CPU_time /= 12;
}

// подсчет времени GPU
template<typename T>
void get_GPU_time(T call, const double* xs_orig, const double* ys_orig, const int size, double& GPU_res, double& GPU_time) {
    // время копирования с CPU на GPU
    double CPU_GPU_transfer_time = 0;

    double* xs, *ys;
    cudaMalloc(&xs, size * sizeof(double));
    cudaMalloc(&ys, size * sizeof(double));

    // Синхронизация устройств
    SAFE_CALL(cudaDeviceSynchronize(),
        "Cannot synchronize CUDA kernel: %s\n");

    // Создание обработчиков событий
    cudaEvent_t start, stop;
    float gpuTime = 0;

    SAFE_CALL(cudaEventCreate(&start),
        "Cannot create CUDA start event: %s\n");

    SAFE_CALL(cudaEventCreate(&stop),
        "Cannot create CUDA end event: %s\n");

    // выполняем вычисления на GPU 12 раз, усредняем
    GPU_time = 0;
    for (int i = 0; i < 12; ++i) {

        cudaMemcpy(xs, xs_orig, size * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(ys, ys_orig, size * sizeof(double), cudaMemcpyDeviceToDevice);

        // установка точки старта
        SAFE_CALL(cudaEventRecord(start, 0),
            "Cannot record CUDA event: %s\n");

        call(xs, ys, size);

        cudaError_t cuerr = cudaGetLastError();
        if (cuerr != cudaSuccess)
        {
            return;
        }

        // установка точки окончания
        SAFE_CALL(cudaEventRecord(stop, 0),
            "Cannot record CUDA event: %s\n");

        SAFE_CALL(cudaDeviceSynchronize(),
            "Cannot synchronize CUDA kernel: %s\n");
        // расчет времени
        SAFE_CALL(cudaEventElapsedTime(&gpuTime, start, stop), "Cannot measure time: %s\n");
        GPU_time += gpuTime / 1000;
    }

    GPU_time /= 12;

    GPU_res = 0;
    // копирование результата на хост
    SAFE_CALL(cudaMemcpy(&GPU_res, xs, sizeof(double), cudaMemcpyDeviceToHost),
        "Cannot copy C array from device to host: %s\n");

    cudaFree(xs);
    cudaFree(ys);
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
    if (n < 1) {
        std::cout << "incorrect input: n must be >= 1\n";
        return 1;
    }
    std::cout << n << "\n";
    double *xs = (double*)malloc(n * sizeof(double));
    double* ys = (double*)malloc(n * sizeof(double));

    double *d_xs, *d_ys;
    cudaMalloc(&d_xs, n * sizeof(double));
    cudaMalloc(&d_ys, n * sizeof(double));

    // заполнение случайными числами
    GPU_fill_rand(d_xs, n);
    GPU_fill_rand(d_ys, n);

    cudaMemcpy(xs, d_xs, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ys, d_ys, n * sizeof(double), cudaMemcpyDeviceToHost);

    std::vector<double(*)(double*, double*, int)> CPU_implementations = {CPU_calc_with_if, CPU_calc_without_if};

    std::vector<std::string> CPU_names = { "CPU calculation with if", "CPU calculation without if" };
    std::vector<double> CPU_times, CPU_results;

    for (int i = 0; i < CPU_implementations.size(); ++i) {
        std::cout << CPU_names[i] << "\n";
        auto CPU_impl = CPU_implementations[i];
        double CPU_time, CPU_res;
        get_CPU_time([&]() { return CPU_impl(xs, ys, n); }, CPU_res, CPU_time);
        CPU_times.push_back(CPU_time);
        CPU_results.push_back(CPU_res);
    }

    std::vector<void(*)(double*, double*, int)> GPU_implementations = { GPU_calc_with_if, 
        GPU_calc_without_if, launch_PI_value_reduction_integrated };
    std::vector<std::string> GPU_names = { "GPU calculation with if", 
        "GPU calculation without if", "GPU calculation with radius integrated into reduction"};
    std::vector<double> GPU_times, GPU_results;

    for (int i = 0; i < GPU_implementations.size(); ++i) {
        std::cout << GPU_names[i] << "\n";
        auto GPU_implementation = GPU_implementations[i];
        double GPU_time, GPU_res;
        get_GPU_time([&](double* xs, double* ys, int size) { return GPU_implementation(xs, ys, n); }, 
            d_xs, d_ys, n, GPU_res, GPU_time);

        cudaError_t cuerr = cudaGetLastError();
        if (cuerr != cudaSuccess)
        {
            return 1;
        }

        GPU_times.push_back(GPU_time);
        GPU_results.push_back(GPU_res);
    }

    for (int i = 0; i < CPU_implementations.size(); ++i) {
        double CPU_res = CPU_results[i];
        double CPU_time = CPU_times[i];
        for (int j = 0; j < GPU_implementations.size(); ++j) {
            std::cout << "comparison: " << CPU_names[i] << " vs. " << GPU_names[j] << "\n\n";
            double GPU_res = GPU_results[j] * 4. / n;
            double GPU_time = GPU_times[j];

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
            printf("parallel time (without transfer): %.9f seconds\n", GPU_time);
            printf("sequential time (without generation): %.9f seconds\n", CPU_time);

            printf("CPU acceleration (considering computing performance only): %.9f times\n", GPU_time / CPU_time);
            printf("GPU-acceleration (considering computing performance only): %.9f times\n\n", CPU_time / GPU_time);
        }
    }

    // освобождаем память на GPU
    cudaFree(d_xs);
    cudaFree(d_ys);

    // освобождаем память на CPU
    free(xs);
    free(ys);
    return 0;
}