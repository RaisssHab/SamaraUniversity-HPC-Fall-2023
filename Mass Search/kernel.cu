#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand.h>
#include <curand_kernel.h>
#include <assert.h>

#include <stdio.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <math.h>
#include <algorithm>

class Formatter
{
public:
    Formatter() {}
    ~Formatter() {}

    template <typename Type>
    Formatter& operator << (const Type& value)
    {
        stream_ << value;
        return *this;
    }

    std::string str() const { return stream_.str(); }
    operator std::string() const { return stream_.str(); }

    enum ConvertToString
    {
        to_str
    };
    std::string operator >> (ConvertToString) { return stream_.str(); }

private:
    std::stringstream stream_;

    Formatter(const Formatter&);
    Formatter& operator = (Formatter&);
};

class CudaException : public std::runtime_error
{
public:
    CudaException(cudaError_t cuerr, std::string additional_info = "") : runtime_error(Formatter() << cudaGetErrorString(cuerr) << "; \n" << "additional information: " << additional_info) {
        std::cout << cudaGetErrorString(cuerr) << "; \n" << "additional information: " << additional_info << "\n";
    }
};

// макрос, который используется с функциями CUDA во избежание повторения кода проверки успешности результата функции
#define SAFE_CALL(CallInstruction, error_str) { \
    cudaError_t cuerr = CallInstruction; \
    if (cuerr != cudaSuccess) { \
        throw CudaException(cuerr, error_str); \
    } \
}

#define SAFE_CALL(CallInstruction) { \
    cudaError_t cuerr = CallInstruction; \
    if (cuerr != cudaSuccess) { \
        throw CudaException(cuerr); \
    } \
}

#define SAFE_KERNEL_CALL(KernelInstruction) { \
    KernelInstruction; \
    SAFE_CALL(cudaDeviceSynchronize()); \
}

class CudaSafeFunc {
public:
    template<class T>
    static void cudaMalloc(T** dev_ptr, size_t size) {
        SAFE_CALL(::cudaMalloc(dev_ptr, size));
    }

    static void cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
        SAFE_CALL(::cudaMemcpy(dst, src, count, kind));
    }
    static void cudaDeviceSynchronize() {
        SAFE_CALL(::cudaDeviceSynchronize());
    }
    static void cudaEventCreate(cudaEvent_t* event) {
        SAFE_CALL(::cudaEventCreate(event));
    }
    static void cudaEventRecord(cudaEvent_t event, cudaStream_t stream) {
        SAFE_CALL(::cudaEventRecord(event, stream));
    }
    static void cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
        SAFE_CALL(::cudaEventElapsedTime(ms, start, end));
    }
    static void cudaFree(void* devPtr) {
        SAFE_CALL(::cudaFree(devPtr));
    }
};

__global__ void setup_kernel(curandState* state) {

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(1234, idx, 0, &state[idx]);
}


template <class T>
__global__ void generate_kernel(curandState* my_curandstate, int n, T max_rand_int, 
                                T min_rand_int, T* result, int result_size) {

    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    n = (idx + n <= result_size) ? n : (result_size - idx);

    for (int count = 0; count < n; ++count) {
        float myrandf = curand_uniform(my_curandstate + idx);
        myrandf *= (max_rand_int - min_rand_int + 0.999999);
        myrandf += min_rand_int;
        T myrand = (T)truncf(myrandf);

        assert(myrand <= max_rand_int);
        assert(myrand >= min_rand_int);

        result[idx + count] = myrand;
    }
}

template<class T>
void generate_general(T max_rand_int, T min_rand_int, 
                    T* d_result, T* h_result, int result_size) {
    int thread_num = std::min((int)1e6, result_size);
    int thread_area_size = (result_size + thread_num - 1) / thread_num;
    int grid_size = (result_size + 1023) / 1024;

    curandState* d_state;
    cudaMalloc(&d_state, thread_num * sizeof(curandState));

    setup_kernel<<<grid_size, 1024 >>> (d_state);
    CudaSafeFunc::cudaDeviceSynchronize();

    generate_kernel <<<grid_size, 1024>>> (d_state, thread_area_size, max_rand_int, min_rand_int, d_result, result_size);
    CudaSafeFunc::cudaDeviceSynchronize();
    
    CudaSafeFunc::cudaMemcpy(h_result, d_result, result_size * sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost);
    CudaSafeFunc::cudaFree(d_state);
}

void generate_sizes(int max_rand_int, int min_rand_int,
                    int* d_result, int* h_result, int result_size) {
    generate_general(max_rand_int, min_rand_int, d_result, h_result, result_size);
}

void generate_string(char* d_result, char* h_result, int result_size) {
    /*
    int* d_result_int, *h_result_int;
    int int_size = (sizeof(char) * result_size + sizeof(int) - 1) / sizeof(int);
    SAFE_CALL(cudaMalloc(&d_result_int, int_size * sizeof(int)),
        "Cannot malloc to d_result_int: %s\n");
    h_result_int = new int[int_size];

    generate_general(INT_MAX, 0, d_result_int, h_result_int, int_size);

    SAFE_CALL(cudaMemcpy(d_result, d_result_int, sizeof(char) * result_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice),
        "Cannot copy d_result_int to d_result: %s\n");
    SAFE_CALL(cudaMemcpy(h_result, h_result_int, sizeof(char) * result_size, cudaMemcpyKind::cudaMemcpyHostToHost),
        "Cannot copy h_result_int to h_result: %s\n");

    cudaFree(d_result_int);
    delete h_result_int;
    */

    generate_general((unsigned char)255, (unsigned char)0, (unsigned char*)d_result, (unsigned char*)h_result, result_size);
}

__global__ void vectorSumShared(int* input, int* output, int size) {
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

void launchVectorSumShared(int* input, int size) {
    while (size > 1) {
        int gridSize = (size + 1023) / 1024;
        //int d_size;
        //cudaMemcpy(&d_size, &size, sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);
        vectorSumShared <<< gridSize, 1024, 1024 * sizeof(double) >>> (input, input, size);
        CudaSafeFunc::cudaDeviceSynchronize();

        size = gridSize;
    }
}

template<typename T>
void get_CPU_time(T call, double& CPU_time) {
    double clock_start, clock_end;

    CPU_time = 0;
    for (int i = 0; i < 12; ++i) {
        clock_start = clock();
        call();
        clock_end = clock();
        CPU_time += (double)(clock_end - clock_start) / CLOCKS_PER_SEC;
    }
    CPU_time /= 12;
}

template<typename T>
void get_GPU_time(T call, const double* xs_orig, const double* ys_orig, const int size, double& GPU_res, double& GPU_time) {
    // время копирования с CPU на GPU
    double CPU_GPU_transfer_time = 0;

    double* xs, * ys;
    cudaMalloc(&xs, size * sizeof(double));
    cudaMalloc(&ys, size * sizeof(double));

    CudaSafeFunc::cudaDeviceSynchronize();

    // Создание обработчиков событий
    cudaEvent_t start, stop;
    float gpuTime = 0;

    CudaSafeFunc::cudaEventCreate(&start);
    CudaSafeFunc::cudaEventCreate(&stop);

    // выполняем вычисления на GPU 12 раз, усредняем
    GPU_time = 0;
    for (int i = 0; i < 12; ++i) {

        CudaSafeFunc::cudaMemcpy(xs, xs_orig, size * sizeof(double), cudaMemcpyDeviceToDevice);
        CudaSafeFunc::cudaMemcpy(ys, ys_orig, size * sizeof(double), cudaMemcpyDeviceToDevice);

        // установка точки старта

        CudaSafeFunc::cudaEventRecord(start, 0);

        call(xs, ys, size);
        /*
        cudaError_t cuerr = cudaGetLastError();
        if (cuerr != cudaSuccess)
        {
            return;
        }
        */

        // установка точки окончания
        CudaSafeFunc::cudaEventRecord(stop, 0);
        CudaSafeFunc::cudaDeviceSynchronize();

        // расчет времени
        CudaSafeFunc::cudaEventElapsedTime(&gpuTime, start, stop);
        GPU_time += gpuTime / 1000;
    }

    GPU_time /= 12;

    GPU_res = 0;
    // копирование результата на хост
    CudaSafeFunc::cudaMemcpy(&GPU_res, xs, sizeof(double), cudaMemcpyDeviceToHost);

    CudaSafeFunc::cudaFree(xs);
    CudaSafeFunc::cudaFree(ys);
}



class InputException : public std::runtime_error
{
public:
    InputException(const std::string what) : runtime_error(what) {}
};


class Params {
public:
    const int h, substr_count, min_substr_size, max_substr_size, how_to_search;

    Params(int h, int substr_count, int min_substr_size, int max_substr_size, int how_to_search) :
        h(h), substr_count(substr_count), min_substr_size(min_substr_size), max_substr_size(max_substr_size), how_to_search(how_to_search) {}
};

class InputReader {
public:
    static Params read(int argc, char** argv) {
        if (argc != 6) {
            throw InputException(Formatter() << "expected 5 arguments but got " << argc << " instead");
        }
        int h, substr_count, min_substr_size, max_substr_size, how_to_search;
        std::stringstream ss;

        for (int i = 1; i < argc; ++i) {
            ss << argv[i] << " ";
        }

        ss >> h;
        if (!ss) {
            throw InputException(Formatter() << "incorrect input when reading h");
        }
        if (h < 1) {
            throw InputException(Formatter() << "incorrect input: h must be >= 1");
        }

        ss >> substr_count;

        if (!ss) {
            throw InputException(Formatter() << "incorrect input when reading substring count");
        }
        if (substr_count < 1) {
            throw InputException(Formatter() << "incorrect input: substring must be >= 1");
        }

        ss >> min_substr_size;

        if (!ss) {
            throw InputException(Formatter() << "incorrect input when reading minimum substring size");
        }
        if (min_substr_size < 1) {
            throw InputException(Formatter() << "incorrect input: minimum substring size must be >= 1");
        }
        if (min_substr_size >= h) {
            throw InputException(Formatter() << "incorrect input: minimum substring size must be < h");
        }

        ss >> max_substr_size;

        if (!ss) {
            throw InputException(Formatter() << "incorrect input when reading maximum substring size");
        }
        if (max_substr_size < 1) {
            throw InputException(Formatter() << "incorrect input: maximum substring size must be >= 1");
        }
        if (max_substr_size >= h) {
            throw InputException(Formatter() << "incorrect input: maximum substring size must be < h");
        }
        if (max_substr_size < min_substr_size) {
            throw InputException(Formatter() << "incorrect input: maximum substring size must be >= minimum substring size");
        }

        ss >> how_to_search;

        if (!ss) {
            throw InputException(Formatter() << "incorrect input when reading how to search");
        }
        if (how_to_search != 1 && how_to_search != 2) {
            throw InputException(Formatter() << "incorrect input: how to search must be either 1 or 2");
        }

        return Params(h, substr_count, min_substr_size, max_substr_size, how_to_search);
    }
};

class InputData {
public:
    char const * const H;
    int const h;
    int const * const sizes;
    int const * const starts;
    char const * const all_strings;
    int const sizes_sum;
    int const substr_count;

    InputData(char* H, int h, int* sizes, int* starts, char* all_strings, int sizes_sum, int substr_count)
            : H(H), h(h), sizes(sizes), starts(starts), all_strings(all_strings), sizes_sum(sizes_sum), substr_count(substr_count) {}
};

class DataCreatorException : public std::runtime_error
{
public:
    DataCreatorException(char* msg) : runtime_error(msg) {}
};

class DataCreator {
private:
    char* h_H, *d_H;
    int* h_sizes, *d_sizes;
    int* h_starts, *d_starts;
    char* h_all_strings, *d_all_strings;
    int sizes_sum;

    Params const params;
    bool calculated = false;
public:
    DataCreator(Params params) : params(params), h_H(nullptr), h_sizes(nullptr), h_starts(nullptr), h_all_strings(nullptr),
                    d_H(nullptr), d_sizes(nullptr), d_starts(nullptr), d_all_strings(nullptr) {}
    ~DataCreator() {
        delete h_H;
        CudaSafeFunc::cudaFree(d_H);
        delete h_sizes;
        CudaSafeFunc::cudaFree(d_sizes);
        delete h_starts;
        CudaSafeFunc::cudaFree(d_starts);
        delete h_all_strings;
        CudaSafeFunc::cudaFree(d_all_strings);
    }
    void create() {
        if (calculated) {
            throw DataCreatorException("already created");
        }

        h_H = new char[params.h];
        CudaSafeFunc::cudaMalloc(&d_H, params.h * sizeof(char));

        generate_string(d_H, h_H, params.h);

        h_sizes = new int[params.substr_count];
        CudaSafeFunc::cudaMalloc(&d_sizes, params.substr_count * sizeof(int));

        generate_sizes(params.max_substr_size, params.min_substr_size, d_sizes, h_sizes, params.substr_count);

        int* dup_d_sizes;

        CudaSafeFunc::cudaMalloc(&dup_d_sizes, params.substr_count * sizeof(int));
        CudaSafeFunc::cudaMemcpy(dup_d_sizes, d_sizes, params.substr_count * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToDevice);

        launchVectorSumShared(dup_d_sizes, params.substr_count);

        CudaSafeFunc::cudaMemcpy(&sizes_sum, dup_d_sizes, sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);

        CudaSafeFunc::cudaFree(dup_d_sizes);

        h_all_strings = new char[sizes_sum];
        CudaSafeFunc::cudaMalloc(&d_all_strings, sizes_sum * sizeof(char));

        generate_string(d_all_strings, h_all_strings, sizes_sum);

        h_starts = new int[params.substr_count];
        CudaSafeFunc::cudaMalloc(&d_starts, params.substr_count * sizeof(int));

        h_starts[0] = 0;
        for (int i = 1; i < params.substr_count; ++i) {
            h_starts[i] = h_starts[i - 1] + h_sizes[i - 1];
        }

        CudaSafeFunc::cudaMemcpy(d_starts, h_starts, params.substr_count * sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);

        calculated = true;
    }
    InputData forCPU() {
        if (!calculated) {
            throw DataCreatorException("not created yet");
        }
        return InputData(h_H, params.h, h_sizes, h_starts, h_all_strings, sizes_sum, params.substr_count);
    }
    InputData forGPU() {
        if (!calculated) {
            throw DataCreatorException("not created yet");
        }
        return InputData(d_H, params.h, d_sizes, d_starts, d_all_strings, sizes_sum, params.substr_count);
    }
};

template<class Input, class Output>
class Task {
public:
    virtual Output& process(Input& input) = 0;
};

class WorkingMatrix {
private:
    int** working_matrix;
    int rows, cols;
public:
    WorkingMatrix(int** working_matrix, int rows, int cols) 
        : working_matrix(working_matrix), rows(rows), cols(cols) {}

    int** getMatrix() {
        return working_matrix;
    }

    int getRows() {
        return rows;
    }

    int getCols() {
        return cols;
    }
};



class PrehandlerResult {
public:
    InputData& input_data;
    WorkingMatrix& working_matrix;
    int** occurencies_substr;
    int** occurencies_index;
    int* sizes;
public:
    PrehandlerResult(InputData& input_data, WorkingMatrix& working_matrix, int** occurencies_substr, int** occurencies_index, int* sizes) 
        : input_data(input_data), working_matrix(working_matrix), occurencies_substr(occurencies_substr), 
        occurencies_index(occurencies_index), sizes(sizes) {}
};

class AlgorithmPrehandler : public Task<InputData, PrehandlerResult> {};

class AlgorithmPrehandlerCPU : public AlgorithmPrehandler {
private:
    WorkingMatrix* working_matrix;
    PrehandlerResult* prehandler_result;
public:
    PrehandlerResult& process(InputData& input) override {
        int** working_matrix_arr = new int*[input.substr_count];
        for (int i = 0; i < input.substr_count; ++i) {
            working_matrix_arr[i] = new int[input.h];
        }
        for (int i = 0; i < input.substr_count; ++i) {
            for (int j = 0; j < input.h; ++j) {
                working_matrix_arr[i][j] = input.sizes[i];
            }
        }
        int** occurencies_substr = new int*[256];
        int** occurencies_index = new int* [256];
        int* sizes = new int[256];
        for (int i = 0; i < 256; ++i) {
            sizes[i] = 0;
        }

        for (int i = 0; i < input.sizes_sum; ++i) {
            unsigned char c = input.all_strings[i];
            ++sizes[+c];
        }

        for (int i = 0; i < 256; ++i) {
            occurencies_substr[i] = new int[sizes[i]];
            occurencies_index[i] = new int[sizes[i]];
        }

        int* cur_indexes = new int[256];
        for (int i = 0; i < 256; ++i) {
            cur_indexes[i] = 0;
        }

        for (int substr_index = 0; substr_index < input.substr_count; ++substr_index) {
            for (int not_shifted_i = 0; not_shifted_i < input.sizes[substr_index]; ++not_shifted_i) {
                int i = not_shifted_i + input.starts[substr_index];
                unsigned char c = input.all_strings[i];
                int cur_char_index = cur_indexes[c]++;
                occurencies_substr[c][cur_char_index] = substr_index;
                occurencies_index[c][cur_char_index] = not_shifted_i;
                
            }
        }

        /*
        std::cout << "sizes: ";
        for (int i = 0; i < 256; ++i) {
            std::cout << sizes[i] << " ";
        }
        std::cout << "\n";

        for (int i = 0; i < 256; ++i) {
            std::cout << i << ": ";
            for (int j = 0; j < sizes[i]; ++j) {
                std::cout << "(" << occurencies_substr[i][j] << ", " << occurencies_index[i][j] << ") ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
        */

        delete cur_indexes;

        working_matrix = new WorkingMatrix(working_matrix_arr, input.substr_count, input.h);
        prehandler_result = new PrehandlerResult(input, *working_matrix, occurencies_substr, occurencies_index, sizes);
        return *prehandler_result;
    }
};

__global__ void fill_working_matrix(int** working_matrix, const int* sizes, int cols, int rows) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= rows) {
        return;
    }

    if (j >= cols) {
        return;
    }

    int filler = sizes[i];

    working_matrix[i][j] = filler;
}

template<class T>
__global__ void fill_array_with_value(T* arr, T value, int size) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= size) {
        return;
    }
    arr[threadId] = value;
}

__global__ void count_symbols(const char* string, int size, int* sizes) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = gridDim.x * blockDim.x;
    for (int pos = threadId; pos < size; pos += gridSize) {
        unsigned char c = string[pos];
        atomicAdd(&sizes[c], 1);
    }
}

__global__ void fill_occurencies(int** occurencies_substr, int** occurencies_index, int* substr_count, const int* sizes, 
                                const char* all_strings, const int* starts, int* cur_indexes) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= *substr_count) {
        return;
    }

    for (int not_shifted_i = 0; not_shifted_i < sizes[i]; ++not_shifted_i) {
        int shifted_pos = not_shifted_i + starts[i];
        unsigned char c = all_strings[shifted_pos];
        int cur_char_index = atomicAdd(&cur_indexes[+c], 1);
        occurencies_substr[+c][cur_char_index] = i;
        occurencies_index[+c][cur_char_index] = not_shifted_i;

    }

    /*
    for (int substr_index = 0; substr_index < input.substr_count; ++substr_index) {
        for (int not_shifted_i = 0; not_shifted_i < input.sizes[substr_index]; ++not_shifted_i) {
            int i = not_shifted_i + input.starts[substr_index];
            unsigned char c = input.all_strings[i];
            int cur_char_index = cur_indexes[c]++;
            occurencies_substr[c][cur_char_index] = substr_index;
            occurencies_index[c][cur_char_index] = not_shifted_i;

        }
    }
    */
}

class AlgorithmPrehandlerGPU : public AlgorithmPrehandler {
private:
    WorkingMatrix* working_matrix;
    PrehandlerResult* prehandler_result;
public:
    PrehandlerResult& process(InputData& input) override {
        int** working_matrix_arr_h, ** working_matrix_arr_d;
        working_matrix_arr_h = new int* [input.substr_count];
        CudaSafeFunc::cudaMalloc(&working_matrix_arr_d, sizeof(int*) * input.substr_count);

        for (int i = 0; i < input.substr_count; ++i) {
            CudaSafeFunc::cudaMalloc(&working_matrix_arr_h[i], sizeof(int) * input.h);
        }

        CudaSafeFunc::cudaMemcpy(working_matrix_arr_d, working_matrix_arr_h, sizeof(int*) * input.substr_count, cudaMemcpyKind::cudaMemcpyHostToDevice);

        /*
        int** working_matrix_arr = new int* [input.substr_count];
        for (int i = 0; i < input.substr_count; ++i) {
            working_matrix_arr[i] = new int[input.h];
        }
        for (int i = 0; i < input.substr_count; ++i) {
            for (int j = 0; j < input.h; ++j) {
                working_matrix_arr[i][j] = input.sizes[i];
            }
        }
        */

        dim3 grid((input.h + 31) / 32, (input.substr_count + 31) / 32);
        dim3 block(32, 32);

        fill_working_matrix<<<grid, block>>> (working_matrix_arr_d, input.sizes, input.h, input.substr_count);
        CudaSafeFunc::cudaDeviceSynchronize();

        /*
        int** occurencies_substr = new int* [256];
        int** occurencies_index = new int* [256];
        int* sizes = new int[256];
        for (int i = 0; i < 256; ++i) {
            sizes[i] = 0;
        }
        */

        int** occurencies_substr;
        int** occurencies_index;
        CudaSafeFunc::cudaMalloc(&occurencies_substr, sizeof(int*) * 256);
        CudaSafeFunc::cudaMalloc(&occurencies_index, sizeof(int*) * 256);

        int* sizes;
        CudaSafeFunc::cudaMalloc(&sizes, sizeof(int) * 256);
        fill_array_with_value << <1, 256 >> > (sizes, 0, 256);
        CudaSafeFunc::cudaDeviceSynchronize();


        /*
        for (int i = 0; i < input.sizes_sum; ++i) {
            unsigned char c = input.all_strings[i];
            ++sizes[+c];
        }
        */

        count_symbols<<<1, 256>>>(input.all_strings, input.sizes_sum, sizes);
        CudaSafeFunc::cudaDeviceSynchronize();

        /*

        for (int i = 0; i < 256; ++i) {
            occurencies_substr[i] = new int[sizes[i]];
            occurencies_index[i] = new int[sizes[i]];
        }
        */

        int* sizes_h = new int[256];
        CudaSafeFunc::cudaMemcpy(sizes_h, sizes, sizeof(int) * 256, cudaMemcpyKind::cudaMemcpyDeviceToHost);

        int** occurencies_substr_h = new int* [256];
        int** occurencies_index_h = new int* [256];

        for (int i = 0; i < 256; ++i) {
            CudaSafeFunc::cudaMalloc(&(occurencies_substr_h[i]), sizeof(int) * sizes_h[i]);
            CudaSafeFunc::cudaMalloc(&(occurencies_index_h[i]), sizeof(int) * sizes_h[i]);
        }

        CudaSafeFunc::cudaMemcpy(occurencies_substr, occurencies_substr_h, sizeof(int*) * 256, cudaMemcpyKind::cudaMemcpyHostToDevice);
        CudaSafeFunc::cudaMemcpy(occurencies_index, occurencies_index_h, sizeof(int*) * 256, cudaMemcpyKind::cudaMemcpyHostToDevice);

        /*
            int* cur_indexes = new int[256];
            for (int i = 0; i < 256; ++i) {
                cur_indexes[i] = 0;
            }
        */

        int* cur_indexes;
        CudaSafeFunc::cudaMalloc(&cur_indexes, sizeof(int) * 256);

        fill_array_with_value << <1, 256 >> > (cur_indexes, 0, 256);
        CudaSafeFunc::cudaDeviceSynchronize();

        int* substr_count_d;
        CudaSafeFunc::cudaMalloc(&substr_count_d, sizeof(int));
        CudaSafeFunc::cudaMemcpy(substr_count_d, &input.substr_count, sizeof(int), cudaMemcpyKind::cudaMemcpyHostToDevice);
        
        fill_occurencies << <(input.substr_count + 1023) / 1024, 1024 >> > (occurencies_substr, occurencies_index, substr_count_d, input.sizes,
            input.all_strings, input.starts, cur_indexes);
        CudaSafeFunc::cudaDeviceSynchronize();

        /*
        std::cout << "occurencies_substr:\n";
        for (int i = 0; i < 256; ++i) {
            int* occur_part = new int[sizes_h[i]];
            CudaSafeFunc::cudaMemcpy(occur_part, occurencies_substr_h[i], sizeof(int) * sizes_h[i], cudaMemcpyKind::cudaMemcpyDeviceToHost);
            for (int j = 0; j < sizes_h[i]; ++j) {
                std::cout << j << ": " << occur_part[j] << " ";
            }
            std::cout << "\n";
            delete occur_part;
        }

        std::cout << "\noccurencies_index:\n";
        for (int i = 0; i < 256; ++i) {
            int* occur_part = new int[sizes_h[i]];
            CudaSafeFunc::cudaMemcpy(occur_part, occurencies_index_h[i], sizeof(int) * sizes_h[i], cudaMemcpyKind::cudaMemcpyDeviceToHost);
            for (int j = 0; j < sizes_h[i]; ++j) {
                std::cout << j << ": " << occur_part[j] << " ";
            }
            std::cout << "\n";
            delete occur_part;
        }

        int* cur_indexes_h = new int[256];
        CudaSafeFunc::cudaMemcpy(cur_indexes_h, cur_indexes, sizeof(int) * 256, cudaMemcpyKind::cudaMemcpyDeviceToHost);

        std::cout << "\ncur_indexes_h: " << "\n";

        for (int i = 0; i < 256; ++i) {
            std::cout << cur_indexes_h[i] << " ";
        }
        std::cout << "\n";

        delete cur_indexes_h;
        */

        working_matrix = new WorkingMatrix(working_matrix_arr_d, input.substr_count, input.h);
        prehandler_result = new PrehandlerResult(input, *working_matrix, occurencies_substr, occurencies_index, sizes);
        return *prehandler_result;

        /*
        for (int substr_index = 0; substr_index < input.substr_count; ++substr_index) {
            for (int not_shifted_i = 0; not_shifted_i < input.sizes[substr_index]; ++not_shifted_i) {
                int i = not_shifted_i + input.starts[substr_index];
                unsigned char c = input.all_strings[i];
                int cur_char_index = cur_indexes[c]++;
                occurencies_substr[c][cur_char_index] = substr_index;
                occurencies_index[c][cur_char_index] = not_shifted_i;

            }
        }

        delete cur_indexes;

        working_matrix = new WorkingMatrix(working_matrix_arr, input.substr_count, input.h);
        prehandler_result = new PrehandlerResult(input, *working_matrix, occurencies_substr, occurencies_index, sizes);
        return *prehandler_result;
        */
    }
};

class WorkingMatrixHandler : public Task<PrehandlerResult, WorkingMatrix> {

};

class WorkingMatrixHandlerCPU : public WorkingMatrixHandler {
public:
    WorkingMatrix& process(PrehandlerResult& input) override {
        int** working_matrix = input.working_matrix.getMatrix();
        int cols = input.working_matrix.getCols();
        int rows = input.working_matrix.getRows();

        for (int j = 0; j < cols; ++j) {
            unsigned char c = input.input_data.H[j];
            for (int t = 0; t < input.sizes[c]; ++t) {
                //std::cout << input.sizes[c] << " ";
                int substr_index = input.occurencies_substr[c][t];
                int substr_pos = input.occurencies_index[c][t];
                //std::cout << "(" << substr_index << ", " << substr_pos << ") ";
                if (j - substr_pos >= 0) {
                    --working_matrix[substr_index][j - substr_pos];
                }
            }
        }

        return input.working_matrix;
    }
};

__global__ void operate_working_matrix(int** working_matrix, int cols, int* sizes, int** occurencies_substr, int** occurencies_index, const char* H) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadId >= cols) {
        return;
    }
    
    unsigned char c = H[threadId];
    for (int t = 0; t < sizes[c]; ++t) {
        int substr_index = occurencies_substr[c][t];
        int substr_pos = occurencies_index[c][t];
        
        if (threadId - substr_pos >= 0) {
            atomicAdd(&working_matrix[substr_index][threadId - substr_pos], -1);
            //working_matrix[substr_index][threadId - substr_pos] -= 1;
        }
        
    }
}

class WorkingMatrixHandlerGPU : public WorkingMatrixHandler {
public:
    WorkingMatrix& process(PrehandlerResult& input) override {
        int** working_matrix = input.working_matrix.getMatrix();
        int cols = input.working_matrix.getCols();
        int rows = input.working_matrix.getRows();
        
        operate_working_matrix << <(cols + 1023) / 1024, 1024 >> > (working_matrix, cols, input.sizes, input.occurencies_substr, input.occurencies_index, input.input_data.H);
        CudaSafeFunc::cudaDeviceSynchronize();
        return input.working_matrix;
    }
};

// Base class for Result
class BaseResult {
};

template<class T>
class Result : public BaseResult {
private: 
    T result;
public:
    Result() {}
    Result(T& result) : result(result) {}
    Result(const Result& result_instance) {
        this->result = result_instance.result;
    }
    Result<T> operator=(const Result<T>& result_instance) {
        this->result = result_instance.result;
        return *this;
    }
    T& get() {
        return result;
    }
};

template<class T>
class Posthandler : public Task<std::pair<InputData*, WorkingMatrix*>, T> {};

class SimpleOccurenceResult : public Posthandler<Result<bool*>*> {};

class DetailedOccurenceResult : public Posthandler<Result<std::pair<int**, int*>>*> {};

class SimpleOccurenceResultCPU : public SimpleOccurenceResult {
private:
    Result<bool*>* result;
public:
    SimpleOccurenceResultCPU() : result() {}
    Result<bool*>*& process(std::pair<InputData*, WorkingMatrix*>& input) override {
        int substr_count = input.first->substr_count;
        bool* answers = new bool[substr_count];
        for (int i = 0; i < substr_count; ++i) {
            answers[i] = false;
        }

        int** working_matrix = input.second->getMatrix();
        int cols = input.second->getCols();
        int rows = input.second->getRows();
        for (int i = 0; i < substr_count; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (working_matrix[i][j] == 0) {
                    answers[i] = true;
                    break;
                }
            }
        }
        result = new Result<bool*>(answers);
        return result;
    }
};

class DetailedOccurenceResultCPU : public DetailedOccurenceResult {
private:
    Result<std::pair<int**, int*>>* result;
public:
    DetailedOccurenceResultCPU() {}
    Result<std::pair<int**, int*>>*& process(std::pair<InputData*, WorkingMatrix*>& input) override {
        int substr_count = input.first->substr_count;

        int** occurencies = new int*[substr_count];
        int* sizes = new int[substr_count];
        for (int i = 0; i < substr_count; ++i) {
            sizes[i] = 0;
        }

        int** working_matrix = input.second->getMatrix();
        int cols = input.second->getCols();
        int rows = input.second->getRows();
        for (int i = 0; i < substr_count; ++i) {
            int occurence_count = 0;
            for (int j = 0; j < cols; ++j) {
                if (working_matrix[i][j] == 0) {
                    ++occurence_count;
                }
            }
            occurencies[i] = new int[occurence_count];
            sizes[i] = occurence_count;

            int cur_occurence_index = 0;
            for (int j = 0; j < cols; ++j) {
                if (working_matrix[i][j] == 0) {
                    occurencies[i][cur_occurence_index++] = j;
                }
            }
        }
        std::pair<int**, int*> answer;
        answer.first = occurencies;
        answer.second = sizes;
        result = new Result<std::pair<int**, int*>>(answer);
        return result;
    }
};

__global__ void find_answers(int** working_matrix, int rows, int cols, bool* answers) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= rows) {
        return;
    }

    if (j >= cols) {
        return;
    }

    if (working_matrix[i][j] == 0) {
        answers[i] = true;
    }
}

class SimpleOccurenceResultGPU : public SimpleOccurenceResult {
private:
    Result<bool*>* result;
public:
    Result<bool*>*& process(std::pair<InputData*, WorkingMatrix*>& input) override {
        int substr_count = input.first->substr_count;
        /*
        bool* answers = new bool[substr_count];
        for (int i = 0; i < substr_count; ++i) {
            answers[i] = false;
        }
        */
        bool* answers;
        CudaSafeFunc::cudaMalloc(&answers, substr_count * sizeof(bool));

        fill_array_with_value << <(substr_count + 1023) / 1024, 1024 >> > (answers, false, substr_count);
        CudaSafeFunc::cudaDeviceSynchronize();

        int** working_matrix = input.second->getMatrix();
        int cols = input.second->getCols();
        int rows = input.second->getRows();

        /*
        for (int i = 0; i < substr_count; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (working_matrix[i][j] == 0) {
                    answers[i] = true;
                    break;
                }
            }
        }
        */

        dim3 grid((cols + 31) / 32, (substr_count + 31) / 32);
        dim3 block(32, 32);

        find_answers << <grid, block >> > (working_matrix, rows, cols, answers);
        CudaSafeFunc::cudaDeviceSynchronize();

        result = new Result<bool*>(answers);
        return result;
    }
};

__global__ void count_occurencies(int** working_matrix, int rows, int cols, int* occurencies) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= rows) {
        return;
    }

    if (j >= cols) {
        return;
    }

    if (working_matrix[i][j] == 0) {
        atomicAdd(&occurencies[i], 1);
    }
}

__global__ void find_occurencies(int** working_matrix, int** occurencies, int* cur_occurence_indexes, int rows, int cols) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= rows) {
        return;
    }

    if (j >= cols) {
        return;
    }

    if (working_matrix[i][j] == 0) {
        occurencies[i][atomicAdd(&cur_occurence_indexes[i], 1)] = j;
    }
}

class DetailedOccurenceResultGPU : public DetailedOccurenceResult {
private:
    Result<std::pair<int**, int*>>* result;
public:
    Result<std::pair<int**, int*>>*& process(std::pair<InputData*, WorkingMatrix*>& input) override {
        int substr_count = input.first->substr_count;

        /*
        int** occurencies = new int* [substr_count];
        int* sizes = new int[substr_count];
        for (int i = 0; i < substr_count; ++i) {
            sizes[i] = 0;
        }
        */

        int** occurencies;
        int** occurencies_h = new int*[substr_count];
        CudaSafeFunc::cudaMalloc(&occurencies, sizeof(int*) * substr_count);

        int** working_matrix = input.second->getMatrix();
        int cols = input.second->getCols();
        int rows = input.second->getRows();

        int* sizes;
        CudaSafeFunc::cudaMalloc(&sizes, sizeof(int) * substr_count);
        fill_array_with_value << <(substr_count + 1023) / 1024, 1024 >> > (sizes, 0, substr_count);
        CudaSafeFunc::cudaDeviceSynchronize();

        /*
        for (int i = 0; i < substr_count; ++i) {
            int occurence_count = 0;
            for (int j = 0; j < cols; ++j) {
                if (working_matrix[i][j] == 0) {
                    ++occurence_count;
                }
            }
            occurencies[i] = new int[occurence_count];
            sizes[i] = occurence_count;

            int cur_occurence_index = 0;
            for (int j = 0; j < cols; ++j) {
                if (working_matrix[i][j] == 0) {
                    occurencies[i][cur_occurence_index++] = j;
                }
            }
        }
        */

        dim3 grid((cols + 31) / 32, (substr_count + 31) / 32);
        dim3 block(32, 32);
        count_occurencies<<<grid, block>>>(working_matrix, rows, cols, sizes);
        CudaSafeFunc::cudaDeviceSynchronize();

        int* sizes_h = new int[substr_count];
        CudaSafeFunc::cudaMemcpy(sizes_h, sizes, sizeof(int) * substr_count, cudaMemcpyKind::cudaMemcpyDeviceToHost);
        for (int i = 0; i < substr_count; ++i) {
            CudaSafeFunc::cudaMalloc(&occurencies_h[i], sizes_h[i] * sizeof(int*));
        }

        delete sizes_h;

        CudaSafeFunc::cudaMemcpy(occurencies, occurencies_h, sizeof(int*) * substr_count, cudaMemcpyKind::cudaMemcpyHostToDevice);

        int* cur_occurence_indexes;
        CudaSafeFunc::cudaMalloc(&cur_occurence_indexes, sizeof(int) * substr_count);
        fill_array_with_value << <(substr_count + 1023) / 1024, 1024 >> > (cur_occurence_indexes, 0, substr_count);
        CudaSafeFunc::cudaDeviceSynchronize();

        find_occurencies << <grid, block >> > (working_matrix, occurencies, cur_occurence_indexes, rows, cols);
        CudaSafeFunc::cudaDeviceSynchronize();

        std::pair<int**, int*> answer;
        answer.first = occurencies;
        answer.second = sizes;
        result = new Result<std::pair<int**, int*>>(answer);
        return result;
    }
};

template<class PrehandlerOutput, class WorkingMatrixOutput>
class Algorithm {
private:
    InputData& input_data;
    Task<InputData, PrehandlerOutput>* algorithm_prehandler;
    Task<PrehandlerOutput, WorkingMatrixOutput>* working_matrix_handler;
    std::vector < Task < std::pair<InputData*, WorkingMatrixOutput*>, BaseResult* >*> result_task_vector;
    std::vector<BaseResult*> results;
public:
    Algorithm(InputData& input_data, Task<InputData, PrehandlerOutput>* algorithm_prehandler, 
        Task<PrehandlerOutput, WorkingMatrixOutput>* working_matrix_handler, 
        std::vector < Task < std::pair<InputData*, WorkingMatrixOutput*>, BaseResult* >*> result_task_vector)
        : input_data(input_data), algorithm_prehandler(algorithm_prehandler), working_matrix_handler(working_matrix_handler), result_task_vector(result_task_vector) {}

    std::vector<BaseResult*>& run() {
        PrehandlerOutput& prehandler_output = algorithm_prehandler->process(input_data);
        WorkingMatrixOutput& working_matrix_output = working_matrix_handler->process(prehandler_output);

        /*
        for (int i = 0; i < input_data.substr_count; ++i) {
            for (int j = 0; j < input_data.h; ++j) {
                std::cout << working_matrix_output.getMatrix()[i][j] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
        */

        for (Task < std::pair<InputData*, WorkingMatrixOutput*>, BaseResult* >* result_task : result_task_vector) {
            std::pair<InputData*, WorkingMatrixOutput*> pair(&input_data, &working_matrix_output);
            BaseResult* result = result_task->process(pair);
            results.push_back(result);
        }
        return results;
    }
};

bool* get_cpu_simple_result(InputData& cpu_data, Params& params) {
    AlgorithmPrehandlerCPU cpu_prehandler;
    WorkingMatrixHandlerCPU cpu_matrix_handler;
    SimpleOccurenceResultCPU cpu_simple_handler;

    std::vector< Task < std::pair<InputData*, WorkingMatrix*>, BaseResult* >*> cpu_posthandlers;
    cpu_posthandlers.push_back((Task < std::pair<InputData*, WorkingMatrix*>, BaseResult* >*) & cpu_simple_handler);

    Algorithm<PrehandlerResult, WorkingMatrix> cpu_algorithm(cpu_data, &cpu_prehandler, &cpu_matrix_handler, cpu_posthandlers);
    std::vector<BaseResult*>& results = cpu_algorithm.run();

    Result<bool*>* cpu_simple_result = static_cast<Result<bool*>*> (results[0]);
    return cpu_simple_result->get();
}

std::pair<int**, int*> get_cpu_detailed_result(InputData& cpu_data, Params& params) {
    AlgorithmPrehandlerCPU cpu_prehandler;
    WorkingMatrixHandlerCPU cpu_matrix_handler;
    DetailedOccurenceResultCPU cpu_detailed_handler;

    std::vector< Task < std::pair<InputData*, WorkingMatrix*>, BaseResult* >*> cpu_posthandlers;
    cpu_posthandlers.push_back((Task < std::pair<InputData*, WorkingMatrix*>, BaseResult* >*) & cpu_detailed_handler);

    Algorithm<PrehandlerResult, WorkingMatrix> cpu_algorithm(cpu_data, &cpu_prehandler, &cpu_matrix_handler, cpu_posthandlers);
    std::vector<BaseResult*>& results = cpu_algorithm.run();

    Result<std::pair<int**, int*>>* cpu_detailed_result = static_cast<Result<std::pair<int**, int*>>*> (results[0]);
    return cpu_detailed_result->get();
}

bool* get_gpu_simple_result(InputData& gpu_data, Params& params) {
    AlgorithmPrehandlerGPU gpu_prehandler;
    WorkingMatrixHandlerGPU gpu_matrix_handler;
    SimpleOccurenceResultGPU gpu_simple_handler;
    DetailedOccurenceResultGPU gpu_detailed_handler;

    std::vector< Task < std::pair<InputData*, WorkingMatrix*>, BaseResult* >*> gpu_posthandlers;
    gpu_posthandlers.push_back((Task < std::pair<InputData*, WorkingMatrix*>, BaseResult* >*) & gpu_simple_handler);

    Algorithm<PrehandlerResult, WorkingMatrix> gpu_algorithm(gpu_data, &gpu_prehandler, &gpu_matrix_handler, gpu_posthandlers);
    std::vector<BaseResult*>& gpu_results = gpu_algorithm.run();

    Result<bool*>* gpu_simple_result = static_cast<Result<bool*>*> (gpu_results[0]);

    bool* gpu_simple_result_arr = gpu_simple_result->get();
    bool* gpu_simple_result_arr_h = new bool[params.substr_count];
    CudaSafeFunc::cudaMemcpy(gpu_simple_result_arr_h, gpu_simple_result_arr, sizeof(bool) * params.substr_count, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    return gpu_simple_result_arr_h;
}

std::pair<int**, int*> get_gpu_detailed_result(InputData& gpu_data, Params& params) {
    AlgorithmPrehandlerGPU gpu_prehandler;
    WorkingMatrixHandlerGPU gpu_matrix_handler;
    SimpleOccurenceResultGPU gpu_simple_handler;
    DetailedOccurenceResultGPU gpu_detailed_handler;

    std::vector< Task < std::pair<InputData*, WorkingMatrix*>, BaseResult* >*> gpu_posthandlers;
    gpu_posthandlers.push_back((Task < std::pair<InputData*, WorkingMatrix*>, BaseResult* >*) & gpu_detailed_handler);

    Algorithm<PrehandlerResult, WorkingMatrix> gpu_algorithm(gpu_data, &gpu_prehandler, &gpu_matrix_handler, gpu_posthandlers);
    std::vector<BaseResult*>& gpu_results = gpu_algorithm.run();

    Result<std::pair<int**, int*>>* gpu_detailed_result = static_cast<Result<std::pair<int**, int*>>*> (gpu_results[0]);

    std::pair<int**, int*> detailed_result_pair = gpu_detailed_result->get();
    int* sizes_h = new int[params.substr_count];
    CudaSafeFunc::cudaMemcpy(sizes_h, detailed_result_pair.second, params.substr_count * sizeof(int), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    int** occurencies_h = new int* [params.substr_count];
    CudaSafeFunc::cudaMemcpy(occurencies_h, detailed_result_pair.first, params.substr_count * sizeof(int*), cudaMemcpyKind::cudaMemcpyDeviceToHost);

    for (int i = 0; i < params.substr_count; ++i) {
        int* occur_part_h = new int[sizes_h[i]];
        CudaSafeFunc::cudaMemcpy(occur_part_h, occurencies_h[i], sizeof(int) * sizes_h[i], cudaMemcpyKind::cudaMemcpyDeviceToHost);
        occurencies_h[i] = occur_part_h;
    }

    return std::pair<int**, int*>(occurencies_h, sizes_h);
}

void output_all(char* filename, InputData cpu_data, bool* simple_result, std::pair<int**, int*> detailed_result) {
    std::ofstream out(filename);
    if (out.is_open()) {
        out << "STRING TO SEARCH IN\n\n";
        for (int i = 0; i < cpu_data.h; ++i) {
            out << +(unsigned char)cpu_data.H[i] << " ";
        }
        out << "\n\n";

        out << "SUBSTRINGS\n\n";

        for (int i = 0; i < cpu_data.substr_count; ++i) {
            out << i << ": ";
            for (int j = 0; j < cpu_data.sizes[i]; ++j) {
                out << +(unsigned char)cpu_data.all_strings[cpu_data.starts[i] + j] << " ";
            }
            out << "\n";
        }
        out << "\n\n";

        out << "RESULTS\n\n";
        for (int i = 0; i < cpu_data.substr_count; ++i) {
            out << "SUBSTRING " << i << ": " << ((simple_result[i]) ? "FOUND" : "NOT FOUND") << ": ";
            for (int j = 0; j < detailed_result.second[i]; ++j) {
                out << detailed_result.first[i][j] << " ";
            }
            out << "\n";
        }
        out << "\n";
    }
    out.close();
}

int main(int argc, char** argv)
{
    Params params = InputReader::read(argc, argv);
    std::cout << params.h << " " << params.substr_count << " " << params.min_substr_size << " " << params.max_substr_size << " " << params.how_to_search << "\n";

    DataCreator data_creator(params);
    data_creator.create();

    InputData cpu_data = data_creator.forCPU();
    InputData gpu_data = data_creator.forGPU();

    bool* cpu_simple_result_arr_h = get_cpu_simple_result(cpu_data, params);
    std::pair<int**, int*> cpu_detailed_result_pair = get_cpu_detailed_result(cpu_data, params);

    bool* gpu_simple_result_arr_h = get_gpu_simple_result(gpu_data, params);
    std::pair<int**, int*> gpu_detailed_result_pair = get_gpu_detailed_result(gpu_data, params);

    output_all("output_cpu.txt", cpu_data, cpu_simple_result_arr_h, cpu_detailed_result_pair);
    output_all("output_gpu.txt", cpu_data, gpu_simple_result_arr_h, gpu_detailed_result_pair);

    bool simple_results_equal = true;
    bool detailed_results_equal = true;

    for (int i = 0; i < cpu_data.substr_count; ++i) {
        if (cpu_simple_result_arr_h[i] != gpu_simple_result_arr_h[i]) {
            simple_results_equal = false;
            break;
        }
    }

    for (int i = 0; i < cpu_data.substr_count; ++i) {
        if (cpu_detailed_result_pair.second[i] != gpu_detailed_result_pair.second[i]) {
            detailed_results_equal = false;
            std::cout << cpu_detailed_result_pair.second[i] << " " << gpu_detailed_result_pair.second[i] << "\n";
            break;
        }
        std::vector<int> cpu, gpu;
        for (int j = 0; j < cpu_detailed_result_pair.second[i]; ++j) {
            cpu.push_back(cpu_detailed_result_pair.first[i][j]);
            gpu.push_back(gpu_detailed_result_pair.first[i][j]);
        }
        std::sort(cpu.begin(), cpu.end());
        std::sort(gpu.begin(), gpu.end());
        if (cpu != gpu) {
            detailed_results_equal = false;
            for (int value : cpu) {
                std::cout << value << " ";
            }
            std::cout << "\n";
            for (int value : gpu) {
                std::cout << value << " ";
            }
            std::cout << "\n";
            break;
        }
    }

    if (simple_results_equal) {
        std::cout << "simple results are EQUAL\n";
    }
    else {
        std::cout << "simple results are NOT EQUAL\n";
    }

    if (detailed_results_equal) {
        std::cout << "detailed results are EQUAL\n";
    }
    else {
        std::cout << "detailed results are NOT EQUAL\n";
    }

    double cpu_simple_time = 0, cpu_detailed_time = 0;
    double gpu_simple_time = 0, gpu_detailed_time = 0;

    get_CPU_time([&]() {get_cpu_simple_result(cpu_data, params);}, cpu_simple_time);
    get_CPU_time([&]() {get_cpu_detailed_result(cpu_data, params);}, cpu_detailed_time);
    get_CPU_time([&]() {get_gpu_simple_result(gpu_data, params);}, gpu_simple_time);
    get_CPU_time([&]() {get_gpu_detailed_result(gpu_data, params);}, gpu_detailed_time);

    std::cout << "cpu simple result time: " << cpu_simple_time << "\n";
    std::cout << "cpu detailed result time: " << cpu_detailed_time << "\n";
    std::cout << "gpu simple result time: " << gpu_simple_time << "\n";
    std::cout << "gpu detailed result time: " << gpu_detailed_time << "\n";
    std::cout << "\n";
    std::cout << "simple speedup: " << cpu_simple_time / gpu_detailed_time << "\n";
    std::cout << "detailed speedup: " << cpu_detailed_time / gpu_detailed_time << "\n";

    /*

    std::cout << n << "\n";
    double* xs = (double*)malloc(n * sizeof(double));
    double* ys = (double*)malloc(n * sizeof(double));

    double* d_xs, * d_ys;
    cudaMalloc(&d_xs, n * sizeof(double));
    cudaMalloc(&d_ys, n * sizeof(double));

    // заполнение случайными числами
    GPU_fill_rand(d_xs, n);
    GPU_fill_rand(d_ys, n);

    cudaMemcpy(xs, d_xs, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(ys, d_ys, n * sizeof(double), cudaMemcpyDeviceToHost);

    std::vector<double(*)(double*, double*, int)> CPU_implementations = { CPU_calc_with_if, CPU_calc_without_if };

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
        "GPU calculation without if", "GPU calculation with radius integrated into reduction" };
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
    */
}