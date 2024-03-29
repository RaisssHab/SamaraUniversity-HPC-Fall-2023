<h1>Mass search</h1>
<h2>Постановка эксперимента</h2>
Цель – исследовать ускорение при массовом поиске подстрок исходной строки на CUDA относительно последовательной реализации.

Задачи:
1.	Рассчитать время, достигаемое последовательной реализацией алгоритма.
2.	Рассчитать время и ускорение, достигаемое с использованием написанной параллельной реализации на CUDA.
3.	Проанализировать результат, сделать выводы.

<h2>Инструментальные средства эксперимента</h2>
<h3>Программные средства</h3>
Язык программирования – C++. 

Было создано семейство классов, посвященных решению задачи.
Класс Algorithm - основной класс, который задает общую канву алгоритма, его трехэтапность и использование следующим этапов результатов предыдущего. По сути это пайплайн, но жестко фиксированный на предложенной структуре алгоритма.
```c++
template<class PrehandlerOutput, class WorkingMatrixOutput>
class Algorithm {
private:
    InputData& input_data;
    Task<InputData, PrehandlerOutput>* algorithm_prehandler;
    Task<PrehandlerOutput, WorkingMatrixOutput>* working_matrix_handler;
    std::vector < Task < std::pair<InputData*, WorkingMatrixOutput*>, BaseResult* >*> result_task_vector;
    std::vector<BaseResult*> results;
public:
    ...
    std::vector<BaseResult*>& run() {...}
}
```

Классы, соответствующие этапам:
- AlgorithmPrehandler (AlgorithmPrehandlerCPU, AlgorithmPrehandlerGPU) - для предварительного этапа алгоритма;
- WorkingMatrixHandler (WorkingMatrixHandlerCPU, WorkingMatrixHandlerGPU) - для основного этапа алгоритма, связанного с обработкой рабочей матрицы;
- BaseResult, Result - для хранения результатов на основе обработанной рабочей матрицы;
- Posthandler, SimpleOccurenceResult, DetailedOccurenceResult (SimpleOccurenceResultCPU, SimpleOccurenceResultGPU, DetailedOccurenceResultCPU, DetailedOccurenceResultGPU) - для формирования результатов на основе обработанной рабочей матрицы.

Дополнительные классы:
- InputData - для хранения исходных данных, используется в двух экземплярах - для CPU и GPU;
- DataCreator - для формирования исходных данных с учетом заданной параметризации.

Приложение консольное, параметры идут в следующем порядке:
- длина строки, в которой будет происходить поиск;
- количество подстрок;
- минимальный размер подстроки;
- максимальный размер подстроки;
- режим (простой или детализированный результат, в программе хоть и хранится, но из-за экспериментов не учитывается).

Следует отметить особенность последнего этапа класса Algorithm. Он заключается в том, что возвращается вектор результатов типа BaseResult. Это делается потому, что результат алгоритма допускает много форм, и можно задать несколько обработчиков для рабочей матрицы, чтобы получить необходимый результат.

Чтобы минимизировать риск ошибок, связанных с использованием CUDA-функций, заводится вспомогательный класс с часто используемыми функциями, которые выбрасывают исключение в случае некорректного завершения выполнения функции:
```cpp
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
```

Последовательная реализация дана в соответствии с указаниями из методички. Параллельная реализация представляет собой переписанную последовательную реализацию с использованием параллелизма там, где это удалось сделать.

Из недостатков программы следует отметить отсутствие работы с памятью - выделенная память не освобождается. Это объясняется ограниченностью времени.

<h3>Системные средства</h3>
Операционная система – Ubuntu (компилятор NVCC, Google Colab).

<h3>Аппаратные средства</h3>

Видеокарта – T4 (Google Colab). 

<h2>Выбор параметров эксперимента</h2>

Будет проведено два эксперимента: 1) c варьированием количества подстрок (10, 100, 10000, 50000) относительно зафиксированного размера исходной строки (размер - 1000); 2) с варьированием размера исходной строки (10, 100, 10000, 50000) относительно зафиксированного количества подстрок (1000). Параметры выбирались таким образом, чтобы охватить случаи ожидаемо низкого ускорения и высокого ускорения.

<h2>Теоретическое предсказание результатов эксперимента</h2>

1) На одном из параметров при варьировании количества подстрок удастся получить ускорение больше единицы;
2) Хотя бы на одном из параметров при варировании размера исходной строки удастся получить ускорение больше единицы.

<h2>Проведение эксперимента</h2>

Для разных параметров в Google Colab запускается код, в котором уже выполнены все необходимые вычисления и получено ускорение. Эти значения выписываются и далее строятся графики.

<h2>Представление результатов</h2>

Результаты представлены рисунках 1-2.

![изображение](https://github.com/RaisssHab/SamaraUniversity-HPC-Fall-2023/assets/60664914/704d6784-9cb8-471b-af60-01a982fbdb2b)

Рисунок 1 – Зависимость ускорения от количества подстрок при длине исходной строки 1000 и длинах подстрок от 1 до 3

![изображение](https://github.com/RaisssHab/SamaraUniversity-HPC-Fall-2023/assets/60664914/470e0476-9bf3-44e4-9d6f-acf7e23db6c7)

Рисунок 2 – Зависимость ускорения от размера строки при длинах подстрок от 1 до 3 (количество подстрок 1000)

<h2>Описание результатов</h2>

1.	При варьировании количества подстрок ускорение сначала возрастает, потом демонстрирует скачок вниз. Максимальное достигнутое ускорение - около единицы.
2.	При варьировании размера строки ускорение возрастает. Максимальное ускорение соответствует примерно 12.
3.	Ускорение для упрощенного ответа ниже, чем для детализированного ответа.

<h2>Анализ результатов</h2>

1. Возможно, размер исходной строки слишком мал для того, чтобы увидеть существенное ускорение.
2. Размер строки играет важную роль для ускорения.
3. Детализированный ответ формировать сложнее, возникает синхронизация.
4. Этап с формированием ответа на основе рабочей матрицы имеет достаточно большой объем вычислений.

<h2>Заключение</h2>

Таким образом, в ходе данной работы была достигнута поставленная цель – проанализировано ускорение параллельной программы на CUDA. Написанный код может использоваться для реализации данного алгоритма, он допускает возможность различной интерпретации рабочей матрицы благодаря введению вектора возможных результатов.
