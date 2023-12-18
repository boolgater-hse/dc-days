#include <cstdio>
#include <cuda.h>
#include <ctime>
#include <sys/time.h>

#define GET_TIME(now) { struct timeval t; gettimeofday(&t, NULL); now = t.tv_sec + t.tv_usec/1000000.0; }

#define VERBOSE

void serialSolution(int *matrixDimSize, double **initialMatrixCopy, double *resultsVectorForCpu) {
    for (int i = 0; i < *matrixDimSize - 1; i++) {
        for (int j = i + 1; j < *matrixDimSize; j++) {
            for (int k = i + 1; k < *matrixDimSize + 1; k++) {
                initialMatrixCopy[j][k] =
                        ((-initialMatrixCopy[j][i] / initialMatrixCopy[i][i]) * initialMatrixCopy[i][k]) + initialMatrixCopy[j][k];
            }
            initialMatrixCopy[j][i] = 0;
        }
    }

    for (int i = *matrixDimSize - 1; i >= 0; i--) {
        for (int j = *matrixDimSize - 1; j >= i; j--) {
            if (j == i) {
                resultsVectorForCpu[i] = initialMatrixCopy[i][*matrixDimSize] / initialMatrixCopy[i][j];
            } else {
                initialMatrixCopy[i][*matrixDimSize] -= initialMatrixCopy[i][j] * resultsVectorForCpu[j];
            }
        }
    }
}

__global__ void transformToUpperTriangularLinearSystem(double **initialMatrix, int i) {
    int currentThreadInRowNum = threadIdx.x;
    int currentRowNum = blockIdx.x;

    if (currentThreadInRowNum > i && currentRowNum > i && initialMatrix[currentRowNum][i] != 0) {
        initialMatrix[currentRowNum][currentThreadInRowNum] =
                ((-initialMatrix[currentRowNum][i] / initialMatrix[i][i]) * initialMatrix[i][currentThreadInRowNum]) +
                initialMatrix[currentRowNum][currentThreadInRowNum];
    }
    __syncthreads();
    if (currentThreadInRowNum == i && currentRowNum > i && initialMatrix[currentRowNum][i] != 0) {
        initialMatrix[currentRowNum][currentThreadInRowNum] = 0;
    }
}

__global__ void calculateLinearSystemResults(int *matrixDimSize, double **initialMatrix, double *resultsVector) {
    int currentThreadInBlockNum = threadIdx.x;

    double tempResult = initialMatrix[currentThreadInBlockNum][*matrixDimSize];

    for (int j = *matrixDimSize - 1; j >= 0; j--) {
        if (currentThreadInBlockNum == j) {
            resultsVector[j] = tempResult / initialMatrix[currentThreadInBlockNum][currentThreadInBlockNum];
        }
        __syncthreads();
        if (currentThreadInBlockNum < j) {
            tempResult -= initialMatrix[currentThreadInBlockNum][j] * resultsVector[j];
        }
    }
}

int main() {
    double start, end;
    int *matrixDimSize;
    double **initialMatrix;
    double *resultsVector;
    double **initialMatrixCopy;
    double *resultsVectorForCpu;

    cudaMallocManaged(&matrixDimSize, 1 * sizeof(int));
    *matrixDimSize = 1000;

    cudaMallocManaged(&resultsVector, *matrixDimSize * sizeof(double));
    cudaMallocManaged(&initialMatrix, *matrixDimSize * sizeof(double *));
    for (int i = 0; i < *matrixDimSize; i++) {
        cudaMallocManaged(&initialMatrix[i], (*matrixDimSize + 1) * sizeof(double));
    }
    resultsVectorForCpu = (double *) malloc(*matrixDimSize * sizeof(double));
    initialMatrixCopy = (double **) malloc(*matrixDimSize * sizeof(double *));
    for (int i = 0; i < *matrixDimSize; i++) {
        initialMatrixCopy[i] = (double *) malloc((*matrixDimSize + 1) * sizeof(double));
    }

    srand(time(NULL));
    for (int i = 0; i < *matrixDimSize; i++) {
        for (int j = 0; j < *matrixDimSize + 1; j++) {
            initialMatrix[i][j] = (double) rand();
            initialMatrixCopy[i][j] = initialMatrix[i][j];
        }
    }

    GET_TIME(start);
    serialSolution(matrixDimSize, initialMatrixCopy, resultsVectorForCpu);
    GET_TIME(end);

    printf("Computing time for single threaded CPU solution: %.9lf\n", end - start);
#ifdef VERBOSE
    printf("Solution:");
    for (int i = 0; i < *matrixDimSize; i++) {
        printf(" %lf", resultsVectorForCpu[i]);
    }
    printf("\n");
#endif

    GET_TIME(start);
    for (int i = 0; i < *matrixDimSize - 1; i++) {
        transformToUpperTriangularLinearSystem<<<*matrixDimSize, (*matrixDimSize + 1)>>>(initialMatrix, i);
    }
    calculateLinearSystemResults<<<1, *matrixDimSize>>>(matrixDimSize, initialMatrix, resultsVector);
    cudaDeviceSynchronize();
    GET_TIME(end);

    printf("Computing time for CUDA solution: %.9lf\n", end - start);
#ifdef VERBOSE
    printf("Solution:");
    for (int i = 0; i < *matrixDimSize; i++) {
        printf(" %lf", resultsVector[i]);
    }
    printf("\n");
#endif

    for (int i = 0; i < *matrixDimSize; i++) {
        cudaFree(initialMatrix[i]);
    }
    cudaFree(initialMatrix);
    cudaFree(resultsVector);
    for (int i = 0; i < *matrixDimSize; i++) {
        free(initialMatrixCopy[i]);
    }
    free(initialMatrixCopy);
    free(resultsVectorForCpu);
    cudaFree(matrixDimSize);

    return 0;
}
