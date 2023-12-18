#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <time.h>
#include <sys/time.h>

#define GET_TIME(now) { struct timeval t; gettimeofday(&t, NULL); now = t.tv_sec + t.tv_usec / 1000000.0; }
#define GET_TIME_DIFF(start, end) (end - start)

#define N 512

// #define VERBOSE

int* alloc_matrix(int n) {
    int* matrix = malloc(n * n * sizeof(int));
    for (int i = 0; i < n * n; ++i) {
        matrix[i] = 0;
    }

    return matrix;
}

int* alloc_matrix_and_fill(int n) {
    int* matrix = malloc(n * n * sizeof(int));
    for (int i = 0; i < n * n; ++i) {
        matrix[i] = (int) (rand() % (10 + 1 - (-5)) + (-5));
    }

    return matrix;
}

void print_matrix(int* matrix, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (j == n - 1) {
                printf("%4d", matrix[i * n + j]);
                continue;
            }
            printf("%4d ", matrix[i * n + j]);
        }
        printf("\n");
    }
}

void matrices_multiplication(const int* a, const int* b, int size, int* result) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                result[i * size + j] += a[i * size + k] * b[k * size + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    srand(time(0));

    MPI_Comm comm;
    int rank;
    int world_size;

    int* matrix_a = 0;
    int* matrix_b = 0;
    int* result = 0;
    int n_pow2 = N * N;

    int broadcast_data[2], coord[2];

    int* local_matrix_a;
    int* local_matrix_b;
    int* local_result;
    int left, right, up, down;

    double start, end;

    GET_TIME(start)
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        matrix_a = alloc_matrix_and_fill(N);
        matrix_b = alloc_matrix_and_fill(N);
        result = alloc_matrix(N);

        int sqrt_world_size = (int) sqrt(world_size);
        int block_size = (int) N / sqrt_world_size;

        broadcast_data[0] = sqrt_world_size;
        broadcast_data[1] = block_size;

#ifdef VERBOSE
        print_matrix(matrix_a, N);
        printf("*\n");
        print_matrix(matrix_b, N);
#endif
    }

    MPI_Bcast(&broadcast_data, 2, MPI_INT, 0, MPI_COMM_WORLD);
    int sqrt_world_size = broadcast_data[0];
    int block_size = broadcast_data[1];

    int dim[2] = {sqrt_world_size, sqrt_world_size};
    int period[2] = {1, 1};
    int reorder = 1;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &comm);

    int matrix_sizes[2] = {N, N};
    int submatrix_sizes[2] = {block_size, block_size};
    int start_indexes[2] = {0, 0};
    MPI_Datatype type, subarray_type;
    MPI_Type_create_subarray(2, matrix_sizes, submatrix_sizes, start_indexes, MPI_ORDER_C, MPI_INT, &type);
    MPI_Type_create_resized(type, 0, block_size * sizeof(int), &subarray_type);
    MPI_Type_commit(&subarray_type);

    int* shared_matrix_a = 0;
    int* shared_matrix_b = 0;
    int* shared_result = 0;
    if (rank == 0) {
        shared_matrix_a = matrix_a;
        shared_matrix_b = matrix_b;
        shared_result = result;
    }

    int* send_counts = (int*) malloc(world_size * sizeof(int));
    int* displacements = (int*) malloc(world_size * sizeof(int));

    if (rank == 0) {
        for (int i = 0; i < world_size; ++i) {
            send_counts[i] = 1;
        }

        for (int i = 0, count = 0; i < sqrt_world_size; ++i) {
            for (int j = 0; j < sqrt_world_size; ++j) {
                displacements[i * sqrt_world_size + j] = count++;
            }
            count += (block_size - 1) * sqrt_world_size;
        }
    }

    local_matrix_a = alloc_matrix(block_size);
    local_matrix_b = alloc_matrix(block_size);

    MPI_Scatterv(shared_matrix_a, send_counts, displacements, subarray_type, local_matrix_a,
                 n_pow2 / world_size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(shared_matrix_b, send_counts, displacements, subarray_type, local_matrix_b,
                 n_pow2 / world_size, MPI_INT, 0, MPI_COMM_WORLD);

    local_result = alloc_matrix(block_size);

    MPI_Cart_coords(comm, rank, 2, coord);

    MPI_Cart_shift(comm, 1, coord[0], &left, &right);
    MPI_Sendrecv_replace(local_matrix_a, block_size * block_size, MPI_INT, left, 1, right, 1, comm, MPI_STATUS_IGNORE);

    MPI_Cart_shift(comm, 0, coord[1], &up, &down);
    MPI_Sendrecv_replace(local_matrix_b, block_size * block_size, MPI_INT, up, 1, down, 1, comm, MPI_STATUS_IGNORE);

    int* multiplication_result = alloc_matrix(block_size);
    for (int k = 0; k < sqrt_world_size; ++k) {
        matrices_multiplication(local_matrix_a, local_matrix_b, block_size, multiplication_result);

        for (int i = 0; i < block_size; ++i) {
            for (int j = 0; j < block_size; ++j) {
                 local_result[i * block_size + j] += multiplication_result[i * block_size + j];
            }
        }

        MPI_Cart_shift(comm, 1, 1, &left, &right);
        MPI_Sendrecv_replace(local_matrix_a, block_size * block_size, MPI_INT, left, 1, right, 1, comm, MPI_STATUS_IGNORE);

        MPI_Cart_shift(comm, 0, 1, &up, &down);
        MPI_Sendrecv_replace(local_matrix_b, block_size * block_size, MPI_INT, up, 1, down, 1, comm, MPI_STATUS_IGNORE);
    }

    MPI_Gatherv(local_result, n_pow2 / world_size, MPI_INT,
                shared_result, send_counts, displacements, subarray_type,
                0, MPI_COMM_WORLD);

#ifdef VERBOSE
    if (rank == 0) {
        printf("=\n");
        print_matrix(shared_result, N);
        printf("\n");
    }
#endif

    MPI_Type_free(&type);
    MPI_Type_free(&subarray_type);
    MPI_Finalize();
    GET_TIME(end)

    if (rank == 0) {
        printf("%lf\n", GET_TIME_DIFF(start, end));
    }

    free(local_matrix_a);
    free(local_matrix_b);
    free(local_result);
    free(multiplication_result);
    free(shared_matrix_a);
    free(shared_matrix_b);
    free(shared_result);

    return 0;
}
