#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

#define M 5000
#define N 5000

// #define PRINT
// #define CHECK

double random_double(double min, double max) {
    return min + (rand() / (RAND_MAX / (max - min)));
}

void fillRandomly(double *a, double *x, int m, int n) {
#ifdef PRINT
    printf("Matrix A:\n");
#endif
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            a[i * n + j] = random_double(-100, 100);
#ifdef PRINT
            printf("%f\t", a[i * n + j]);
#endif
        }
#ifdef PRINT
        printf("\n");
#endif
    }
#ifdef PRINT
    printf("Vector x:\n");
#endif
    for (int j = 0; j < n; ++j) {
        x[j] = random_double(-100, 100);
#ifdef PRINT
        printf("%f\t", x[j]);
#endif
    }
#ifdef PRINT
    printf("\n");
#endif
}

void fillToCheck(double *a, double *x, int m, int n) {
#ifdef PRINT
    printf("Matrix A:\n");
#endif
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                a[i * n + j] = 1;
            } else {
                a[i * n + j] = 0;
            }
            x[j] = j + 1;
#ifdef PRINT
            printf("%f\t", a[i * n + j]);
#endif
        }
#ifdef PRINT
        printf("\n");
#endif
    }
#ifdef PRINT
    printf("Vector x:\n");
#endif
    for (int j = 0; j < n; ++j) {
        x[j] = j + 1;
#ifdef PRINT
        printf("%f\t", x[j]);
#endif
    }
#ifdef PRINT
    printf("\n");
#endif
}

void rowWise() {
    int my_rank, comm_sz;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int m_local = M / comm_sz + (M % comm_sz != 0);
    int n_local = N / comm_sz + (N % comm_sz != 0);
    double *a = NULL;
    double *x = NULL;

    if (my_rank == 0) {
        a = (double *) malloc(sizeof(double) * M * N);
        x = (double *) malloc(sizeof(double) * N);
#ifdef CHECK
        fillToCheck(a, x, M, N);
#else
        fillRandomly(a, x, M, N);
#endif
    }

    double *a_local = (double *) malloc(sizeof(double) * m_local * N);
    double *x_local = (double *) malloc(sizeof(double) * n_local);
    double *y_local = (double *) malloc(sizeof(double) * m_local);
    double *y = (double *) malloc(sizeof(double) * m_local * comm_sz);

    double t = MPI_Wtime();

    int *displs = (int *) malloc(sizeof(int) * comm_sz);
    int *sendcounts = (int *) malloc(sizeof(int) * comm_sz);
    int *recvcounts = (int *) malloc(sizeof(int) * comm_sz);
    for (int rank = 0; rank < comm_sz; ++rank) {
        displs[rank] = rank * m_local * N;
        sendcounts[rank] = m_local * N;
        recvcounts[rank] = m_local * N;
    }
    MPI_Scatter(x, n_local, MPI_DOUBLE, x_local, n_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(a, sendcounts, displs, MPI_DOUBLE, a_local, recvcounts[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double *x_full = (double *) malloc(sizeof(double) * N);
    MPI_Allgather(x_local, n_local, MPI_DOUBLE, x_full, n_local, MPI_DOUBLE, MPI_COMM_WORLD);
    for (int i = 0; i < m_local; i++) {
        y_local[i] = 0.0;
        for (int j = 0; j < N; j++) {
            y_local[i] += a_local[i * N + j] * x_full[j];
        }
    }

    MPI_Gather(y_local, m_local, MPI_DOUBLE, y, m_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    t = MPI_Wtime() - t;
    if (my_rank == 0) {
#ifdef PRINT
        printf("\nResult vector y:\n");
        for (int i = 0; i < M; ++i) {
            printf("%f\t", y[i]);
        }
#endif
        printf("\n\ntime - %lf\n", t);
    }

    MPI_Finalize();
    free(a);
    free(x);
    free(y);
    free(a_local);
    free(x_local);
    free(y_local);
    free(x_full);
    free(displs);
    free(sendcounts);
    free(recvcounts);
}

void columnWise() {
    int my_rank, comm_sz;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int n_local = N / comm_sz + (N % comm_sz != 0);
    double *a = NULL;
    double *x = NULL;

    if (my_rank == 0) {
        a = (double *) malloc(sizeof(double) * M * N);
        x = (double *) malloc(sizeof(double) * N);
#ifdef CHECK
        fillToCheck(a, x, M, N);
#else
        fillRandomly(a, x, M, N);
#endif
    }

    double *a_local = (double *) malloc(sizeof(double) * M * n_local);
    double *x_local = (double *) malloc(sizeof(double) * n_local);
    double *y_local = (double *) malloc(sizeof(double) * M);
    double *y = (double *) malloc(sizeof(double) * M);

    double t = MPI_Wtime();

    int *displs = (int *) malloc(sizeof(int) * comm_sz);
    int *sendcounts = (int *) malloc(sizeof(int) * comm_sz);
    int *recvcounts = (int *) malloc(sizeof(int) * comm_sz);
    for (int rank = 0; rank < comm_sz; ++rank) {
        displs[rank] = rank * n_local;
        sendcounts[rank] = n_local;
        recvcounts[rank] = n_local;
    }
    for (int row = 0; row < M; ++row) {
        MPI_Scatterv(&a[row * N], sendcounts, displs, MPI_DOUBLE, &a_local[row * recvcounts[my_rank]],
                     recvcounts[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    MPI_Scatter(x, n_local, MPI_DOUBLE, x_local, n_local, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < M; ++i) {
        y_local[i] = 0.0;
    }
    for (int j = 0; j < n_local; ++j) {
        for (int i = 0; i < M; ++i) {
            y_local[i] += a_local[i * n_local + j] * x_local[j];
        }
    }

    MPI_Reduce(y_local, y, M, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    t = MPI_Wtime() - t;
    if (my_rank == 0) {
#ifdef PRINT
        printf("\nResult vector y:\n");
        for (int i = 0; i < M; ++i) {
            printf("%f\t", y[i]);
        }
#endif
        printf("\n\ntime - %lf\n", t);
    }

    MPI_Finalize();
    free(a);
    free(x);
    free(y);
    free(a_local);
    free(x_local);
    free(y_local);
    free(displs);
    free(sendcounts);
    free(recvcounts);
}

int getBlockSize(int *m_local, int *n_local, int sz) {
    int comm_sqrt = (int) sqrt(sz);
    for (int s = comm_sqrt; s > 0; --s) {
        if (sz % s == 0) {
            int q = sz / s;
            if (M % s == 0 && N % q == 0) {
                *m_local = M / s;
                *n_local = N / q;
                return 0;
            }
            if (M % q == 0 && N % s == 0) {
                *m_local = M / q;
                *n_local = N / s;
                return 0;
            }
        }
    }
    return -1;
}

void blockWise() {
    int my_rank, comm_sz;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    double *a = NULL;
    double *x = NULL;
    int n_local, m_local;
    if (getBlockSize(&m_local, &n_local, comm_sz) == -1) {
        if (my_rank == 0) {
            printf("Wrong number of proccesses - unable to divide matrix into blocks\n");
        }
        MPI_Finalize();
        free(a);
        free(x);
        return;
    }

    if (my_rank == 0) {
        a = (double *) malloc(sizeof(double) * M * N);
        x = (double *) malloc(sizeof(double) * N);
#ifdef CHECK
        fillToCheck(a, x, M, N);
#else
        fillRandomly(a, x, M, N);
#endif
    }

    double *a_local = (double *) malloc(sizeof(double) * m_local * n_local);
    double *x_local = (double *) malloc(sizeof(double) * n_local);
    double *y_local = (double *) malloc(sizeof(double) * m_local);
    double *y_group = (double *) malloc(sizeof(double) * m_local);
    double *y = (double *) malloc(sizeof(double) * M);

    double t = MPI_Wtime();

    int *displs = (int *) malloc(sizeof(int) * comm_sz);
    int *sendcounts = (int *) malloc(sizeof(int) * comm_sz);
    int *recvcounts = (int *) malloc(sizeof(int) * comm_sz);
    for (int rank = 0; rank < comm_sz; ++rank) {
        displs[rank] = N * m_local * (int) (rank / (N / n_local)) + n_local * (rank % (N / n_local));
        sendcounts[rank] = n_local;
        recvcounts[rank] = n_local;
    }
    for (int row = 0; row < m_local; ++row) {
        MPI_Scatterv(&a[row * N], sendcounts, displs, MPI_DOUBLE, &a_local[row * recvcounts[my_rank]],
                     recvcounts[my_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    MPI_Scatter(x, N / comm_sz + (N % comm_sz != 0), MPI_DOUBLE, x_local, N / comm_sz + (N % comm_sz != 0), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    double *x_full = (double *) malloc(sizeof(double) * N);
    MPI_Allgather(x_local, N / comm_sz + (N % comm_sz != 0), MPI_DOUBLE, x_full, N / comm_sz + (N % comm_sz != 0),
                  MPI_DOUBLE, MPI_COMM_WORLD);
    for (int i = 0; i < m_local; i++) {
        y_local[i] = 0.0;
        for (int j = 0; j < n_local; j++) {
            y_local[i] += a_local[i * n_local + j] * x_full[(int) (my_rank % (N / n_local)) * n_local + j];
        }
    }

    int my_row = (int) (my_rank / (M / m_local));
    MPI_Comm my_row_comm;
    MPI_Comm_split(MPI_COMM_WORLD, my_row, my_rank, &my_row_comm);
    MPI_Allreduce(y_local, y_group, m_local, MPI_DOUBLE, MPI_SUM, my_row_comm);

    int *displs_gather = (int *) malloc(sizeof(int) * comm_sz);
    int *recvcounts_gather = (int *) malloc(sizeof(int) * comm_sz);
    for (int rank = 0; rank < comm_sz; ++rank) {
        if (rank % (N / n_local) == 0) {
            displs_gather[rank] = (int) (rank / (N / n_local)) * m_local;
            recvcounts_gather[rank] = m_local;
        } else {
            displs_gather[rank] = 0;
            recvcounts_gather[rank] = 0;
        }
    }
    MPI_Gatherv(y_group, m_local, MPI_DOUBLE, y, recvcounts_gather, displs_gather, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    t = MPI_Wtime() - t;
    if (my_rank == 0) {
#ifdef PRINT
        printf("\nResult vector y:\n");
        for (int i = 0; i < M; ++i) {
            printf("%f\t", y[i]);
        }
#endif
        printf("\n\ntime - %lf\n", t);
    }

    MPI_Finalize();
    free(a);
    free(x);
    free(y);
    free(a_local);
    free(x_local);
    free(y_local);
    free(y_group);
    free(x_full);
    free(displs);
    free(sendcounts);
    free(recvcounts);
    free(displs_gather);
    free(recvcounts_gather);
}

int main() {
    srand(time(0));

    blockWise();

    return 0;
}
