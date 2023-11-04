#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <sys/time.h>
#include <omp.h>

#define G 6.67430e-11f
#define DELTA 1e-3f
#define STEPS 10
#define REPEATS 5
#define THREADS 4

#define EFFICIENT_THREADS_USE
#define MINIMUM_POINTS_PER_THREAD 1000

typedef struct Pair {
    float x;
    float y;
} Pair;

typedef struct Points {
    Pair* coords;
    Pair* velocities;
    int size;
} Points;

typedef struct PointsPaths {
    int points_number;
    int steps_number;
    float* time;
    float* masses;
    Points** points;
} PointsPaths;

typedef struct ThreadArgs {
    int start;
    int end;
    float delta;
    int current;
    Points* points;
    PointsPaths* paths;
} ThreadArgs;

PointsPaths* get_points_from_file(char* input_file, int steps) {
    FILE* input = fopen(input_file, "r");

    int n;
    fscanf(input, "%d", &n);

    PointsPaths* paths = malloc(sizeof(PointsPaths));
    paths->points_number = n;
    paths->time = malloc(steps * sizeof(float));
    paths->masses = malloc(n * sizeof(float));
    paths->points = malloc(steps * sizeof(Points));
    paths->steps_number = steps;

    Points* points = malloc(sizeof(Points));
    points->coords = malloc(n * sizeof(Pair));
    points->velocities = malloc(n * sizeof(Pair));
    points->size = n;

    for (int i = 0; i < n; ++i) {
        fscanf(input, "%f %f %f %f %f", &paths->masses[i],
               &points->coords[i].x, &points->coords[i].y,
               &points->velocities[i].x, &points->velocities[i].y);
    }
    paths->points[0] = points;

    fclose(input);

    return paths;
}

void create_output_csv_file(char* output_file, PointsPaths* paths) {
    FILE* output = fopen(output_file, "w");

    int n = paths->points_number;
    int iterations = paths->steps_number;
    fprintf(output, "timestamp,");
    for (int i = 0; i < n - 1; ++i) {
        fprintf(output, "x%d,y%d,", i + 1, i + 1);
    }
    fprintf(output, "x%d,y%d\n", n, n);
    for (int i = 0; i < iterations; ++i) {
        fprintf(output, "%f,", paths->time[i]);
        Points* point = paths->points[i];
        for (int j = 0; j < n; ++j) {
            fprintf(output, "%f,%f", point->coords[j].x, point->coords[j].y);
            if (j != n - 1) {
                fprintf(output, ",");
            }
        }
        fprintf(output, "\n");
    }

    fclose(output);
}

Pair calculate_F(Points* q, int qId, const float* masses) {
    int n = q->size;
    float current_x = q->coords[qId].x;
    float current_y = q->coords[qId].y;
    float fx = 0, fy = 0;
    for (int i = 0; i < n; ++i) {
        float dis_x = q->coords[i].x - current_x;
        float dis_y = q->coords[i].y - current_y;
        float dis = dis_x * dis_x + dis_y * dis_y;
        if (dis >= 1e-6) {
            float dis_cubed = dis * sqrtf(dis);
            fx += masses[i] / dis_cubed * dis_x;
            fy += masses[i] / dis_cubed * dis_y;
        }
    }
    fx *= G * masses[qId];
    fy *= G * masses[qId];

    return (Pair) {fx, fy};
}

void calculate_single_threaded(float delta, int steps, PointsPaths* paths) {
    int n = paths->points_number;
    for (int current = 1; current < steps; ++current) {
        Points* points = (Points*) malloc(sizeof(Points));
        points->coords = malloc(n * sizeof(Pair));
        points->velocities = malloc(n * sizeof(Pair));
        points->size = n;

        for (int i = 0; i < n; ++i) {
            float fx, fy;
            Points* previous_points = paths->points[current - 1];
            Pair result = calculate_F(previous_points, i, paths->masses);
            fx = result.x;
            fy = result.y;
            points->coords[i].x = previous_points->coords[i].x + previous_points->velocities[i].x * delta;
            points->coords[i].y = previous_points->coords[i].y + previous_points->velocities[i].y * delta;
            points->velocities[i].x = previous_points->velocities[i].x + fx / paths->masses[i] * delta;
            points->velocities[i].y = previous_points->velocities[i].y + fy / paths->masses[i] * delta;
        }

        paths->time[current] = paths->time[current - 1] + delta;
        paths->points[current] = points;
    }
}

void calculate_with_omp(float delta, int steps, PointsPaths* paths) {
    int n = paths->points_number;
    for (int current = 1; current < steps; ++current) {
        Points* points = (Points*) malloc(sizeof(Points));
        points->coords = malloc(n * sizeof(Pair));
        points->velocities = malloc(n * sizeof(Pair));
        points->size = n;

#pragma omp parallel for default(none) shared(n, delta, current, points, paths)
        for (int i = 0; i < n; ++i) {
            float fx, fy;
            Points* previous_points = paths->points[current - 1];
            Pair result = calculate_F(previous_points, i, paths->masses);
            fx = result.x;
            fy = result.y;
            points->coords[i].x = previous_points->coords[i].x + previous_points->velocities[i].x * delta;
            points->coords[i].y = previous_points->coords[i].y + previous_points->velocities[i].y * delta;
            points->velocities[i].x = previous_points->velocities[i].x + fx / paths->masses[i] * delta;
            points->velocities[i].y = previous_points->velocities[i].y + fy / paths->masses[i] * delta;
        }

        paths->time[current] = paths->time[current - 1] + delta;
        paths->points[current] = points;
    }
}

void* thread_routine(void* args) {
    ThreadArgs* casted_args = (ThreadArgs*) args;
    int start = casted_args->start;
    int end = casted_args->end;
    float delta = casted_args->delta;
    int current = casted_args->current;
    Points* points = casted_args->points;
    PointsPaths* paths = casted_args->paths;

    for (int i = start; i < end; ++i) {
        float fx, fy;
        Points* previous_points = paths->points[current - 1];
        Pair result = calculate_F(previous_points, i, paths->masses);
        fx = result.x;
        fy = result.y;
        points->coords[i].x = previous_points->coords[i].x + previous_points->velocities[i].x * delta;
        points->coords[i].y = previous_points->coords[i].y + previous_points->velocities[i].y * delta;
        points->velocities[i].x = previous_points->velocities[i].x + fx / paths->masses[i] * delta;
        points->velocities[i].y = previous_points->velocities[i].y + fy / paths->masses[i] * delta;
    }

    return 0;
}

void calculate_multi_threaded(float delta, int steps, PointsPaths* paths, pthread_t* thread_handles, int thread_count) {
    ThreadArgs thread_args[thread_count];

    int n = paths->points_number;
    for (int current = 1; current < steps; ++current) {
        Points* points = (Points*) malloc(sizeof(Points));
        points->coords = malloc(n * sizeof(Pair));
        points->velocities = malloc(n * sizeof(Pair));
        points->size = n;

        int thread_to_start = 0;

#ifdef EFFICIENT_THREADS_USE
        int chunk_size = n / thread_count;
        while (chunk_size != n && chunk_size < MINIMUM_POINTS_PER_THREAD) {
            thread_count--;
            chunk_size = n / thread_count;
        }
#else
        int chunk_size = n / thread_count;
#endif

        int remainder = n - chunk_size * thread_count;
        for (int start = 0, end = chunk_size + remainder;
             start < n;
             start = end, end = start + chunk_size) {
            thread_args[thread_to_start].start = start;
            thread_args[thread_to_start].end = end;
            thread_args[thread_to_start].delta = delta;
            thread_args[thread_to_start].current = current;
            thread_args[thread_to_start].points = points;
            thread_args[thread_to_start].paths = paths;
            pthread_create(&thread_handles[thread_to_start], 0, thread_routine, (void*) &thread_args[thread_to_start]);
            thread_to_start++;
        }
        for (int i = 0; i < thread_to_start; ++i) {
            pthread_join(thread_handles[i], 0);
        }

        paths->time[current] = paths->time[current - 1] + delta;
        paths->points[current] = points;
    }
}

void release(PointsPaths* paths) {
    for (int i = 0; i < paths->steps_number; ++i) {
        free(paths->points[i]->coords);
        free(paths->points[i]->velocities);
        free(paths->points[i]);
    }
    free(paths->points);
    free(paths->masses);
    free(paths->time);
}

void do_simulation(char* input_file, char* output_file, float delta, int steps, int repeats, int thread_count) {
    omp_set_num_threads(thread_count);

    struct timeval begin, end;
    long long avg_ms = 0;

    PointsPaths* paths;
    for (int i = 0; i < repeats; ++i) {
        paths = get_points_from_file(input_file, steps);
        pthread_t* thread_handles = malloc(thread_count * sizeof(pthread_t));

        gettimeofday(&begin, 0);
        calculate_multi_threaded(delta, steps, paths, thread_handles, thread_count);
        gettimeofday(&end, 0);

        long long elapsed_ms = (end.tv_sec - begin.tv_sec) * 1000 + (end.tv_usec - begin.tv_usec) / 1000;
        printf("repeat %d took %lld ms\n", i + 1, elapsed_ms);
        avg_ms += elapsed_ms;

        if (i == repeats - 1) {
            create_output_csv_file(output_file, paths);
        }
        free(thread_handles);
        release(paths);
    }

    avg_ms /= repeats;
    printf("avg %llu ms", avg_ms);
}

int main() {
    char* input = "../benchmark/10k.txt";
    char* output = "../output.csv";

    do_simulation(input, output, DELTA, STEPS, REPEATS, THREADS);

    return 0;
}
