#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <string.h>
#include <immintrin.h>

pthread_mutex_t mutex;
int counter[4]; // A C G T order

#define SIZE (256ULL * 1024 * 1024) // 256 MB
#define THREAD_NUM 4
char *buffer;

typedef struct {
	char *buffer;
	size_t size;
	long long local_counter[4];
} threadData;

void count_scalar(char *buffer, size_t size) {
	for (size_t i = 0; i < size; i++) {
		if (buffer[i] == 'A') counter[0]++;
		else if (buffer[i] == 'C') counter[1]++;
		else if (buffer[i] == 'G') counter[2]++;
		else if (buffer[i] == 'T') counter[3]++;
	}
	return NULL;
}

void count_simd(char *buffer, size_t size, long long *local_counter) { //AVX2 is 256-bit based and a char is 8 bits, so we can process 32 chars as a time
	__m256i a = _mm256_set1_epi8('A');
	__m256i c = _mm256_set1_epi8('C');
	__m256i g = _mm256_set1_epi8('G');
	__m256i t = _mm256_set1_epi8('T');
	size_t i = 0;
	for (; i < size - 32; i += 32) {
		__m256i chars = _mm256_loadu_si256((__m256i*)&buffer[i]);
		__m256i res_a = _mm256_cmpeq_epi8(chars, a);
		__m256i res_c = _mm256_cmpeq_epi8(chars, c);
		__m256i res_g = _mm256_cmpeq_epi8(chars, g);
		__m256i res_t_ = _mm256_cmpeq_epi8(chars, t);

		local_counter[0] += __builtin_popcount(_mm256_movemask_epi8(res_a));
		local_counter[1] += __builtin_popcount(_mm256_movemask_epi8(res_c));
		local_counter[2] += __builtin_popcount(_mm256_movemask_epi8(res_g));
		local_counter[3] += __builtin_popcount(_mm256_movemask_epi8(res_t_));
	}

	for (;i < size; i++) {
		if (buffer[i] == 'A') local_counter[0]++;
		else if (buffer[i] == 'C') local_counter[1]++;
		else if (buffer[i] == 'G') local_counter[2]++;
		else if (buffer[i] == 'T') local_counter[3]++;
	}

	pthread_mutex_lock(&mutex);
	for (int j = 0; j < 4; j++) counter[j] += local_counter[j];
	pthread_mutex_unlock(&mutex);

	return NULL;
}


void *worker_scalar(void *args) {
	threadData data = *(threadData *)args;
	for (size_t i = 0; i < data.size; i++) {
		if (data.buffer[i] == 'A') data.local_counter[0]++;
		else if (data.buffer[i] == 'C') data.local_counter[1]++;
		else if (data.buffer[i] == 'G') data.local_counter[2]++;
		else if (data.buffer[i] == 'T') data.local_counter[3]++;
	}

	pthread_mutex_lock(&mutex);
	for (int j = 0; j < 4; j++) counter[j] += data.local_counter[j];
	pthread_mutex_unlock(&mutex);
	return NULL;
}

void *worker_simd(void *args) {
	threadData data = *(threadData *)args;
	long long counter[4] = {0,0,0,0};
	count_simd(data.buffer, data.size, counter);
	return NULL;
}

double get_time() {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
	char nucleo[] = "ACGT";
	buffer = (char *) malloc(SIZE);
	if (buffer == NULL) {
		perror("Malloc");
		return 1;
	}
	for (size_t i = 0; i < SIZE; i++) buffer[i] = nucleo[rand() % 4];

	for (int i = 0; i < 4; i++) {
		counter[i] = 0;
	}
	pthread_mutex_init(&mutex, NULL);

	printf("Scalar\n");
	double start = get_time();
	count_scalar(buffer, SIZE);
	double end = get_time();
	printf("%d	%d	%d	%d\n", counter[0],counter[1],counter[2],counter[3]);
	printf("%f sec\n", end - start);
	for (int i = 0; i < 4; i++) {
		counter[i] = 0;
        }

	printf("Multithreading\n");
	start = get_time();
	pthread_t threads[THREAD_NUM];
	threadData data[THREAD_NUM] = {0};
	for (int i = 0; i < THREAD_NUM; i++) {
		size_t chunk_size  = (SIZE / THREAD_NUM);
		data[i].buffer = buffer + (i * chunk_size);
		if (i == THREAD_NUM - 1) {
			data[i].size = SIZE - (i * chunk_size);
		} else {
			data[i].size = chunk_size;
		}
		if (pthread_create(&threads[i], NULL, worker_scalar, &data[i]) != 0) {
			perror("Thread creation");
			return -1;
		}
	}
	for (int i = 0; i < THREAD_NUM; i++) {
		if (pthread_join(threads[i], NULL) != 0) {
			perror("Thread join");
			return -1;
		}
	}
	end = get_time();
	printf("%d      %d      %d      %d\n", counter[0],counter[1],counter[2],counter[3]);
	printf("%f sec\n", end - start);

	for (int i = 0; i < 4; i++) counter[i] = 0;

	printf("SIMD\n");
	start = get_time();
	long long local_counter[4] = {0};
	count_simd(buffer, SIZE,local_counter);
	end = get_time();
	printf("%d      %d      %d      %d\n", counter[0],counter[1],counter[2],counter[3]);
	printf("%f sec\n", end - start);

	for (int i = 0; i < 4; i++) counter[i] = 0;

	printf("SIMD with Multithreading\n");
	start = get_time();
	for (int i = 0; i < THREAD_NUM; i++) {
		if (pthread_create(&threads[i], NULL, worker_simd, &data[i]) != 0) {
			perror("Thread creation");
			return -1;
		}
	}
	for (int i = 0; i < THREAD_NUM; i++) {
		if (pthread_join(threads[i], NULL) != 0) {
			perror("Thread join");
			return -1;
		}
	}
	end = get_time();
	printf("%d      %d      %d      %d\n", counter[0],counter[1],counter[2],counter[3]);
	printf("%f sec\n", end - start);
	
	return 0;
}


