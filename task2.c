#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <string.h>
#include <immintrin.h>

//#define SIZE 200
#define SIZE (256ULL * 1024 * 1024)
#define THREAD_NUM 4

typedef struct {
	char *buffer;
	size_t size;
} threadData;

void simd_upper(char *buffer, size_t size) {
	__m256i small_bound = _mm256_set1_epi8('a' - 1);
	__m256i large_bound = _mm256_set1_epi8('z' + 1);
	__m256i thirty_two = _mm256_set1_epi8(32);

	size_t i = 0;
	for (; i <= size - 32; i += 32) {
		__m256i chars = _mm256_loadu_si256((__m256i*)&buffer[i]);
		__m256i gt_a = _mm256_cmpgt_epi8(chars, small_bound);
		__m256i lt_z = _mm256_cmpgt_epi8(large_bound, chars);
		__m256i mask = _mm256_and_si256(gt_a, lt_z);
		__m256i lower = _mm256_and_si256(mask, thirty_two);
		chars = _mm256_sub_epi8(chars, lower);

		_mm256_storeu_si256((__m256i*)&buffer[i], chars);

	}

	for (; i < size; i++) {
		if (buffer[i] >= 'a' && buffer[i] <= 'z') buffer[i] -= 32;
	}
}

void *scalar(void *args) {
	threadData data = *(threadData *)args;
	for (size_t i = 0; i < data.size; i++) {
		if (data.buffer[i] >= 'a' && data.buffer[i] <= 'z') data.buffer[i] -= 32;
	}

	return NULL;
}

void *worker_simd(void *args) {
	threadData data = *(threadData *)args;
	simd_upper(data.buffer, data.size);

	return NULL;
}

double get_time() {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
	char *buffer = (char *) malloc(SIZE);
	if (buffer == NULL) {
		perror("Malloc");
		return 1;
	}
	char *buffer2 = (char *) malloc(SIZE);
	if (buffer2 == NULL) {
		perror("Malloc");
		return 1;
		free(buffer);
	}
	char *buffer3 = (char *) malloc(SIZE);
	if (buffer3 == NULL) {
		perror("Malloc");
		free(buffer);
		free(buffer2);
		return 1;
	}
	const char *all_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+[]{}|;:,.<>?/ ";
	for (int i = 0; i < SIZE; i++) buffer[i] = all_chars[rand() % strlen(all_chars)];
	memcpy(buffer2, buffer, SIZE);
	memcpy(buffer3, buffer, SIZE);

	//printf("Original: %s\n", buffer);

	printf("Multithreading\n");
	double start = get_time();
	pthread_t threads[THREAD_NUM];
	threadData data[THREAD_NUM];
	for (int i = 0; i < THREAD_NUM; i++) {
		size_t chunk_size  = (SIZE / THREAD_NUM);
		data[i].buffer = buffer + (i * chunk_size);
		if (i == THREAD_NUM - 1) {
			data[i].size = SIZE - (i * chunk_size);
		} else {
			data[i].size = chunk_size;
		}
		if (pthread_create(&threads[i], NULL, scalar, &data[i]) != 0) {
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
	double end = get_time();
	//printf("%s\n", buffer);
	printf("%f sec\n\n", end - start);

	start = get_time();
	simd_upper(buffer2, SIZE);
	end = get_time();
	//printf("%s\n", buffer2);
	printf("%f sec\n\n", end - start);


	printf("Multithreading with SIMD\n");
	start = get_time();
	for (int i = 0; i < THREAD_NUM; i++) {
		size_t chunk_size  = (SIZE / THREAD_NUM);
		data[i].buffer = buffer3 + (i * chunk_size);
		if (i == THREAD_NUM - 1) {
			data[i].size = SIZE - (i * chunk_size);
		} else {
			data[i].size = chunk_size;
		}
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
	//printf("%s\n", buffer2);
	printf("%f sec\n\n", end - start);
	return 0;
}
