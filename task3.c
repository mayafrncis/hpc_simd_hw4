#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <string.h>
#include <immintrin.h>

#define NUM_THREADS 4

typedef struct {
	int width;
	int height;
	unsigned char *data;
} image;

typedef struct {
	int start;
	int end;
	unsigned char *input;
	unsigned char *output;
} threadData;

image read_ppm(char *filename) {
	FILE *fp = fopen(filename, "rb");
	if (fp == NULL) {
		perror("PPM file opening");
		exit(1);
	}
	char format[4]; // Expecting P6\0
	int width, height, max_val;
	fscanf(fp, "%s\n%d %d\n%d\n", format, &width, &height, &max_val);

	size_t size = (size_t)width * height * 3;

	unsigned char *data = (unsigned char *) malloc(size);

	fread(data, 1, size, fp);
	fclose(fp);
	return (image){width, height, data};
}

void write_ppm(char *filename, image i) {
	FILE *fp = fopen(filename, "wb");
	fprintf(fp, "P6\n%d %d\n255\n", i.width, i.height);
	fwrite(i.data, 1, (size_t)i.width * i.height * 3, fp);
	fclose(fp);
}

double get_time() {
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void scalar(int start, int end, unsigned char *input, unsigned char *output) {
	for (size_t i = start; i < end ; i++) { // i is in terms of pixels
		int index = i * 3; // index is in terms of bytes
		unsigned char r = input[index];
		unsigned char g = input[index + 1];
		unsigned char b = input[index + 2];
		unsigned char grey = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);

		output[index] = grey;
		output[index + 1] = grey;
		output[index + 2] = grey;
	}
}

void *worker_scalar(void *args) {
	threadData data = *((threadData *) args);
	scalar(data.start, data.end, data.input, data.output);
	return NULL;
}

void simd(int start, int end, unsigned char *input, unsigned char *output) {
	//size_t num_pixels = (size_t)width * height;
	__m256 r_weight = _mm256_set1_ps(0.299f);
	__m256 g_weight = _mm256_set1_ps(0.587f);
	__m256 b_weight = _mm256_set1_ps(0.114f);
	size_t i = (size_t)start;
	for (; i + 8 <= (size_t)end; i += 8) { // We can only process 8 pixels at a time because 256 / 32 (size of a float) = 8
		// We need to separate out each colour into its own vector so the weight can be applied on it separately
		unsigned char *p = &input[i * 3];
		float r_arr[8], g_arr[8], b_arr[8];
		for(int j=0; j<8; j++) {
			r_arr[j] = (float)p[j*3];
			g_arr[j] = (float)p[j*3+1];
			b_arr[j] = (float)p[j*3+2];
		}

		__m256 r = _mm256_loadu_ps(r_arr);
		__m256 g = _mm256_loadu_ps(g_arr);
		__m256 b = _mm256_loadu_ps(b_arr);
		__m256 grey = _mm256_add_ps(_mm256_mul_ps(r, r_weight),_mm256_add_ps(_mm256_mul_ps(g, g_weight),_mm256_mul_ps(b, b_weight)));
		float res[8];
		_mm256_storeu_ps(res, grey);

		for(int j=0; j<8; j++) {
			output[(i+j)*3] = (unsigned char)res[j];
			output[(i+j)*3+1] = (unsigned char)res[j];
			output[(i+j)*3+2] = (unsigned char)res[j];
		}

	}

	for (; i < end; i++) {
		int index = i * 3;
		unsigned char grey = (unsigned char)((0.299f * (float)input[index]) + ((0.587f * (float)input[index + 1]) + (0.114f * (float)input[index + 2])));
		output[index] = output[index+1] = output[index+2] = grey;
	}
}

void *worker_simd(void *args) {
	threadData *data = ((threadData *) args);
	simd(data->start, data->end, data->input, data->output);
	return NULL;
}

int main() {
	image img = read_ppm("input.ppm");
	size_t num_pixels = img.width * img.height;
	size_t data_size = num_pixels * 3;
	unsigned char *scalar_output = (unsigned char *) malloc(data_size);
	if (scalar_output == NULL) {
		perror("Malloc");
		return 1;
	}

	printf("Image size: %d x %d\n", img.width, img.height);
	printf("Threads used: %d\n\n", NUM_THREADS);

	printf("Scalar time:  ");

	double start = get_time();
	scalar(0, num_pixels, img.data, scalar_output);
	double end = get_time();
	printf("%f sec\n", end - start);

	unsigned char *scalar_thread_output = (unsigned char *) malloc(data_size);
	if (scalar_thread_output == NULL) {
		perror("Malloc");
		free(scalar_output);
		return 1;
	}

	pthread_t threads[NUM_THREADS];
	threadData data[NUM_THREADS];

	int chunk = num_pixels / NUM_THREADS;

	start = get_time();

	for (int i = 0; i < NUM_THREADS; i++) {
		data[i].start = chunk * i;
		if (i == NUM_THREADS - 1) {
			data[i].end = num_pixels;
		} else {
			data[i].end = (i + 1) * chunk;
		}
		data[i].input = img.data;
		data[i].output = scalar_thread_output;
		if (pthread_create(&threads[i], NULL, worker_scalar, &data[i]) != 0) {
			perror("Thread creation");
			free(scalar_output);
			return 1;
		}
	}

	for (int i = 0; i < NUM_THREADS; i++) {
		pthread_join(threads[i], NULL);
	}

	end = get_time();
	printf("Multithreading:  ");
	printf("%f sec\n", end - start);

	int verify_multithread = memcmp(scalar_thread_output, scalar_output, data_size); // returns 0 if they're identical

	unsigned char *simd_output = (unsigned char *) malloc(data_size);
	if (simd_output == NULL) {
		perror("Malloc");
		free(scalar_output);
		free(scalar_thread_output);
		return 1;
        }

	start = get_time();
	simd(0, num_pixels, img.data, simd_output);
	end = get_time();
	printf("SIMD:  ");
	printf("%f sec\n", end - start);

	int verify_simd = memcmp(simd_output, scalar_output, data_size);

	unsigned char *simd_thread_output = (unsigned char *) malloc(data_size);
	if (simd_thread_output == NULL) {
		perror("Malloc");
		free(simd_output);
		free(scalar_output);
		free(scalar_thread_output);
		return 1;
	}

	start = get_time();
	for (int i = 0; i < NUM_THREADS; i++) {
                data[i].output = simd_thread_output;
                if (pthread_create(&threads[i], NULL, worker_simd, &data[i]) != 0) {
                        perror("Thread creation");
			free(simd_thread_output);
			free(simd_output);
			free(scalar_output);
			free(scalar_thread_output);
                        return 1;
                }
        }

	for (int i = 0; i < NUM_THREADS; i++) {
		pthread_join(threads[i], NULL);
	}

	end = get_time();
	printf("SIMD with Multithreading:  ");
	printf("%f sec\n\n", end - start);

	int verify_multithread_simd = memcmp(simd_thread_output, scalar_output, data_size);

	if (verify_multithread_simd == 0 && verify_simd == 0) {
		printf("Verification: PASSED\n");
	} else {
		printf("Verification: failed\n");
	}
	image output_img;
	output_img.width = img.width;
	output_img.height = img.height;
	output_img.data = simd_output;

	write_ppm("output.ppm", output_img);
	printf("Output image: output.ppm\n");
	free(simd_thread_output);
	free(simd_output);
	free(scalar_output);
	free(scalar_thread_output);
	return 0;
}

