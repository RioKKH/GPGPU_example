#include <stdio.h>
#include <cuda.h>
#include <thrust/random.h>
#include <thrust/generate.h>

#define POPSIZE	   16
#define CHROMOSOME	8

int my_rand(void)
{
	static thrust::default_random_engine rng;
	static thrust::uniform_int_distribution<int> dist(0, 1);

	return dist(rng);
}

void init(int *population)
{
	thrust::generate(population, population + POPSIZE * CHROMOSOME, my_rand);
}

int main()
{
	int *population;
	population = (int *)malloc(POPSIZE * CHROMOSOME * sizeof(int));
	init(population);
	for (int i = 0; i < POPSIZE; ++i)
	// for (int i = 0; i < N; ++i)
	{
		for (int j = 0; j < CHROMOSOME; ++j)
		// for (int j = 0; j < M; ++j)
		{
			printf("%d", population[i * CHROMOSOME + j]);
			// printf("%d", population[i * N + j]);
		}
		printf("\n");
	}

	return 0;
}
