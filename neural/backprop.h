#pragma once

#include "misc.h"

#include "network.h"

typedef struct BP_Error
{
	uint size;
	real *error;
	real *buffer;
}
BP_Error;

typedef struct BP_Gradient
{
	uint input_size;
	uint output_size;
	real *grad_weight;
	real *grad_bias;
	real *buffer;
}
BP_Gradient;

typedef struct BP_Buffer
{
	uint depth;
	BP_Error    *error;
	BP_Gradient *gradient;
}
BP_Buffer;

#ifdef __cplusplus
extern "C" {
#endif

void BP_randomize(Network *network, uint seed);
void BP_shuffle(uint length, void **array, uint seed);

BP_Buffer *BP_createBuffer(const Network *network);
void BP_destroyBuffer(BP_Buffer *buffer);

real BP_computeCost(const Network *network, const real *result);
void BP_computeError(const Network *network, BP_Buffer *buffer, const real *result);

void BP_addGradient(const Network *network, BP_Buffer *buffer);
void BP_normalizeGradient(BP_Buffer *buffer, uint sample_length);
void BP_clearGradient(BP_Buffer *buffer);

void BP_performDescent(Network *network, BP_Buffer *buffer, const real rate);

#ifdef __cplusplus
}
#endif

