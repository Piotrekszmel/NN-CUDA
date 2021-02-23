#pragma once

#include <cstdio>
#include <random>

#define gpuErrCheck(ans) {gpuAssert((ans), __FILE__, __LINE__); }

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);