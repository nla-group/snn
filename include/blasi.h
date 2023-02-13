/*
MIT License

Copyright (c) 2022 Stefan Güttel, Xinye Chen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

*/


#ifndef BLASI_H
#define BLASI_H

// #include "blasi.cpp"
template<typename T> T 
matrix_major_col_index(const T *mat, const int *row, const int *col, const int *rows);

// BLAS Level 1
template<typename T> void 
vector_scalar_add(const T *arr, const T *scalar, T *ret, const int *size, const int *start, const int *end);

template<typename T1, typename T2> void 
vector_scalar_sub(const T1 *arr, const T2 *scalar, T1 *ret, const int *size, const int *start, const int *end);


template<typename T> void 
vector_scalar_divide(const T *arr, const T *scalar, T *ret, const int *size, const int *start, const int *end);

template<typename T> void 
vector_scalar_multi(const T *arr, const T *scalar, T *ret, const int *size, const int *start, const int *end);


template<typename T1, typename T2, typename T3> void 
vector_vector_sub(T1 *arr1, T2 *arr2, T3 *ret, const int *size);

template<typename T1, typename T2, typename T3> void 
vector_inner_prod(const T1 *arr1, const T2 *arr2, T3 *ret, const int *size);

template<typename T> void 
filter_less(const T *arr, const T *filterVal, bool *ret, const int *size);



// BLAS Level 2

template<typename T> void 
extern matrix_vector_prod(const T *mat, const T *arr, T *ret, const int *rows, const int *cols);


void // double limit
blas_matrix_vector_prod1(const double *mat, const double *arr, double *ret, const int *rows, const int *cols);

template<typename T1, typename T2> void // trivial implementation
blas_matrix_vector_prod2(const T1*mat, const T2 *arr, T1 *ret, const int *rows, const int *cols,
                         const double *alpha, const int *start, const int *end);


template<typename T> void 
extern blas_norm_matrix(const T *mat, T *ret, const int *rows, const int *cols);


void
blas_matrix_inner_prod(const double *mat, double *ret, const int *rows, const int *cols);

template<typename T> void 
calculate_matrix_mean(const T *mat, double *ret, const int *rows, const int *cols);


template<typename T> void 
calculate_matrix_std(const T *mat, const double *mu, double *ret, const int *rows, const int *cols);


void
calculate_skip_euclid_norm(const double *xxt, const double *mat, const double *arr, double *ret,
                             const int *rows, const int *cols, const int *start, const int *end);

void
calculate_skip_euclid_norm(const double *xxt, const double *mat, const float *arr, double *ret,
                             const int *rows, const int *cols, const int *start, const int *end);
template<typename T> T * 
selectArray(const T *arr, const int start, const int end);


#endif // BLASI_H