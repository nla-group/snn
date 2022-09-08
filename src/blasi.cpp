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

#include "blasi.h"
#include <algorithm> 
#include <cmath> 
#include "cblas.h"

template<typename T> T 
matrix_major_col_index(const T *mat, const int *row, const int *col, const int *rows){
    return mat[*col* (*rows) + *row];
}

template int 
matrix_major_col_index<int>(const int *mat, const int *row, const int *col, const int *rows);

template float 
matrix_major_col_index<float>(const float *mat, const int *row, const int *col, const int *rows);

template double 
matrix_major_col_index<double>(const double *mat, const int *row, const int *col, const int *rows);


// BLAS Level 1
template<typename T> void 
vector_scalar_add(const T *arr, const T *scalar, T *ret, const int *size, const int *start, const int *end){
    for (int i=*start; i<std::min(*size, *end); i++){
        *(ret + i) = *(arr + i) +  *scalar;
    }
}

template void 
vector_scalar_add<int>(const int *arr, const int *scalar, int *ret, const int *size, const int *start, const int *end);

template void 
vector_scalar_add<float>(const float *arr, const float *scalar, float *ret, const int *size, const int *start, const int *end);

template void 
vector_scalar_add<double>(const double *arr, const double *scalar, double *ret, const int *size, const int *start, const int *end);


template<typename T1, typename T2> void 
vector_scalar_sub(const T1 *arr, const T2 *scalar, T1 *ret, const int *size, const int *start, const int *end){
    for (int i=*start; i<std::min(*size, *end); i++){
        *(ret + i) = *(arr + i) -  *scalar;
    }
}

template void 
vector_scalar_sub<int, int>(const int *arr, const int *scalar, int *ret, const int *size, const int *start, const int *end);

template void 
vector_scalar_sub<float, float>(const float *arr, const float *scalar, float *ret, const int *size, const int *start, const int *end);

template void 
vector_scalar_sub<double, double>(const double *arr, const double *scalar, double *ret, const int *size, const int *start, const int *end);

template void 
vector_scalar_sub<float, double>(const float *arr, const double *scalar, float *ret, const int *size, const int *start, const int *end);


template<typename T> void 
vector_scalar_divide(const T *arr, const T *scalar, T *ret, const int *size, const int *start, const int *end){
    for (int i=*start; i<std::min(*size, *end); i++){
        *(ret + i) = *(arr + i) /  *scalar;
    }
}

template void 
vector_scalar_divide<int>(const int *arr, const int *scalar, int *ret, const int *size, const int *start, const int *end);

template void 
vector_scalar_divide<float>(const float *arr, const float *scalar, float *ret, const int *size, const int *start, const int *end);

template void 
vector_scalar_divide<double>(const double *arr, const double *scalar, double *ret, const int *size, const int *start, const int *end);


template<typename T> void 
vector_scalar_multi(const T *arr, const T *scalar, T *ret, const int *size, const int *start, const int *end){
    for (int i=*start; i<std::min(*size, *end); i++){
        *(ret + i) = *(arr + i) *  (*scalar);
    }
}


template void 
vector_scalar_multi<int>(const int *arr, const int *scalar, int *ret, const int *size, const int *start, const int *end);

template void 
vector_scalar_multi<float>(const float *arr, const float *scalar, float *ret, const int *size, const int *start, const int *end);

template void 
vector_scalar_multi<double>(const double *arr, const double *scalar, double *ret, const int *size, const int *start, const int *end);


template<typename T1, typename T2, typename T3> void 
vector_vector_sub(T1 *arr1, T2 *arr2, T3 *ret, const int *size){
    for (int i=0; i<*size; i++){
        *(ret + i) = *(arr1 + i) - *(arr2 + i);
    }
}

template void 
vector_vector_sub<int, int, int>(int *arr1, int *arr2, int *ret, const int *size);

template void 
vector_vector_sub<float, float, float>(float *arr1, float *arr2, float *ret, const int *size);

template void 
vector_vector_sub<double, double, double>(double *arr1, double *arr2, double *ret, const int *size);

template void 
vector_vector_sub<int, double, int>(int *arr1, double *arr2, int *ret, const int *size);

template void 
vector_vector_sub<float, double, float>(float *arr1, double *arr2, float *ret, const int *size);

template void 
vector_vector_sub<double, double, float>(double *arr1, double *arr2, float *ret, const int *size);


template<typename T1, typename T2, typename T3> void 
vector_inner_prod(const T1 *arr1, const T2 *arr2, T3 *ret, const int *size){
    *ret = 0;
    for (int i=0; i<*size; i++){
        *ret += *(arr1 + i) * *(arr2 + i);
    }
}


template void 
vector_inner_prod<int, int, int>(const int *arr1, const int *arr2, int *ret, const int *size);

template void 
vector_inner_prod<float, float, float>(const float *arr1, const float *arr2, float *ret, const int *size);

template void 
vector_inner_prod<double, double, double>(const double *arr1, const double *arr2, double *ret, const int *size);

template void 
vector_inner_prod<float, double, double>(const float *arr1, const double *arr2, double *ret, const int *size);

template void 
vector_inner_prod<float, float, double>(const float *arr1, const float *arr2, double *ret, const int *size);


template<typename T> void 
filter_less(const T *arr, const T *filterVal, bool *ret, const int *size){
    for (int i=0; i<*size;i++){
        if (*(arr+i) <= (*filterVal)){
            ret[i] = true;
        }else{
            ret[i] = false;
        }
    }
}


template void 
filter_less<int>(const int *arr, const int *filterVal, bool *ret, const int *size);

template void 
filter_less<float>(const float *arr, const float *filterVal, bool *ret, const int *size);

template void 
filter_less<double>(const double *arr, const double *filterVal, bool *ret, const int *size);


// BLAS Level 2

template<typename T> void 
matrix_vector_prod(const T *mat, const T *arr, T *ret, const int *rows, const int *cols){
    // column-major order
    for (int row=0; row<*rows; row++){
        *(ret+row) = 0;
        for (int  col=0; col<*cols; col++){
            *(ret+row) += matrix_major_col_index(mat, &row, &col, rows) * arr[col];
        }
    }
}

template void 
matrix_vector_prod<int>(const int *mat, const int *arr, int *ret, const int *rows, const int *cols);

template void 
matrix_vector_prod<float>(const float *mat, const float *arr, float *ret, const int *rows, const int *cols);

template void 
matrix_vector_prod<double>(const double *mat, const double *arr, double *ret, const int *rows, const int *cols);


void // double limit
blas_matrix_vector_prod1(const double *mat, const double *arr, double *ret, const int *rows, const int *cols){
    cblas_dgemv(CblasColMajor, CblasNoTrans, *rows, *cols, 1, mat, *rows, arr, 1, 0, ret, 1);
}


template<typename T1, typename T2> void // trivial implementation
blas_matrix_vector_prod2(const T1 *mat, const T2 *arr, T1 *ret, const int *rows, const int *cols, const double *alpha, const int *start, const int *end){
    for (int i=*start; i<std::min(*rows, *end); i++){
        *(ret + i) = 0.0;
        for (int j=0; j<*cols; j++){
            *(ret + i) += *alpha * *(mat + i + j* (*rows)) * *(arr + j);
        }
    }
}


template void 
blas_matrix_vector_prod2<int, int>(const int *mat, const int *arr, int *ret, const int *rows, const int *cols,
                                             const double *alpha, const int *start, const int *end);
template void 
blas_matrix_vector_prod2<float, float>(const float *mat, const float *arr, float *ret, const int *rows, const int *cols, 
                                            const double *alpha, const int *start, const int *end);
template void 
blas_matrix_vector_prod2<double, double>(const double *mat, const double *arr, double *ret, const int *rows, const int *cols,
                                             const double *alpha, const int *start, const int *end);
template void 
blas_matrix_vector_prod2<double, float>(const double *mat, const float *arr, double *ret, const int *rows, const int *cols,
                                             const double *alpha, const int *start, const int *end);

template<typename T> void 
blas_norm_matrix(const T *mat, T *ret, const int *rows, const int *cols){
    T *temp = new T;
    
    for (int i=0; i<*rows; i++){
        // std::copy(mat + i* (*cols), mat + (i + 1)* (*cols), temp);
        // *(ret + i) = cblas_dnrm2(*cols, temp, 1);
        *temp  = 0.;
        for (int j=0; j<*cols; j++){
            *temp = *temp + pow(mat[j * (*rows) + i], 2);
        }
        *(ret + i) = *temp;
    }
}

template void 
blas_norm_matrix<int>(const int *mat, int *ret, const int *rows, const int *cols);

template void 
blas_norm_matrix<float>(const float *mat, float *ret, const int *rows, const int *cols);

template void 
blas_norm_matrix<double>(const double *mat, double *ret, const int *rows, const int *cols);



void // double limit
blas_matrix_inner_prod(const double *mat, double *ret, const int *rows, const int *cols){
    cblas_dgemm (CblasColMajor, 
        CblasTrans, CblasNoTrans, 
        *cols, *cols, *rows, 1, 
        mat, *cols, mat, *rows, 
        0, ret, *cols);
}


template<typename T> void 
calculate_matrix_mean(const T *mat, double *ret, const int *rows, const int *cols){
    double sum(0);
    for (int i=0; i<*cols; i++){
        sum = 0;
        for (int j=0; j<*rows; j++){
            sum = sum + *(mat + i*(*rows) + j);
        }
        *(ret + i) = sum / (*rows);
    }
}

template void 
calculate_matrix_mean<int>(const int *mat, double *ret, const int *rows, const int *cols);

template void 
calculate_matrix_mean<float>(const float *mat, double *ret, const int *rows, const int *cols);

template void 
calculate_matrix_mean<double>(const double *mat, double *ret, const int *rows, const int *cols);


template<typename T> void 
calculate_matrix_std(const T *mat, const double *mu, double *ret, const int *rows, const int *cols){
    double squre_sum(0);
    for (int i=0; i<*cols; i++){
        squre_sum = 0;
        for (int j=0; j<*rows; j++){
            squre_sum = squre_sum + pow(*(mat + i*(*rows) + j) - *(mu+i), 2);
        }
        *(ret + i) = sqrt(squre_sum / *rows);
    }
}

template void 
calculate_matrix_std<int>(const int *mat, const double *mu, double *ret, const int *rows, const int *cols);

template void 
calculate_matrix_std<float>(const float *mat, const double *mu, double *ret, const int *rows, const int *cols);

template void 
calculate_matrix_std<double>(const double *mat, const double *mu, double *ret, const int *rows, const int *cols);


void 
calculate_skip_euclid_norm(const double *xxt, const double *mat, const double *arr, double *ret,
                             const int *rows, const int *cols, const int *start, const int *end){
                                // the rows refers to the mat
    
    double inner_prod;
    double MatV[*rows];
    double alpha = 2;
    vector_inner_prod(arr, arr, &inner_prod, cols);
    vector_scalar_add(xxt, &inner_prod, ret,  rows, start, end);
    blas_matrix_vector_prod2(mat, arr, MatV, rows, cols, &alpha, start, end);
    vector_vector_sub(ret, MatV, ret, rows);
}


void 
calculate_skip_euclid_norm(const double *xxt, const double *mat, const float *arr, double *ret,
                             const int *rows, const int *cols, const int *start, const int *end){
                                // the rows refers to the mat
    
    double inner_prod;
    double MatV[*rows];
    double alpha = 2;
    vector_inner_prod(arr, arr, &inner_prod, cols);
    vector_scalar_add(xxt, &inner_prod, ret,  rows, start, end);
    blas_matrix_vector_prod2(mat, arr, MatV, rows, cols, &alpha, start, end);
    vector_vector_sub(ret, MatV, ret, rows);
}



template<typename T> T * 
selectArray(const T *arr, const int start, const int end){
    T *indices = new T[end - start];
    std::copy(arr+start, arr + end, indices);
    return indices;
}

template int * 
selectArray<int>(const int *arr, const int start, const int end);

template float * 
selectArray<float>(const float *arr, const int start, const int end);

template double * 
selectArray<double>(const double *arr, const int start, const int end);

