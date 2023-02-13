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
#include "eign.h"
#include "snn.h"
#include <omp.h>
#include <cmath>
#include <vector>
#include <numeric>
#include <iostream>
#include <algorithm>


template<typename T> std::vector<int>
argsort(const T *arr, int *size) {
    std::vector<T> array(arr, arr + *size);
    std::vector<int> indices(*size);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&array](int left, int right) -> bool {
                  // sort indices according to corresponding array element
                  return array[left] < array[right];
              });

    return indices;
}

template std::vector<int>
argsort<int>(const int *arr, int *size);

template std::vector<int>
argsort<float>(const float *arr, int *size);

template std::vector<int>
argsort<double>(const double *arr, int *size);


template<typename T>
void reorderArray1D(T *arr, T *copy_arr, std::vector<int> index, const int *size) {
    // T *copy_arr = new T[*size];
    for (int i=0; i<*size; i++){
        *(copy_arr+i) = *(arr + index[i]);
    }
}

template void
reorderArray1D<int>(int *arr, int *copy_arr, std::vector<int> index, const int *size);

template void
reorderArray1D<float>(float *arr, float *copy_arr, std::vector<int> index, const int *size);

template void
reorderArray1D<double>(double *arr, double *copy_arr, std::vector<int> index, const int *size);


template<typename T>
void reorderArray2D(T *mat, T *copy_mat, std::vector<int> index, const int *rows, const int *cols) {
    // T *copy_mat = new T[*rows * *cols];
    for (int i=0; i<*rows; i++){
        for (int j=0; j<*cols; j++){
            copy_mat[i + j* *rows] = mat[index[i] + j* *rows];
        }
    }
}

template void
reorderArray2D<int>(int *arr, int *copy_arr, std::vector<int> index, const int *rows, const int *cols);

template void
reorderArray2D<float>(float *arr, float *copy_arr, std::vector<int> index, const int *rows, const int *cols);

template void
reorderArray2D<double>(double *arr, double *copy_arr, std::vector<int> index, const int *rows, const int *cols);


/*
namespace debug{
    template <typename T>
    void print_vector(T *arr, int size){
        for (int i=0; i<size; i++){
            std::cout << arr[i] << " ";
        }
        std::cout << std::endl;
    }

    template <typename T>
    void printMat(T *mat, int rows, int cols){
        double value;
        for (int i=0; i<rows; i++){
            for (int j=0; j<cols; j++){
                value = *(mat + i + j*rows);;
                std::cout << value << " "; 
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

}
*/





template <class T1, class T2>
SNN_MODEL<T1, T2>::SNN_MODEL(T1 *data, int r, int c){    
    rows = r; cols = c;

    double *u = new double[rows*rows];
    double *s = new double[std::min(rows, cols)];
    
    // std::cout << "work 1" << std::endl;
    mu = new double[cols];
    std = new double[cols];
    
    int size = rows * cols;

    normData = new double[size];
    double *temp_normData = new double[size];
    std::copy(data, data+size, temp_normData);

    sortVals = new double[rows];
    double *temp_sortVals = new double[rows];

    calculate_matrix_mean(data, mu, &rows, &cols);
    calculate_matrix_std(data, mu, std, &rows, &cols);
    
    vt = new double[cols*cols];
    principal_axis = new double[cols];

    int start(0), end(0); // use throughout the program

    // standardize data
    for (int i=0; i<cols; i++){
        start = i*rows;
        end = i*rows + rows;              
        vector_scalar_sub(temp_normData, mu+i, temp_normData, &size, &start, &end);
        // vector_scalar_divide(temp_normData, std+i, temp_normData, &size, &start, &end);
    }


    // double *svd_solver = new double[size];
    // std::copy(temp_normData, temp_normData+size, svd_solver);

    // int arr_inc = 1; // increment of element of vector used for matrix vector product.
    // singular value decomposition, obtain the sort_values

    double sign_flip;

    if (cols > 1) { 
        svd_ge_sovler(temp_normData, u, s, vt, &rows, &cols);
        extract_principal_axis(vt, principal_axis, &cols, 0);

        sign_flip = (principal_axis[0] > 0) ? 1 : ((principal_axis[0] < 0) ? -1 : 0); // flip sign
        start = 0; end = cols;
        vector_scalar_multi(principal_axis, &sign_flip, principal_axis, &cols, &start, &end);
        blas_matrix_vector_prod1(temp_normData, principal_axis, temp_sortVals, &rows, &cols);

    }else if (cols == 1){
        vt[0] = 1.0;
        std::copy(temp_normData, temp_normData+rows, temp_sortVals);

    }else{
        std::cerr << "Error occured in input, please enter correct value for cols." << std::endl;
    }


    sortID.resize(rows);
    sortID = argsort(temp_sortVals, &rows);
    reorderArray1D(temp_sortVals, sortVals, sortID, &rows);
    reorderArray2D(temp_normData, normData, sortID, &rows, &cols);

    xxt = new double[rows];
    blas_norm_matrix(normData, xxt, &rows, &cols);
    // delete []svd_solver;
    delete []u;
    delete []s;
    delete []temp_sortVals;
    delete []temp_normData;
}


template <class T1, class T2> inline void
SNN_MODEL<T1, T2>::extract_principal_axis(const double *vt, double *axis, const int *cols, const int num){ // Column major
    for (int i=0; i<*cols; i++){
        *(axis + i) = *(vt + num + i* *cols);
    }
}


template <class T1, class T2> 
SNN_MODEL<T1, T2>::~SNN_MODEL(){
    delete []vt;
    delete []mu;
    delete []std;
    delete []sortVals;
}


// query
template <class T1, class T2> void 
SNN_MODEL<T1, T2>::radius_single_query(T2 *query, double radius, std::vector<int> *knnID, std::vector<double> *knnDist){
    T1 *query_copy = new T1[cols];
    std::copy(query, query+cols, query_copy);
    vector_vector_sub(query, mu, query_copy, &cols);
    
    double sv_q;
    vector_inner_prod(query_copy, principal_axis, &sv_q, &cols);
    int left = binarySearch(sortVals, sv_q-radius, &rows);
    int right = binarySearch(sortVals, sv_q+radius, &rows);
    
    double *dist_set = new double[rows];
    calculate_skip_euclid_norm(xxt, normData, query_copy, dist_set, &rows, &cols, &left, &right);

    radius = pow(radius, 2);

    (*knnID).clear();
    (*knnDist).clear();

    for (int i=left; i<right; i++){
        if (dist_set[i] <= radius){
            (*knnID).push_back(sortID[i]);
            (*knnDist).push_back(sqrt(dist_set[i]));
        }
    }

    delete []query_copy;
    query_copy = nullptr;
}


void 
extract_sample(double *queries, double *query, const int num, const int *rows, const int *cols){ // columns major order
    for (int i=0; i<*cols; i++){
        *(query + i) = *(queries + *rows * i + num);
    }
}

void
extract_sample(double *queries, float *query, const int num, const int *rows, const int *cols){ // columns major order
    for (int i=0; i<*cols; i++){
        *(query + i) = *(queries + *rows * i + num);
    }
}


void
extract_sample(float *queries, double *query, const int num, const int *rows, const int *cols){ // columns major order
    for (int i=0; i<*cols; i++){
        *(query + i) = *(queries + *rows * i + num);
    }
}


void insert_vector(std::vector<std::vector<int> > *knnID, std::vector<std::vector<double> > *knnDist, 
                        std::vector<int> *knnID_unit, std::vector<double> *knnDist_unit, int i, int qcols){

    for (int j=0; j<qcols; j++){
        (*knnID)[i][j] = (*knnID_unit)[j];
        (*knnDist)[i][j] = (*knnDist_unit)[j];
    }
}


// query in batch
template <class T1, class T2> void 
SNN_MODEL<T1, T2>::radius_batch_query(T1 *queries, double radius, std::vector<std::vector<int> > *knnID, 
                                      std::vector<std::vector<double> > *knnDist, const int qrows){
    double query[cols];
    std::vector<int> knnID_unit;
    std::vector<double> knnDist_unit;


    (*knnID).clear();
    (*knnDist).clear();

    (*knnID).resize(qrows);
    (*knnDist).resize(qrows);

    #pragma omp parallel for
    for (int i=0; i<qrows; i++){
        extract_sample(queries, query, i, &qrows, &cols);
        this->radius_single_query(query, radius, &knnID_unit, &knnDist_unit);

        (*knnID)[i].resize(knnID_unit.size());
        (*knnDist)[i].resize(knnDist_unit.size());
        insert_vector(knnID, knnDist, &knnID_unit, &knnDist_unit, i, knnID_unit.size());
    }
        
}



template <class T1, class T2>  inline int // for 1-dimensional data
SNN_MODEL<T1, T2>::binarySearch(T2 *arr, T2 point, int *size){
    int lo = 0, hi = *size - 1;
    int mid;

    while (hi - lo > 1) {
        mid = trunc((hi + lo) / 2);
        if (arr[mid] < point) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
        
    }
    if (arr[lo] == point) {
        return lo;
    }
    // else if (arr[sortID[hi]] == point) {
    //     return hi;
    // }
    else {
        return hi;
    }
}


// template class SNN_MODEL<int, double>;
template class SNN_MODEL<float, double>;
template class SNN_MODEL<double, double>;