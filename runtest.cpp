#include <iostream>
#include "blasi.h"
#include <chrono>
#include "eign.h"
#include "snn.h"
using namespace std;

// #define DEBUG // The part you would like to hide.


template <typename T> void print_vector(T *, int);
template <typename T> void print_bool_vector(T *, int);
template <typename T1, typename T2> void generate_random_arr(const T1 *, const T1 *, T2 *);
template <typename T> void clear_elements(T *, int);
template <typename T> void printMat(T *, int, int);

int main(){
    int size = 5;

    // test vector_scalar_sub(double *vec, double *scalar, double *ret, int size)
    cout << "test 1" << endl;
    double array[] = {0.5, 0.6, 0.3, 0.2, 0.3};
    double scalar = 0.5;
    
    double *vssRet = new double[size];
    int st=2, ed=3;
    vector_scalar_sub(array, &scalar, vssRet, &size, &st, &ed); 
    cout << "result is: ";
    print_vector(vssRet, 5);
    cout << endl << endl;

    #ifndef DEBUG
    // test vector_inner_prod(double *vec1, double *vec2, double *ret, int size)
    cout << "test 2" << endl;
    double array1[] = {0.5, 0.6, 0.3};
    double array2[] = {1, 0.6, 1};

    double ret;
    vector_inner_prod<double>(array1, array2, &ret, &size); // result is 1.16
    cout << "result is: " << ret << endl << endl;


    // test filter_less(const double *arr, const double filterVal, const bool *ret, const int size)
    cout << "test 3" << endl;
    // array[] = {0.5, 0.6, 0.3};
    // scalar = 0.5;
    bool *vssRet2 = new bool[size];
    filter_less(array, &scalar, vssRet2, &size);
    cout << "result is: ";
    print_bool_vector(vssRet2, size);
    cout << endl << endl;

    // test matrix_vector_prod(const double *mat, const double *arr, double *ret, const int rows, const int cols, const int size)
    cout << "test 4" << endl;
    
    // double mat[] = {0.5, 0.6, 0.3, 0.4};
    // double arr[] = {0.2, 0.2};
    // double *ret2 = new double[2];

    int rows = 10000;
    int cols = 100;
    double *mat = new double[rows * cols];
    double *arr = new double[cols];
    double *ret2 = new double[rows];

    clear_elements(ret2, rows);
    int lda = 1;
    generate_random_arr(&rows, &cols, mat);
    generate_random_arr(&lda, &cols, arr);

     /*
    cout << "mat: ";
    print_vector(mat, rows*cols);
    cout << "b: ";
    print_vector(arr, cols);
    cout << endl;
     */
    
    auto start = std::chrono::high_resolution_clock::now();
    matrix_vector_prod(mat, arr, ret2, &rows, &cols); //result is {0.16, 0.2}
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<std::chrono::microseconds>(stop - start);
    cout << "runtime:" << duration.count() << " microseconds" << endl;
    cout << "result is: ";
    print_vector(ret2, 4);
    cout << endl << endl;


    clear_elements(ret2, rows);


    // ret2[0] = 0; ret2[1] = 0; // clear results

    // test void cblas_dgemv (const CBLAS_LAYOUT Layout, 
    // const CBLAS_TRANSPOSE trans, const MKL_INT m, const MKL_INT n,
    // const double alpha, const double *a, const MKL_INT lda, const double *x, const MKL_INT incx, 
    // const double beta, double *y, const MKL_INT incy);
    // Cblas make mistakes here
    
    cout << "test 5" << endl;
    
    rows = 2;
    cols = 3;
    double matTest[rows * cols];
    double retTest[cols * cols];
    generate_random_arr(&rows, &cols, matTest);
    blas_matrix_inner_prod(matTest, retTest, &rows, &cols);
    
    print_vector(matTest, rows*cols);
    cout << endl;
    print_vector(retTest, cols*cols);
    cout << endl << endl;


    cout << "test 6" << endl;
    rows = 3;
    cols = 3;
    double svd_test[rows * cols] = {3., 1., 4., 4., 2., 2., 3., 3., 1.};;
    double *u = new double[rows*rows];
    double *s = new double[std::min(rows,cols)];
    double *vt = new double[cols*cols];
    svd_ge_sovler(svd_test, u, s, vt, &rows, &cols);
    cout << "u: " << endl;
    print_vector(u, rows*rows);
    cout << endl << endl;
    cout << "s" << endl;
    print_vector(s, std::min(rows,cols));
    cout << endl << endl;
    cout << "v" << endl;
    print_vector(vt, cols*cols);
    cout << endl;



    // 
    cout << "test 7" << endl;
    rows = 3; cols = 3;
    cout << "The matrix is: " << endl;
    double dataMat[rows * cols] = {3., 1., 4., 4., 2., 2., 3., 3., 1.};
    printMat(dataMat, rows, cols);

    double _vector[cols] = {3., 2., 1.};

    print_vector(_vector, rows);

    double *_xxt = new double[rows];
    double *_inner_prod = new double;
    double *_xxtV = new double[rows];
    double *_Xv = new double[rows];
    vector_inner_prod(_vector, _vector, _inner_prod, &cols);

    cout << "print inner product of vector" << endl;
    print_vector(_inner_prod, 1);

    cout << "print np.einsum('ij,ij->i', data, data)" << endl;
    blas_norm_matrix(dataMat, _xxt, &rows, &cols);
    print_vector(_xxt, rows);

    cout << "print xxt + np.inner(v,v).ravel()" << endl;
    int _start(0), _end(rows);
    vector_scalar_add(_xxt, _inner_prod, _xxtV,  &cols, &_start, &_end);
    print_vector(_xxtV, rows);

    cout << "print X.dot(v)" << endl;
    blas_matrix_vector_prod1(dataMat, _vector, _Xv, &rows, &cols); //result is {0.16, 0.2}
    print_vector(_Xv, rows);

    cout << "print X.dot(v) start 0, end 2" << endl;
    int ist = 0;int ied = 2;

    double *store_select_indices = new double[ied - ist];
    double alpha = 1;
    blas_matrix_vector_prod2(dataMat, _vector, store_select_indices, &rows, &cols, &alpha, &ist, &ied); //result is 
    print_vector(store_select_indices, ied - ist);

    cout << "print 2*X.dot(v)" << endl;
    double scala = 2;
    vector_scalar_multi(_Xv, &scala, _Xv,  &cols, &_start, &_end);
    print_vector(_Xv, rows);

    cout << "print euclid(xxt, X, v)" << endl;
    vector_vector_sub(_xxtV, _Xv, _xxtV, &rows);
    print_vector(_xxtV, rows);


    cout << "Build pipeline for these:" << endl;
    cout << "select indices: " << endl;
    double *final_ret = new double[ied - ist];
    calculate_skip_euclid_norm(_xxt, dataMat, _vector, final_ret, &rows, &cols, &ist, &ied);
    print_vector(final_ret, ied - ist);
    // double arr_test[] = {1,2,3,4,5};
    // double *arr_new = new double[2];
    // arr_new = selectArray(arr_test, 2, 4); //3:4
    // print_vector(arr_new, 2);


    SNN_MODEL<double, double> snn_model(dataMat, rows, cols);
    print_vector(snn_model.vt, rows);
    print_vector(snn_model.sortVals, rows);
    for (auto i: snn_model.sortID) {
        cout << i << " ";
    }
    cout << endl;
     // [ 0.79439324,  1.10869009, -1.90308333]



    cout << "\ntest 8" << endl;
    rows = 10;
    cols = 3;
    double snn_test_mat[rows*cols] = {0.5488135 , 0.54488318, 0.43758721, 0.38344152, 0.56804456,
                    0.0871293 , 0.77815675, 0.79915856, 0.11827443, 0.94466892,
                    0.71518937, 0.4236548 , 0.891773, 0.79172504, 0.92559664,
                    0.0202184 , 0.87001215, 0.46147936, 0.63992102, 0.52184832,
                    0.60276338, 0.64589411, 0.96366276, 0.52889492, 0.07103606,
                    0.83261985, 0.97861834, 0.78052918, 0.14335329, 0.41466194};

    printMat(snn_test_mat, rows, cols);
    SNN_MODEL<double, double> snn_model_Test(snn_test_mat, rows, cols);
    
    double query[cols] = {0.5488135 , 0.71518937, 0.60276338};
    vector<int> knnID;
    vector<double> knnDist;
    snn_model_Test.radius_single_query(query, 0.4, &knnID, &knnDist);


    cout << "query 1" << endl;
    cout << "sort ID" << endl;
    for (auto i: snn_model_Test.sortID) {
        cout << i << " ";
    }
    cout << endl;
    cout << "print knnID" << endl;
    for (auto i: knnID) {
        cout << i << " ";
    }
    
    cout << endl;
    cout << "print knnDist" << endl;
    for (auto i: knnDist) {
        cout << i << " ";
    }


    cout << endl;
    cout << "\nquery 2" << endl;
    double query2[cols] = {0.944669, 0.521848, 0.414662};
    snn_model_Test.radius_single_query(query2, 0.4, &knnID, &knnDist);

    cout << "sort ID" << endl;
    for (auto i: snn_model_Test.sortID) {
        cout << i << " ";
    }
    cout << endl;
    cout << "knnID" << endl;
    for (auto i: knnID) {
        cout << i << " ";
    }
    
    cout << endl;
    cout << "knnDist" << endl;
    for (auto i: knnDist) {
        cout << i << " ";
    }

    
    cout << "\n\ntest 10" << endl; 
    double query_batch[2*cols] = {0.5488135, 0.944669, 0.71518937, 0.521848, 0.60276338, 0.414662};
    vector<vector<int> > batch_knnID;
    vector<vector<double> > batch_knnDist;
    snn_model_Test.radius_batch_query(query_batch, 0.4, &batch_knnID, &batch_knnDist, 2);

    for (int j=0; j<2; j++){
        cout << "knnID" << endl;
        for (auto i: batch_knnID[j]){
            cout << i << " ";
        }

        cout << "\nknnDist" << endl;
        for (auto i: batch_knnDist[j]){
            cout << i << " ";
        }

        cout << endl;
    }

    #endif
    return 0;

}



template <typename T> void 
print_vector(T *arr, int size){
    for (int i=0; i<size; i++){
        cout << arr[i] << " ";
    }
    cout << endl;
}


template <typename T> void 
print_bool_vector(T *arr, int size){
    for (int i=0; i<size; i++){
        if (arr[i] == true){
            cout << 1 << " ";
        }else{
            cout << 0 << " ";
        }
        
    }
}


template <typename T1, typename T2> void 
generate_random_arr(const T1 *rows, const T1 *cols, T2 *arr){
    srand((unsigned)time(0)); 
    for(int i=0; i<(*rows) * (*cols); i++){ 
        arr[i] = (rand()%100)+1; 
    }
}


template <typename T> void
clear_elements(T *arr, int size){
    for (int i=0; i<size; i++){
        arr[i] = 0;
    }
}


template <typename T> void 
printMat(T *mat, int rows, int cols){
    double value;
    for (int i=0; i<rows; i++){
        for (int j=0; j<cols; j++){
            value = matrix_major_col_index(mat, &i, &j, &rows);
            cout << value << " "; 
        }
        cout << endl;
    }
    cout << endl;
}
