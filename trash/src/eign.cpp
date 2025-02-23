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

#include "eign.h"
#include <lapacke.h>
#include <iostream>
#include <algorithm>
#include <cmath>

void 
svd_ge_sovler(double *mat, double *u, double *s, double *vt, int *rows, int *cols){
    //computing the SVD
    double *svd_solver = new double[*rows * *cols];
    std::copy(mat, mat+*rows * *cols, svd_solver);
    double *superb = new double [std::min(*rows,*cols)-1];
    int info = LAPACKE_dgesvd(LAPACK_COL_MAJOR, 'A', 'A', *rows, *cols, svd_solver, *rows, s, u, *rows, vt, *cols, superb); 
    if (info !=0){
        std::cerr<<"Error occured in dgesdd. Error code :"<<info<<std::endl;
    }
    delete []svd_solver;
}


void 
svd_dc_sovler(double *mat, double *u, double *s, double *vt, int *rows, int *cols){
    //computing the SVD
    double *svd_solver = new double[*rows * *cols];
    std::copy(mat, mat+*rows * *cols, svd_solver);
    int info = LAPACKE_dgesdd(LAPACK_COL_MAJOR, 'A', *rows, *cols, svd_solver, *rows, s, u, *rows, vt, *cols);
    if (info !=0){
        std::cerr<<"Error occured in dgesdd. Error code :"<<info<<std::endl;
    }

    delete []svd_solver;
}