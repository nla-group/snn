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

#include <vector>

template <class T1, class T2>
class SNN_MODEL{
    public:
        
        double *mu, *std, *sortVals; 
        // parameters for singular value decomposition
        double *normData;
        
        double *principal_axis;
        
        double *vt, *xxt; // for norm computation;
        int rows, cols;
        
        std::vector<int> sortID;

        // we won't  store the data, since this can be utilized in the query phaze.

        SNN_MODEL(T1 *data, int r, int c);

        ~SNN_MODEL();
        
        void extract_principal_axis(const double *vt, double *axis, const int *cols, const int num);
        void radius_single_query(T1 *query, double radius, std::vector<int> *knnID, std::vector<double> *knnDist);
        static int binarySearch(T2 *arr, T2 point, int *size);
};
