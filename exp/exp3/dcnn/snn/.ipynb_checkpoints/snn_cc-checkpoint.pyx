# distutils: language=c++

cimport cython
from libcpp.vector cimport vector
cimport numpy as cnp
import numpy as np

cnp.import_array()

def dfs1_cc(cnp.ndarray[cnp.uint8_t, ndim=1, mode='c'] is_core,
                 dict neighborhoods,
                 cnp.ndarray[cnp.npy_intp, ndim=1, mode='c'] labels): # sklearn dbscan
    
    cdef cnp.npy_intp i, j, label_num = 0, v
    cdef cnp.ndarray[cnp.npy_intp, ndim=1] neighb
    cdef vector[cnp.npy_intp] stack

    for i in range(labels.shape[0]):
        if labels[i] != -1 or not is_core[i]:
            continue

        j = i
        while True:
            if labels[j] == -1:
                labels[j] = label_num
                if is_core[j]:
                    neighb = neighborhoods[j]
                    for j in range(neighb.shape[0]):
                        v = neighb[j]
                        if labels[v] == -1:
                            stack.push_back(v)

            if stack.size() == 0:
                break
            j = stack.back()
            stack.pop_back()

        label_num += 1
        
        
def dfs2_cc(cnp.ndarray[cnp.uint8_t, ndim=1, mode='c'] is_core,
                 dict neighborhoods,
                 cnp.ndarray[cnp.npy_intp, ndim=1, mode='c'] labels): # dbscan*

    cdef cnp.npy_intp i, j, label_num = 0, v
    cdef cnp.ndarray[cnp.npy_intp, ndim=1] neighb
    cdef vector[cnp.npy_intp] stack

    for i in range(labels.shape[0]):
        if labels[i] != -1 or not is_core[i]:
            continue
        
        j = i
        while True:
            if labels[j] == -1:
                labels[j] = label_num
                if is_core[j]:
                    neighb = neighborhoods[j]
                    for j in range(neighb.shape[0]):
                        v = neighb[j]
                        if labels[v] == -1 and is_core[j]:
                            stack.push_back(v)

            if stack.size() == 0:
                break
            j = stack.back()
            stack.pop_back()

        label_num += 1
        
        
def ndfs1_cc(cnp.ndarray[cnp.uint8_t, ndim=1, mode='c'] is_core,
                 list neighborhoods,
                 cnp.ndarray[cnp.npy_intp, ndim=1, mode='c'] labels): # sklearn dbscan
    
    cdef cnp.npy_intp i, j, label_num = 0, v
    cdef cnp.ndarray[cnp.npy_intp, ndim=1] neighb
    cdef vector[cnp.npy_intp] stack

    for i in range(labels.shape[0]):
        if labels[i] != -1 or not is_core[i]:
            continue

        # Depth-first search starting from i, ending at the non-core points.
        # This is very similar to the classic algorithm for computing connected
        # components, the difference being that we label non-core points as
        # part of a cluster (component), but don't expand their neighborhoods.
        j = i
        while True:
            if labels[j] == -1:
                labels[j] = label_num
                if is_core[j]:
                    neighb = neighborhoods[j]
                    for j in range(neighb.shape[0]):
                        v = neighb[j]
                        if labels[v] == -1:
                            stack.push_back(v)
                            
            if stack.size() == 0:
                break
                
            j = stack.back()
            stack.pop_back()
            
        label_num += 1
        
        
def ndfs2_cc(cnp.ndarray[cnp.uint8_t, ndim=1, mode='c'] is_core,
                 list neighborhoods,
                 cnp.ndarray[cnp.npy_intp, ndim=1, mode='c'] labels): # dbscan*

    cdef cnp.npy_intp i, j, label_num = 0, v
    cdef cnp.ndarray[cnp.npy_intp, ndim=1] neighb
    cdef vector[cnp.npy_intp] stack

    for i in range(labels.shape[0]):
        if labels[i] != -1 or not is_core[i]:
            continue
        
        j = i
        while True:
            if labels[j] == -1:
                labels[j] = label_num
                if is_core[j]:
                    neighb = neighborhoods[j]
                    for j in range(neighb.shape[0]):
                        v = neighb[j]
                        if labels[v] == -1 and is_core[j]:
                            stack.push_back(v)

            if stack.size() == 0:
                break
                
            j = stack.back()
            stack.pop_back()

        label_num += 1
        
        
def dfs3_cc(cnp.ndarray[cnp.uint8_t, ndim=1, mode='c'] is_core,
                 dict neighborhoods,
                 cnp.ndarray[cnp.npy_intp, ndim=1, mode='c'] subsampleID,
                 cnp.ndarray[cnp.npy_intp, ndim=1, mode='c'] labels): # dbscan++
    
    cdef cnp.npy_intp i, j, ind, label_num = 0, v
    cdef cnp.ndarray[cnp.npy_intp, ndim=1] neighb
    cdef cnp.npy_intp size = subsampleID.shape[0]
    cdef vector[cnp.npy_intp] stack
    
    for i in range(size):
        j = subsampleID[i]
        if labels[j] != -1 or not is_core[j]:
            continue

        while True:
            if labels[j] == -1:
                labels[j] = label_num
                
                if is_core[j]:
                    if j in subsampleID:
                        ind = np.where(subsampleID == j)[0][0]
                        neighb = neighborhoods[ind]
                        for j in range(neighb.shape[0]):
                            v = neighb[j]
                            if labels[v] == -1:
                                stack.push_back(v)

            if stack.size() == 0:
                break
                
            j = stack.back()
            stack.pop_back()

        label_num += 1