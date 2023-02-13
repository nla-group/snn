import numpy as np
import h5py
import warnings

if __name__ == '__main__':

    # Download the dataset with the following command.
    # If the dataset is already available in the current working dir, you can skip this:
    # !wget http://ann-benchmarks.com/glove-100-angular.hdf5
    try:
        f = h5py.File('Angular_data/glove-100-angular.hdf5', 'r')

        # Extract the split and ground-truth
        X_train = f['train']
        X_test = f['test']
        neigh_true = f['neighbors']
        dist = f['distances']


        # How many object have we got?
        for k in f.keys():
            print(f'{k}: shape = {f[k].shape}')


        np.save('Angular_data/glove/train.npy', X_train[:])
        np.save('Angular_dataa/glove/queries.npy', X_test[:])
        np.save('Angular_data/glove/neighbors.npy', neigh_true[:])
    
    except:
        warnings.warn("GLOVE data not found!")
        
        
    try:
        f = h5py.File('Euclidean_data/deep-image-96-angular.hdf5', 'r')

        # Extract the split and ground-truth
        X_train = f['train']
        X_test = f['test']
        neigh_true = f['neighbors']
        dist = f['distances']

        for k in f.keys():
            print(f'{k}: shape = {f[k].shape}')

        np.save('Angular_data/deep/train.npy', X_train[:])
        np.save('Angular_data/deep/queries.npy', X_test[:])
        np.save('Angular_data/deep/neighbors.npy', neigh_true[:])
    except:
        warnings.warn("DEEP1B data not found!")
    
    
    try:
        f = h5py.File('Euclidean_data/fashion-mnist-784-euclidean.hdf5', 'r')

        # Extract the split and ground-truth
        X_train = f['train']
        X_test = f['test']
        neigh_true = f['neighbors']
        dist = f['distances']


        # How many object have we got?
        for k in f.keys():
            print(f'{k}: shape = {f[k].shape}')

        np.save('Euclidean_data/fashion_mnist/train.npy', X_train[:])
        np.save('Euclidean_data/fashion_mnist/queries.npy', X_test[:])
        np.save('Euclidean_data/fashion_mnist/neighbors.npy', neigh_true[:])
    
    except:
        warnings.warn("Fashion-MNIST data not found!")
    
    try:
        f = h5py.File('Euclidean_data/gist-960-euclidean.hdf5', 'r')

        X_train = f['train']
        X_test = f['test']
        neigh_true = f['neighbors']
        dist = f['distances']
        for k in f.keys():
            print(f'{k}: shape = {f[k].shape}')

        np.save('Euclidean_data/gist/train.npy', X_train[:])
        np.save('Euclidean_data/gist/queries.npy', X_test[:])
        np.save('Euclidean_data/gist/neighbors.npy', neigh_true[:])
        
        
    except:
        warnings.warn("GIST data not found!")
