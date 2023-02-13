#!/bin/bash

ADIR="/${PWD}/Angular_data/"
EDIR="/${PWD}/Euclidean_data/"


deepDIR="${ADIR}deep/"
gloveDIR="${ADIR}glove/"
fmnDIR="${EDIR}fashion_mnist/"
siftDIR="${EDIR}sift/"
siftmDIR="${EDIR}siftsmall/"
gistDIR="${EDIR}gist/"


if [ -d "$ADIR" ]; then
        echo "Download data in ${ADIR}..."
        if [! -d "$deepDIR" ]; then
                mkdir $deepDIR
        fi
        if [! -d "$gloveDIR" ]; then
                mkdir $gloveDIR
else    
        mkdir $ADIR
        echo "Download data in ${ADIR}..."
        mkdir $deepDIR
        mkdir $gloveDIR
fi



if [ -d "$EDIR" ]; then
        echo "Download data in ${EDIR}..."
        fi
        if [! -d "$fmnDIR" ]; then
                mkdir $fmnDIR
        fi
        if [! -d "$siftDIR" ]; then
                mkdir $siftDIR
        fi
        if [! -d "$siftmDIR" ]; then
                mkdir $siftmDIR
        fi
        if [! -d "$gistDIR" ]; then
                mkdir $gistDIR
        fi
else    
        mkdir $EDIR
        echo "Download data in ${EDIR}..."
        mkdir $fmnDIR
        mkdir $siftDIR
        mkdir $siftmDIR
        mkdir $gistDIR
fi



if [ "$1" = 1 ]
then
        echo "Download Euclidean datasets."
        wget http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5
        wget ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
        wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
        wget http://ann-benchmarks.com/gist-960-euclidean.hdf5
        if [[ ! -z "$2" ]] ; then
                if $2; then
                        tar -xf siftsmall.tar.gz
                        tar -xf sift.tar.gz
                        mv fashion-mnist-784-euclidean.hdf5 Euclidean_data/fashion-mnist-784-euclidean.hdf5
                        mv siftsmall Euclidean_data/siftsmall
                        mv sift Euclidean_data/sift
                        mv gist-960-euclidean.hdf5 Euclidean_data/gist-960-euclidean.hdf5
                fi
        fi
elif [ "$1" = 2 ]
then
        echo "Download Angular datasets."
        wget http://ann-benchmarks.com/glove-100-angular.hdf5
        wget http://ann-benchmarks.com/deep-image-96-angular.hdf5
        if [[ ! -z "$2" ]] ; then
                if $2; then
                        mv glove-100-angular.hdf5 Angular_data/glove-100-angular.hdf5
                        mv deep-image-96-angular.hdf5 Angular_data/deep-image-96-angular.hdf5
                fi
        fi
elif [ "$1" = 0 ]
then
        echo "Download all the experimental datasets."
        wget http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5
        wget ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
        wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
        wget http://ann-benchmarks.com/gist-960-euclidean.hdf5
        wget http://ann-benchmarks.com/glove-100-angular.hdf5
        wget http://ann-benchmarks.com/deep-image-96-angular.hdf5
        if [[ ! -z "$2" ]] ; then
                if $2; then
                        tar -xf siftsmall.tar.gz
                        tar -xf sift.tar.gz
                        mv fashion-mnist-784-euclidean.hdf5 Euclidean_data/fashion-mnist-784-euclidean.hdf5
                        mv siftsmall Euclidean_data/siftsmall
                        mv sift Euclidean_data/sift
                        mv gist-960-euclidean.hdf5 Euclidean_data/gist-960-euclidean.hdf5
                        mv glove-100-angular.hdf5 Angular_data/glove-100-angular.hdf5
                        mv deep-image-96-angular.hdf5 Angular_data/deep-image-96-angular.hdf5
                fi
        fi
fi

if [[ ! -z "$3" ]] ; then
        if $3; then
                python hdf5_tonpy.py
        fi
fi
