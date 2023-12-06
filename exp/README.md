Guide to reproduce all experiments
---------------------

Below we explain how to reproduce each experiment in the SNN paper "Fast and exact fixed-radius neighbor search based on sorting" (X. Chen and S. Güttel, 2023).

Dependencies
---------

* SNN - `pip install snnpy`
* scikit-learn - `pip install -U scikit-learn`: required in all experiments in the paper whenever we compare against other nearest neighbor search algorithms
* GriSPy - `pip install grispy`: for the comparison in Section 6.2
* [classixclustering](https://github.com/nla-group/classix) - `pip install classixclustering`: only required for the experiment of section 6.3 to load the real-world clustering datasets 

Experiments 
-------------

The experiments in each section of the paper are reproduced as detailed below.

### Section 6.1 and 6.2

The directory ``exp1`` contains code to reproduce the results in Section 6.1 and 6.2: First run ``parameter_test1.ipynb`` and ``parameter_test1.ipynb`` separately. Then run ``generate_plot.ipynb`` to generate the figures of the paper. 

### Section 6.3

The directory ``exp2`` contains code to reproduce the results in Section 6.3: First the datasets need to be downloaded, and this can be done using the shell script ``download.sh`` via
```bash
cd exp2
sh download.sh 0 true true
```

You can also download datasets manually from http://corpus-texmex.irisa.fr/ and https://github.com/erikbern/ann-benchmarks/

The experiments are now reproduced by running the notebooks (in no particular order): ``angular_deep1b1.ipynb``, ``angular_deep1b2.ipynb``, ``angular_deep1b3.ipynb``, ``angular_deep1b4.ipynb``, ``angular_deep1b5.ipynb``, ``angular_glove.ipynb``,  ``euclidean_fmn.ipynb``, ``euclidean_gist.ipynb``, ``euclidean_sifts.ipynb``, ``euclidean_siftsmall.ipynb``. 

Then run the ``printinfo.ipynb`` to obtain summary tables from the paper.

### Section 6.4

The directory ``exp3`` contains code to reproduce the clustering results in Section 6.4: run ``real_cluster.ipynb``. Then run ``printinfo.ipynb`` to produce the summaries from the paper.



