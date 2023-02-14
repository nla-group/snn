Manual guide
---------------------

Dependencies
---------

* scikit-learn - pip install -U scikit-learn: required in all experiments in the paper for nearest search algorithms comparison
* [classixclustering]([https://github.com/nla-group/classix/tree/master/exp](https://github.com/nla-group/classix)) - pip install classixclustering: required in the experiment of section 5.3 for the real-world clustering datasets in the paper




Download datasets manually from: http://corpus-texmex.irisa.fr/ and https://github.com/erikbern/ann-benchmarks/

For the experiment in section 5.2, the datasets required to be downloaded additionally, you can also use run shell script ``download.sh`` to download the necessary datasets, use

```bash
cd exp2
sh download.sh 0 true true
```

After running, all required datasets will be downloaded as well as the associated transform will be performed.

For the directory ``exp1``, it is to reproduce the experimental results in section 5.1 of the paper: First run the ``parameter_test1.ipynb`` and ``parameter_test1.ipynb`` separately. Then run ``generate_plot.ipynb`` to generate the figures corresponding to the paper. 

For the directory ``exp2``, it is to reproduce the experimental results in section 5.2 of the paper: After configuring the required datasets (by running ``~$ sh download.sh 0 true true``), then we run experimental programs that perform querying algorithms on the real-world datasets (without specific order): ``angular_deep1b1.ipynb``, ``angular_deep1b2.ipynb``, ``angular_deep1b3.ipynb``, ``angular_deep1b4.ipynb``, ``angular_deep1b5.ipynb``, ``angular_glove.ipynb``,  ``euclidean_fmn.ipynb``, ``euclidean_gist.ipynb``, ``euclidean_sifts.ipynb``, ``euclidean_siftsmall.ipynb``. 


For the directory ``exp3``, it is to reproduce the clustering results in section 5.3 of the paper: 


