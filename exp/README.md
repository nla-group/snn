Manual guide
---------------------

Dependencies
---------

* scikit-learn - pip install -U scikit-learn: required in all exeriments in the paper for nearest search algorithms comparison
* classix - pip install classixclustering: required in the experiment of the section 5.3 in the paper




Download datasets manually from: http://corpus-texmex.irisa.fr/ and https://github.com/erikbern/ann-benchmarks/

You can also use run shell scirpt ``download.sh`` to download the necessary datasets, use

```bash
sh download.sh 0 true true
```

After running, all required datasets as well as the associated transform will be performed.

For the directory ``exp1`, it is to reproduce section 5.1 of the paper; For the directory ``exp2`, it is to reproduce section 5.2 of the paper; For the directory ``exp3`, it is to reproduce section 5.3 of the paper;


