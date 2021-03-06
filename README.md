# Probabilistic models of scRNA-seq
Package to reproduce results in my MSc Thesis. It implements wrappers (to standardize model calling) and utility methods (to output results) which facilitate model comparison in a set of benchmarking tasks.

See the pre-print for details.


### Requirements
Install via ``python setup.py install``.

Need to install [mpCMF](https://github.com/pedrofale/mpcmf) and [ZINBayes](https://github.com/pedrofale/zinbayes).
 
R needs to be installed with [ZINB-WaVE](https://bioconductor.org/packages/release/bioc/html/zinbwave.html) and [pCMF](https://gitlab.inria.fr/gdurif/pCMF) packages.

See `data/README.md` for information about data sets.


### Acknowledgements
The code to perform imputation tests and to run ZIFA, ZINB-WaVE and scVI was partly adapted from [scVI-reproducibility](https://github.com/romain-lopez/scVI-reproducibility).
ZIFA was originally implemented [here](https://github.com/epierson9/ZIFA).
