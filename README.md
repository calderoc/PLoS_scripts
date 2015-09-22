Utility MLE Routines for Bayesian Nonparametric HDP-SLDS Time Series Analysis
=================

Set of Python 2.7 scripts augmenting:


```bibtex
@article{CalderonBloom2015,
    author = {C.P.~Calderon and K.~Bloom},
    title = {Inferring Latent States and Refining Force Estimates via Hierarchical Dirichlet Process Modeling in Single Particle Tracking Experiments},
    year = 2015,
    journal = {PLOS ONE},
    volume = {10},
    issue = {9},
    pages = {e0137633},
    URL = {http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0137633},
    DOI = {10.1371/journal.pone.0137633},
    PMID = {26384324},
    ISSN = {1932-6203}
}

Subdirectory contents:
* src --- folder containing python scripts

```

The scripts and work above are intended to aid in hyper-parameter tuning of the Hierarchical Dirichlet Process Switching Linear Dynamical System (HDP-SLDS) proposed in:
```bibtex
@article{Fox2011,
author = {Fox, Emily and Sudderth, Erik B. and Jordan, Michael I. and Willsky, Alan S.},
file = {:Users/calderoc/Documents/Mendeley Desktop/Fox et al.\_2011\_Bayesian Nonparametric Inference of Switching Dynamic Linear Models.pdf:pdf},
journal = {IEEE Trans. Signal Process.},
number = {4},
pages = {1569--1585},
title = {{Bayesian Nonparametric Inference of Switching Dynamic Linear Models}},
volume = {59},
year = {2011}
}
```

If you utilize these scripts in your work, please cite our PLOS ONE article.  
Additional data sets and scripts will be uploaded to this repo in the future.

Requirements
------------

* [**numpy**](http://www.numpy.org/)
* [**scipy**](http://www.scipy.org/)




Running the Sample Scripts
-------

The simple illustrations provided in (`src/sample1.py`) and (`src/sample2.py`) illustrate how the utility functions can be leveraged.

```bash
cd GITREPO/src
python sample1.py
```



