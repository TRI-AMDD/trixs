# TRIXS
## Toyota Research Institute X-ray Spectroscopy 

A suite of tools to enable analysis, comparison, classifcation, and active learning on XAS data as a part of Toyota Research Institute's 
collaboration with JCAP and LBNL.


## How to install

The packages required for this repo can be found in requirements.txt.


## Data to use

The data used to generate the figures found in [1] are publically available at 
TRI's 
https://data.matr.io/ .



# Citation
If you use any part of this code for your own purposes, please cite:

[1] S.B. Torrisi *et al*, 
Random Forest Machine Learning Models for Interpretable X-Ray Absorption
Near-Edge Structure Spectrum-Property Relationships, **NPJ Computational 
Materials**, 2020

If you use the Spectrum core class, which is built off of the
 pymatgen software package, don't forget to cite:
 
[2] Shyue Ping Ong, *et al*. *Python Materials Genomics (pymatgen) : A Robust,
    Open-Source Python Library for Materials Analysis.* **Computational
    Materials Science**, 2013, **68**, 314-319. `doi:10.1016/j.commatsci
    .2012.10.028
    <http://dx.doi.org/10.1016/j.commatsci.2012.10.028>`_ 
 
If you use any of the scripts which involve or use Atomate, please cite:

[3]  Mathew, K. *et al.* Atomate: A high-level interface to generate, 
execute, and analyze computational materials science workflows. **Comput. 
Mater. Sci. 139**,
140-152 (2017).


### Acknowledgements
This repo was principally written by Steven B. Torrisi at Toyota Research 
Institute during Summer 2019. While their names may not show up in the 
contributors tab, the feedback of Matthew Carbone, Santosh Suram, and Joseph
 Montoyta were useful in shaping the design of the code in this repository.

*Copyright 2018-2020 Toyota Resarch Institute. All rights reserved.
Use of this source code is governed by an Apache 2.0
license that can be found in the LICENSE file.*