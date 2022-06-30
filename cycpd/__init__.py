"""
Numpy + Cython Implementation of the Coherent Point Drift
([CPD](https://arxiv.org/abs/0905.2635/)) algorithm by 
Myronenko and Song. This implementation aims to speed up the 
PyCPD implementation of CPD by using Cython. It provides 
three registration methods for point clouds: 

1. Scale and rigid registration
2. Affine registration
3. Gaussian regularized non-rigid registration

Licensed under an MIT License (c) Anthony Gatti.
Distributed here: https://github.com/gattia/cycpd
"""

from .affine_registration import affine_registration
from .deformable_registration import deformable_registration, gaussian_kernel
from .rigid_registration import rigid_registration
