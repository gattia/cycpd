# Cython-CPD
[![Build Status](https://travis-ci.com/gattia/cycpd.svg?branch=master)](https://travis-ci.com/gattia/cycpd)

Numpy + Cython Implementation of the Coherent Point Drift Algorithm.

Update to the PyCPD module to include Cython to try and improve performance 

# Introduction / Background

Please see here (https://github.com/siavashk/pycpd) for Pure Numpy implementation <br>
Please see here (https://tinyurl.com/tph4u7e) for original manuscript describing CPD <br>
Please see here (https://sites.google.com/site/myronenko/research/cpd) for original code (Matlab) that you can request <br>
Please see here for Matlab code stored on github by secondary source (https://github.com/markeroon/matlab-computer-vision-routines/tree/master/third_party/CoherentPointDrift)<br>
Please see here (https://github.com/gadomski/cpd) for a C++ implementation<br>


This implementation aims to speed up the PyCPD implementation of CPD. First we added cython functions to compute the expectation step of the EM algorithm. This was able to cut E step times in approximately 1/3. The E step appears to be the major bottle neck of the PyCPD rigid and affine registration methods. Therefore, this method reduces registration of those methods to be ~1/3.

For deformable (non-rigid) registration, the major bottle neck has to do with solving the system of equations. For a 5k point mesh (pair) the whole update_transform step (part of the M-step) takes an average of 11 seconds. Of this 9 seconds went to solving for the transformation parameters. The first approach we took to speed things up is to apply the low-rank method described in the original CPD paper and in the original matlab code. This low-rank method significantly reduced computation time and now the entire M-step takes an average of 0.0008s per iteration - these results are assuming we use the default parameters for lowrank (by default we take the 100 biggest eigenvectors/values of the kernel matrix G, reducing the dimensionality of G using eigenvectors/eigenvalues will contrain the non-rigid deformation, a possible added benefit).

Lowrank was implemented first because it had a huge potential (as reported by Myronenko - https://tinyurl.com/tph4u7e) to reduce computational time of non-rigid registrations (from >3h to 10minutes for 8k points), in fact large point clouds (>30k points) were unable to be registered without lowrank. 

The final addition will be to add the FGT (Fast Gauss Transform) to the current implementation. This has the potential to further increase the performance of the deformable (non-rigid) registrations, but will also increase the speed of rigid and affine. FGT will increase the speed of all three methods as it works on the E-step which applied to all three versions, whereas the M-step is where the algorithms diverge and lowrank works specifically on the M-step of the deformable method. 

# Installation


Must have Cython installed to build package

`pip install cython`

or

`conda install -c anaconda cython`

For any operating system you will have to have a C compiler. If you do not have a C compiler you will get errors when building cycpd. You can often follow these errors to install the appropriate packages. 

Details about installing C-compiler and other steps necessary for installing Cython can be found here: http://docs.cython.org/en/latest/src/quickstart/install.html. Briefly. 

### Linux 
C compiler (gcc) is often present. If it is not, you can install it using: 
`sudo apt-get install build-essential`

### OSX
You will like need to install gcc (if you havent already). This can be done by installing Apple's xcode command line tools:
`xcode-select --install`

### Windows
You will need Visual Studio Community 2019 (free) & Build Tools for Visual Studio 2019. 
These can be downloaded from: https://visualstudio.microsoft.com/downloads/
You may need newer versions of Visual Studio and it's tools, but thats the one that was required as of writing. 


With cython installed:

```
git clone https://github.com/gattia/cycpd
cd cycpd
sudo python setup.py install
```


# Examples

There are three exmples currently implemented. They all show registration of two 3D bones with 5k points together. The Rigid and Affine both apply transformations to a bone and then iuse CPD to return it back to its original shape. The deformable (non-rigid) example tries to warp the bone of one person onto the bone of another person. The deformable example will end at 100 epochs (default), at which time it will not have converged. The deformable likely needs more iterations or may need tuning of the paramters (`alpha`, `beta`, and maybe `w` - although the bones *shouldn't* have outliers so changing `w` is advised against). 

These examples can be run by navigating to the examples folder (after installing) and running: 

```
python knee_rigid.py
python knee_affine.py
python knee_deformable.py
```

## Rigid
![](/gifs/Rigid_knee.gif)


## Affine
![](/gifs/Affine_knee.gif)


## Non-Rigid (Deformable)
![](/gifs/Deformable_knee.gif)


# Tests
Further tests that assert if the proper rotation/translation etc. was recovered using CPD and exactly how long those took can be found in the folder test_time and may be run by navigating there in the commandline and running and of the following:

```
python rigid_test.py
python affine_test.py
python deformable_test.py
```

These tests will include both 2D and 3D results. The only result that does not "*assert*" that the right rotation/translation etc. was recovered is the deformable 3D test, however, the performance of this can be viewed using the `knee_deformable.py` example mentioned above. 



MIT License.

