# Cython-CPD

Numpy + Cython Implementation of the Coherent Point Drift Algorithm.

Update to the PyCPD module to include Cython to try and improve performance 

# Introduction / Background

Please see here (https://github.com/siavashk/pycpd) for Pure Numpy implementation 
Please see here (https://tinyurl.com/tph4u7e) for original manuscript describing CPD
Please see here (https://sites.google.com/site/myronenko/research/cpd) for original code, mostly for Matlab
Please see here (https://github.com/gadomski/cpd) for a C++ implementation


This implementation aims to speed up the PyCPD implementation of CPD. First we have added cython functions to compute the expectation step of the EM algorithm. This was able to cut E step times in approximately 1/3. The E step appears to be the major bottle neck of the PyCPD rigid and affine registration methods. Therefore, this method reduces registration of those methods by ~1/3.

For deformable (non-rigid) registration, the major bottle neck has to do with solving the system of equations. For a 5k point mesh (pair) the whole update_transform step takes an average of 11 seconds. Of this 2 seconds go to creating A, <0.1s goes to creating B, and 9s goes to solving A,B. Therefore, the first step that could be done with cython easily is speeding up creating of A. The second, and bigger speed up would be to reduce the solve portion. The original implementation in Matlab has different methods of solving for W from A and B that may have a speed up. Also, there is the option to use the FGT (fast gauss transform) as well as the Low-Rank methods to further speed up that implementation. The Low-Rank shows the most promise for speeding up non-rigid registration of high dimensional  meshes, in fact very large meshes are unable to be solved without using low-rank (from the original paper). 

# Installation


Must have Cython installed to build package

`pip install cython`

or

`conda install -c anaconda cython`

With cython installed:

```
git clone https://github.com/gattia/cycpd
cd cycpd
sudo python setup.py install
```

# Testing performance times

You can test the computation times for various 2D and 3D registrations using this package by navigating to test_times: 

`cd test_times`

And then running the respective scripts (they will print out the time to complete the registrations and will print out the average and std of completing various steps of the registration. These print outs are hard coded into the code for now and are being used for testing where speed gains can be made. 

Running these timing based tests is as simple as: 

```
python rigid_test.py
python affine_test.py
python deformable_test.py
```

Note there is no assertion/test of the accuracy of the segmentations produced by the deformable_test.py. But there are for the others. The others will error out of something has been done/changed in the code that produces the wrong answer. 



MIT License.

