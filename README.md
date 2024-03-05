# MRI random phase correction algorithm

This project aims to correct MRI images for random phase errors during acquisition, as described in the following paper:
http://www.sciencedirect.com/science/article/pii/S0730725X17301492

# Basic use

This code runs on Matlab, and you will have to save the entire repository into the Matlab path list (Home -> set path -> add with subfolders).
Once this is done you may use it by calling:
[I1,ph,A1,bkgd,jj] = iterative_images_correction_v7(A,thresh_bkgd,max_iterations,thresh_phase,known_bkgd)

Inputs:

A: k-space matrix, complex value. May be more than 2D (we use 8D in our lab). Dimensions higher than 2 are vectorised during the process and treated separately. Required.
thresh_bkgd: threshold applied to estimate the background area from the object (default s 0.15). Optional.
max_iterations: maximum number of iteration per optimisation loop (default is 20). Optional.
thresh_phase: target for the phase correction (default is 0.15). Optional.
known_bkgd: 2D logical image showing the estimation of the background (1 for background, 0 for the object) (default is empty). Optional.

Outputs:
I1: Image matrix, real space (complex values).
ph: phase error matrix
A1: corrected k-space
bkgd: background mask obtained from the algorithm
jj: number of iterations used
