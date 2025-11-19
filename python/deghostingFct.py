"""
Created in July 2021

@author: Marie-Ange STEFANOS

license: LGPLv3
This file is part of the MRI random phase correction algorithm.
Foobar is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Foobar is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with Foobar.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as np
import pdb
import threading

from scipy.optimize import minimize, Bounds, NonlinearConstraint
from scipy import fftpack as fpk
from scipy import ndimage as ndi
from scipy import fft

import torch

import glob

from toolsFctFFC import *
# import mainGetData as gd


def ifft2c(x):
    """
    Orthonormal centered 2D ifft.
    """
    # return np.sqrt(np.size(x))*fpk.fftshift(fpk.ifft2(fpk.ifftshift(x)))
    # return np.fft.fft2(x)
    return fft.fft2(x)


def optim_image(A, bkgd, ph0):
    """
    Corrects for phase encode artifacts due to random phase fluctuations
    between k-space lines.
    
    The phase encode direction should be vertical in the image (along 
    columns).
    
    Parameters
    ----------
    A : 2D-array
        corrected_k
    bkgd : boolean array (shape of the image)
        boolean array
    ph0 : vector
        initial vector the phase
        
    Returns
    -------
    specialPix : 2D-array
        binarized fft : black pixel (with high magnitude) on white background
    """
        
    l, c = np.shape(A)
    
    # Initialising variables
    Aeq = np.zeros((2, c)) # minimize satisfies the condition Aeq*x = beq
    beq = [0, 0]

    # Setting the conditions
        # The average phase correction is set to 0, as this parameter is free
    beq[0] = 0
    Aeq[0, :c] = 1
        # Set the first moment to zero to avoid image shifting
    beq[1] = 0
    Aeq[1, :c] = np.arange(1, c+1)

    # Here is the main optimisation loop, it has to be as fast as possible.
    def minabsim(phi):
        Ac = A*(np.ones((1,1))*np.exp(-1j*phi.T)) # apply the correction in the k-space
        Ic = abs(ifft2c(Ac)) # get the corrected image
        return Ic[bkgd>0].sum() # assess the amount of noise in the background

    def linear_correction(dB):
        """
        dB contains a linear term a*t+b that needs to be removed
        since linear variations of dB can shift the image.
        So we find a and b from dB:
        B = epsilon + a*n + b
        mean(B) = m1 = a*n*(n+1)/2 + b*n
        mean(cumsum(B)) = m2 = a*(n+1)*(n+2)/6 + b*(n+1)/2
        """
        sdB = np.cumsum(dB)
        m1 = np.mean(dB)
        m2 = np.mean(sdB)
        n = len(dB)
        a = (m1 - 2*m2/(n+1))/((n+1)/2 - (n+2)/3)
        b = m1 - a*(n+1)/2
        dB = dB - (a*(np.arange(1, n+1)).T + b) #is .T and np.conj necessary here?
        return dB
    
    bnds = Bounds(-np.pi, np.pi, keep_feasible=True)
    cons = ({'type': 'eq', 'fun': lambda x:  np.dot(Aeq, x)-beq})
    options = {'ftol':1e-6, 'maxfun':1e5, 'maxiter':50000}
    # Solving the system : corrects for random phase on all lines
    res = minimize(minabsim, np.squeeze(ph0.T), method='L-BFGS-B', bounds=bnds, options=options)

    p1 = res.x
    fval = res.fun
    exitflag = res.status

    # Corrects the image using the latest estimation of the phase error
    p1 = linear_correction(p1).T
    A1 = A*np.ones((l,1))*np.exp(-1j*p1)
    I1 = ifft2c(A1)

    return p1, A1, I1, fval, exitflag


class Correction_kspace(torch.nn.Module):
    def __init__(self, ncol, device):
        super(Correction_kspace, self).__init__()
        print(ncol)
        phi = torch.zeros(ncol, device=device)
        self.phi = list()
        # todo: https://discuss.pytorch.org/t/giving-multiple-parameters-in-optimizer/869
        # concatenate parameters? with for loop?
        for i in range(ncol):
            self.phi = self.phi + list(torch.nn.Parameter(torch.zeros(1), requires_grad=True))

    def forward(self, kspace, scalingarray, background):
        phiarray = scalingarray*torch.exp(-1j*self.phi)
        # apply the correction in the k-space
        kspacecor = torch.matmul(kspace, phiarray) # use torch.mul?
        imagecor = torch.fft.ifft2(kspacecor)
        return torch.mul(imagecor, background)  # TODO: use mask


# TODO
def optim_image_torch(    
    kspace: torch.Tensor, bkgd: torch.Tensor, ph0: torch.Tensor
    ) -> torch.Tensor:
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kspace = kspace.to(device)
    bkgd = bkgd.to(device)
    ph0 = ph0.to(device)
    
    l, c = np.shape(kspace)
    
    # Initialising variables
    Aeq = np.zeros((2, c)) # minimize satisfies the condition Aeq*x = beq
    beq = [0, 0]

    # Setting the conditions
        # The average phase correction is set to 0, as this parameter is free
    beq[0] = 0
    Aeq[0, :c] = 1
        # Set the first moment to zero to avoid image shifting
    beq[1] = 0
    Aeq[1, :c] = np.arange(1, c+1)
    
    scalingarray = torch.ones(l,1, device=device)
    target = torch.zeros((l, c), device=device)
    model = Correction_kspace(int(kspace.shape[0]), device)
    
    pdb.set_trace()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10  # Number of cleaning epochs
    for epoch in range(num_epochs):
        prediction = model.forward(kspace, scalingarray, bkgd)
        loss = loss_fn(torch.abs(prediction), target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() 

    def linear_correction_torch(dB):
        """
        dB contains a lilnear term a*t+b that needs to be removed
        since linear variations of dB can shift the image.
        So we find a and b from dB:
        B = epsilon + a*n + b
        mean(B) = m1 = a*n*(n+1)/2 + b*n
        mean(cumsum(B)) = m2 = a*(n+1)*(n+2)/6 + b*(n+1)/2
        """
        sdB = np.cumsum(dB)
        m1 = np.mean(dB)
        m2 = np.mean(sdB)
        n = len(dB)
        a = (m1 - 2*m2/(n+1))/((n+1)/2 - (n+2)/3)
        b = m1 - a*(n+1)/2
        dB = dB - (a*(np.arange(1, n+1)).T + b) #is .T and np.conj necessary here?
        return dB

    pdb.set_trace()
    phi = model.phi.data.item()
    # fval = res.fun
    # exitflag = res.status
    imagecor = model.forward(kspace, scalingarray, bkgd)

    # Corrects the image using the latest estimation of the phase error
    # p1 = linear_correction_torch(p1).T
    # kspacecor = kspace*np.ones((l,1))*np.exp(-1j*p1)
    # imagecor = ifft2c(kspacecor)
    return imagecor


def iterative_images_correction_rs2d(raw_kspace, thresh_bkgd=0.15, max_iterations=20, thresh_phase=0.15, known_bkgd=[], method = 'cpu'):
    # pdb.set_trace() 
    raw_kspace = np.swapaxes(raw_kspace, 0, -1)
    raw_kspace = np.swapaxes(raw_kspace, 1, -2)
    [corrected_image, ph, corrected_k, bkgd, niter] = iterative_images_correction(raw_kspace, thresh_bkgd, max_iterations, thresh_phase, known_bkgd, method)
    corrected_image = np.swapaxes(corrected_image, 0, -1)
    corrected_image = np.swapaxes(corrected_image, 1, -2)
    corrected_k = np.swapaxes(corrected_k, 0, -1)
    corrected_k = np.swapaxes(corrected_k, 1, -2)
    bkgd = np.swapaxes(bkgd, 0, -1)
    bkgd = np.swapaxes(bkgd, 1, -2)
    return corrected_image, ph, corrected_k, bkgd, niter


# TODO
def iterative_images_correction_threaded(raw_image, thresh_bkgd=0.15, max_iterations=20, thresh_phase=0.15, known_bkgd=[]):
    sze = np.shape(raw_image)
    bkgd = []
    raw_image = np.reshape(raw_image, (sze[0], sze[1], -1))
    corrected_image = []; ph = []
    corrected_k = []; niter = []
    # creating the threads:
    threads = []
    # create a function to use the local workspace in the thread and avoid returns
    def process_single_image(imstack,index,thresh_bkgd, max_iterations, thresh_phase, known_bkgd):
        (imstack[:,:,index],
         _,
         _,
         known_bkgd[:,:,index],
         _) = iterative_images_correction(imstack[:,:,index],
                                    thresh_bkgd, max_iterations, thresh_phase, known_bkgd)
        
    for view in range(np.shape(raw_image)[2]):
        t[view] = threading.Thread(target=process_single_image, 
                                   args=(raw_image,
                                   view,
                                   thresh_bkgd,
                                   max_iterations,
                                   thresh_phase, 
                                   known_bkgd,))
        t.start()
    # Wait for all threads to finish
    for t in threads:
        t.join()
    print()
    print("Done!")
    n, m, nbElm = sze[0], sze[1], np.size(corrected_image)
    corrected_image = np.reshape(corrected_image, (n, m, int(nbElm/(n*m))))
    corrected_k = np.reshape(corrected_k, (n, m, int(nbElm/(n*m))))
    bkgd = np.reshape(bkgd, (n, m, int(nbElm/(n*m))))
    return corrected_image, ph, corrected_k, bkgd, niter


def iterative_images_correction(raw_image, thresh_bkgd=0.15,
                                max_iterations=20,
                                thresh_phase=0.15,
                                known_bkgd=[],
                                method='cpu'):
    """
    Parameters
    ----------
    raw_image: 2D-array
        raw image, in the k-space (required)
    thresh_bkgd: float
        normalised threshold level used to estimate the background (optional)
    max_iterations: int
        maximum number of iterations (optional)
    thresh_phase: float
        threshold for the convergence of the phase (optional)
    known_bkgd: boolean array
        estimation of the background provided by the user, if known (optional)
        
    Returns
    -------
    corrected_image: 2D-array
        corrected image, in the normal space
    ph: vector
        estimation of the phase error for each line of k-space
    corrected_k: array
        corrected k-space
    bkgd: 2D-array (same shape of the image)
        background estimated after corrections
    niter: int
        number of iterations used
    """
    ## Dealing with multidimensional image stacks
    if np.ndim(raw_image) > 2:
        # TODO: use the same background for all images in the same slice
        sze = np.shape(raw_image)
        bkgd = []
        raw_image = np.reshape(raw_image, (sze[0], sze[1], -1))
        if len(known_bkgd)>0:
            known_bkgd = np.reshape(known_bkgd, (sze[0], sze[1], -1))
        corrected_image = []; ph = []
        corrected_k = []; niter = []
        for n in range(np.shape(raw_image)[2]):
            print("Processing image " + str(n+1) + " of " + str(np.shape(raw_image)[2]) + "...", end='\r')
            if len(known_bkgd)>0:
                (corrected_image_new, ph_new, corrected_k_new,
                 bkgd_new, niter_new) = iterative_images_correction(raw_image[:,:,n],
                                            thresh_bkgd, max_iterations, thresh_phase,
                                            known_bkgd[:,:,n], method)
            else:
                (corrected_image_new, ph_new, corrected_k_new,
                 bkgd_new, niter_new) = iterative_images_correction(raw_image[:,:,n],
                                            thresh_bkgd, max_iterations, thresh_phase,
                                            [], method)
            if np.shape(niter) != (0,): #adding new results to the existing not empty arrays
                corrected_image = np.insert(corrected_image, np.shape(corrected_image)[0], corrected_image_new, axis=0)
                ph = np.insert(ph, np.shape(ph)[0], ph_new, axis=0)
                corrected_k = np.insert(corrected_k, np.shape(corrected_k)[0], corrected_k_new, axis=0)
                bkgd = np.insert(bkgd, np.shape(bkgd)[0], bkgd_new, axis=0)
                niter = np.insert(niter, np.shape(niter)[0], niter_new, axis=0)
            else: #it is the first iteration : initialization
                corrected_image.append(corrected_image_new)
                ph.append(ph_new)
                corrected_k.append(corrected_k_new)
                bkgd.append(bkgd_new)
                niter.append(niter_new)
        print()
        print("Done!")
        # pdb.set_trace()
        corrected_image = corrected_image.swapaxes(1,2)
        corrected_k = corrected_k.swapaxes(1,2)
        ph = ph.swapaxes(1,2)
        bkgd = bkgd.swapaxes(1,2)
        corrected_image = corrected_image.swapaxes(0,-1)
        corrected_k = corrected_k.swapaxes(0,-1)
        ph = ph.swapaxes(0,-1)
        bkgd = bkgd.swapaxes(0,-1)
        
        # pdb.set_trace()
        corrected_image = np.reshape(corrected_image, sze)
        corrected_k = np.reshape(corrected_k, sze)
        bkgd = np.reshape(bkgd, sze)
        return corrected_image, ph, corrected_k, bkgd, niter
  
    ## Image correction section
    # Reconstructs the image
    initial_image = ifft2c(raw_image)

    # Generates the mask
    sze = np.shape(raw_image)
    # make a background if none was provided
    if np.size(known_bkgd) == 0:
        known_bkgd = np.zeros(sze)
        bkgd_set = False
        threshold_abs = thresh_bkgd*np.amax(np.abs(initial_image))
        bkgd = np.abs(initial_image) < threshold_abs
    else:
        bkgd = known_bkgd
        bkgd_set = True

    # Initialization
    ph0 = np.zeros((1, sze[1])) # initial guess for the phase. Setting it to 0 allows to get the correction increment at each step, which should tend to zero.
    # corrected_k = raw_image # A1 is the corrected k-space. This only allocates enough memory
    # corrected_image = initial_image # same for the corrected image
    ph = np.zeros((1,sze[1])) # list of phase estimates
    bkgd_score = 0
#    print(np.shape(ph0))

    # Iterative correction loop
    for niter in range(max_iterations):
        # pdb.set_trace()
        if method=='cpu':
            dph, raw_image, initial_image, fval, exitflag = optim_image(raw_image, bkgd, ph0) # optimises the phases
        else:
            # convert to torch tensors
            raw_image = torch.from_numpy(raw_image.astype(np.complex64))
            bkgd = torch.from_numpy(bkgd.astype(np.bool))
            ph0 = torch.from_numpy(ph0.astype(np.float64))
            dph, raw_image, _, fval, exitflag = optim_image_torch(raw_image, bkgd, ph0) # optimises the phases
            # revert to numpy arrays
            raw_image = raw_image.numpy()
            bkgd = bkgd.numpy()
            dph = dph.numpy()
            
        ph = ph + dph # accumulates the phase corrections
        if np.std(dph) < thresh_phase: #convergence is reached when the amplitude of the phase correction falls below the threshold
            break
        if bkgd_set:
            # do not change the background estimation
            bkgd = known_bkgd
        else:
            # re-defines the background, remove the first line of the k-space in case of DC artefact
            n, m = np.shape(known_bkgd)
            bkgd_bool = np.ones((n, m))
            cond = np.abs(initial_image)<(thresh_bkgd*np.amax(np.abs(initial_image[1:, 1:])))
            for i in range(n):
                for j in range(m):
                    bkgd_bool[i, j] = (known_bkgd[i, j]>=1) or cond[i, j]
            bkgd = bkgd_bool.astype(int)            
    return initial_image, ph, raw_image, bkgd, niter


# TODO: find best mask for each slice and each
def collapse_background(bkgd):
    b = np.sum(bkgd,1)
    bstack = np.zeros(bkgd.shape)
    for i in range(bkgd.shape[1]):
        bstack[:,i,:,:,:] = b[:,:,:,:]
    return bstack

def signal_estimation(img):
    """
    Parameters
    ----------
    img : 2D-array
        
    Returns
    -------
    Estimation of the signal in the img image computed by the sum of the 
    absolute value of the sum of all the pixels
    """
    return np.sum(abs(img[:, :]))


def paramToStr(Tevo, Bevo):
    """
    Returns a string describing Tevo and Bevo parameters.
    
    Parameters
    ----------
    Tevo : int
        evolution time
    Bevo : int
        evolution field
    
    Returns
    -------
    Str : description of the parameters
    """
    return "Tevo = "+str(Tevo)+", Bevo = "+str(Bevo)


def dispInitialCorrectedImg(initialImg, correctedImg, param):
    """
    Displays the initial and the corrected images with parameters in the title.
    
    Parameters
    ----------
    initialImg : 2D-array
    correctedImg : 2D-array
    param : str
    """
    fig, axs = plt.subplots(ncols=2, figsize=(16,16))
    axs = np.ravel(axs)
    axs[0].imshow(dispImg(initialImg)), axs[0].set_title("Initial image")
    axs[1].imshow(dispImg(correctedImg)), axs[1].set_title("Corrected image")
    plt.suptitle(param)
    plt.show()


def ComputeSortedSig(data):
    """
    Parameters
    ----------
    data : 7D or 9D-array
    
    Returns
    -------
    sigTab : list
        List of the images given in data sorted by decreasing signal amount
    
    """
    if len(np.shape(data)) != 7 and len(np.shape(data)) != 9:
        print("Data shape is neither 7 or 9.")
    else:
        if len(np.shape(data)) == 7:
            n, m, nbSlice, nbSpare, nbAcq, nbT, nbB = np.shape(data)
        elif len(np.shape(data)) == 9:
            n, m, nbAcq, nbSlice, _, nbT, nbB, avgNb, nbChannel = np.shape(data)
        sigTab = []
        for acq in range(nbAcq):
            if acq == 0:
                for Tevo in range(nbT):
                    for Bevo in range(nbB):
                                            
                        fft = gd.chooseData(data, acqNb=acq, Tevo=Tevo, Bevo=Bevo)
                        img = kspaceToNorm(fft)

                        sig_estim = signal_estimation(img)
                        sigTab.append((sig_estim, Tevo, Bevo, fft, img))
    sigTab.sort(key=lambda tup: abs(tup[0]), reverse=True)  # sorts on signal amount
    return sigTab


def getIntoProcess(fft, param):
    """
    Parameters
    ----------
    fft : 2D-array
        image in k-space to apply deghosting algorithm
    param : str
    
    Returns
    -------
    corrected_image : 2D-array
        deghosted image
    bkgd : boolean 2D-array
        final estimation of the background
    """
    print("Begin process "+param)
    res = iterative_images_correction(fft)
    if res != None:
        corrected_image, ph, corrected_k, bkgd, niter = res
    print("End process")
    return corrected_image, bkgd
    

def deghosting1stImg(fftRef, imgRef, acq, Tevo, Bevo, debug=1):
    """
    Applies deghosting algorithm for an image without using any known bkgd.
    
    Parameters
    ----------
    fftRef : 2D-array
        image in k-space
    imgRef : 2D-array
        image in normal space
    acq : int
        acquisition number
    Tevo : int
        evolution time
    Bevo : int
        evolution field
    debug : int
        0 to enter in debugging mode, 1 otherwise and by defalut
        
    
    Returns
    -------
    bkgd : boolean 2D-array
        Final estimation of the background
    """
    param = paramToStr(Tevo, Bevo)
    
    corrected_image, bkgd = getIntoProcess(fftRef, param)
    
    if debug == 0:
        dispInitialCorrectedImg(imgRef, corrected_image, param)
    return bkgd


def deghostingNext(data, known_bkgd, acq, sliceNb, Tevo, Bevo, debug=1):
    """
    Applies deghosting algorithm for the image of the given data with given
    parameters using a known background. Displays the results.
    
    Parameters
    ----------
    data : 7D or 9D-array
    known_bkgd : boolean 2D-array
        estimation of the background got thanks to deghosting1stImg function
    acq : int
        acquisition number
    Tevo : int
        evolution time
    Bevo : int
        evolution field
    debug : int
        0 to enter in debugging mode, 1 otherwise and by default
    """
    fft = gd.chooseData(data, acq=acq, sliceNb=sliceNb, Tevo=Tevo, Bevo=Bevo)
    initImg = kspaceToNorm(fft)
    param = paramToStr(Tevo, Bevo)
    
    corrected_image, bkgd = getIntoProcess(fft, param)

    if debug == 0:
        dispInitialCorrectedImg(initImg, corrected_image, param)
    
    return corrected_image


def deghostingParallel(data, debug=1):
    """
    Applies deghosting to the sum of all image and then to all images of data.
    Parameters
    ----------
    data :
    Returns
    -------
    """
    if len(np.shape(data)) != 7:
        print("Data shape is different than 7.")
    else:
        n, m, acq, nbSlice, echoNb, nbT, nbB = np.shape(data)
        for acqNb in range(acq):
            for sliceNb in range(nbSlice):
                for Tevo in range(nbT):
                    for Bevo in range(nbB):
                                            
                        fft = gd.chooseData(data, acq=acqNb, sliceNb=sliceNb, Tevo=Tevo, Bevo=Bevo)
                        img = kspaceToNorm(fft)
                
                        if Tevo == 0 and Bevo == 0:
                            imgSum = img
                        else:
                            imgSum += img
                
        corrected_image, bkgd = getIntoProcess(fft, "Images sum")
        if debug == 0:
            dispInitialCorrectedImg(img, corrected_image, "Image sum")
        return bkgd
