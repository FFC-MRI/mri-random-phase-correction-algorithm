# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 09:59:51 2021

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

from deghostingFct import *


def deghost(data, tech='parallel', debug=1):
    
    deghosted_data = np.zeros(np.shape(data))

    if tech=="series":
        # SORTING THE IMAGES ON THEIR SIGNAL
        sigTab = ComputeSortedSig(data)
    
        Tevo = sigTab[0][1]
        Bevo = sigTab[0][2]
        fftRef = sigTab[0][3] #choosing the fft with the highest signal
        imgRef = sigTab[0][4] #choosing the corresponding image in kspace
        paramStr = paramToStr(Tevo, Bevo) #getting its parameters
        known_bkgd = deghosting1stImg(fftRef, imgRef, 0, Tevo, Bevo, debug=debug)
        
        if len(np.shape(data)) != 7:
            print("Data shape is different than 7.")
        else:
            n, m, acq, nbSlice, echoNb, nbT, nbB = np.shape(data)
                
        for acqNb in range(acq):
            for sliceNb in range(nbSlice):
                for Tevo in range(nbT):
                    for Bevo in range(nbB):
                        corrected_img = deghostingNext(data, known_bkgd, acq=acqNb, Tevo=Tevo, Bevo=Bevo, debug=debug)
                        deghosted_data[:, :, acqNb, sliceNb, 0, Tevo, Bevo] = computeFft(corrected_img)
        return deghosted_data
    
    elif tech == "parallel":
        
        known_bkgd = deghostingParallel(data, debug=debug)
        
        if len(np.shape(data)) != 7 :
            print("Data shape is different than 7.")
        else:
            n, m, acq, nbSlice, echoNb, nbT, nbB = np.shape(data)
        
        for acqNb in range(acq):
            for sliceNb in range(nbSlice):
                for Tevo in range(nbT):
                    for Bevo in range(nbB):
                        corrected_img = deghostingNext(data, known_bkgd, acq=acqNb, sliceNb=sliceNb, Tevo=Tevo, Bevo=Bevo, debug=debug)
                        deghosted_data[:, :, acqNb, sliceNb, 0, Tevo, Bevo] = computeFft(corrected_img)

        return deghosted_data

    else:
        print("Please enter a correct method ('series' or 'parallel').")
        
# START OF DEGHOSTING SCRIPT
        
plt.rcParams['image.cmap'] = "gray"
plt.rcParams['figure.figsize'] = (10, 10)

# SCRIPT OF DEGHOSTING ALGORITHM : MODELE TO BE USED IN THE PIPELINE
path = ".\.\Data samples\Stroke 21"
data = gd.useData(path)
data = data[:, :, :, :, :, :, :, 0, 0]
deghosted_data = deghost(data) #NB : data are shifted !!

# TESTING THE FORMAT OF THE INPUT
# =============================================================================
# acq = 0
# sliceNb = 0
# echoNb = 0
# Tevo = 0
# Bevo = 0
# channel = 0
# plt.figure(), plt.imshow(dispImg(kspaceToNorm(data[:, :, acq, sliceNb, echoNb, Tevo, Bevo]))), plt.show()
# plt.figure(), plt.imshow(dispImg((kspaceToNorm(deghosted_data[:, :, acq, sliceNb, echoNb, Tevo, Bevo])))), plt.show()
# =============================================================================
