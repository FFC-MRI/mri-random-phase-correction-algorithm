function [corrected_image,ph,corrected_k,bkgd,niter] = iterative_images_correction(raw_image,thresh_bkgd,max_iterations,thresh_phase,known_bkgd)

% [corrected_image,ph,corrected_k,bkgd,niter] =
% iterative_images_correction_v7(raw_image,thresh_bkgd,max_iterations,thresh_phase,known_bkgd) 
% 
% Iterative MRI image correction for random phase fluctuations in the
% phase-encode direction. This code estimates the phase error in each line
% of k-space using an optimisation technique described in Broche 2017
% (http://www.sciencedirect.com/science/article/pii/S0730725X17301492,
% DOI:10.1016/j.mri.2017.07.023)
% Inputs:
%   raw_image: raw image, in the k-space (required)
%   thresh_bkgd: normalised threshold level used to estimate the background
%               (optional)
%   max_iterations: maximum number of iterations (optional)
%   thresh_phase: threshold for the convergence of the phase(optional)
%   known_bkgd: estimation of the background provided by the user, if known
%               (optional)
% Outputs:
%   corrected_image: corrected images, in the normal space
%   ph: estimation of the phase error for each line of k-space
%   corrected_k: corrected k-space
%   bkgd: background estimated after corrections
%   niter: number of iterations used
%
% NOTES: This program has been developed for a specific used (Fast Field
% Cycling MRI) and is not optimised for large images (in fact, it can
% probably be optimised in many ways). Any improvements are welcome and may
% be uploded on the Gitlab page of the project
% (https://gitlab.com/lionel-mri/mri-random-phase-correction-algorithm)
% 
% Author: Dr Lionel M. Broche, August 2019
% License: LGPLv3
%     This file is part of the MRI random phase correction algorithm.
%
%     Foobar is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     Foobar is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with Foobar.  If not, see <https://www.gnu.org/licenses/>.
%

%% dealing with default input values
if nargin<2
    thresh_bkgd = 0.15;
end
if nargin<3
    max_iterations = 20;
end
if nargin<4
    thresh_phase=0.15;
end
if nargin<5
    known_bkgd=[];
end

%% dealing with multidimensional image stacks
if ndims(raw_image)>2
    sze = size(raw_image);
    bkgd = [];
    raw_image = reshape(raw_image,size(raw_image,1),size(raw_image,2),[]);
    for n = 1:size(raw_image,3)
        [corrected_image(:,:,n),ph(:,n),corrected_k(:,:,n),bkgd(:,:,n),niter(n)] = iterative_images_correction(raw_image(:,:,n),thresh_bkgd,max_iterations,thresh_phase,known_bkgd);
    end
    corrected_image = reshape(corrected_image,sze);
    corrected_k = reshape(corrected_k,sze);
    bkgd = reshape(bkgd,sze);
    return
end

%% image correction section
% reconstructs the image:
initial_image = ifft2c(raw_image);

% generates the mask
if nargin<=4
    known_bkgd = zeros(size(raw_image))==1;
    bkgd = abs(initial_image)<(thresh_bkgd*max(abs(initial_image(:))));
elseif isempty(known_bkgd)
    known_bkgd = zeros(size(raw_image))==1;
    bkgd = abs(initial_image)<(thresh_bkgd*max(abs(initial_image(:))));
else
    bkgd = known_bkgd==1;
end

% initialisation:
ph0 = zeros(1,size(raw_image,2));       % initial guess for the phase. Setting it to 0 allows to get the correction increment at each step, which should tend to zero.
corrected_k = raw_image;                         % A1 is the corrected k-space. This only allocates enough memory
corrected_image = initial_image;                         % same for the corrected image
ph = zeros(1,size(raw_image,2));        % list of phase estimates
bkgd_score = 0;

% iterative correction loop:
for niter = 1:max_iterations
    [dph,corrected_k,corrected_image,fval,exitflag] = optim_image(corrected_k,bkgd,ph0);     % optimises the phases
    ph = ph + dph;              % accumulates the phase corrections
    
    if std(dph)<thresh_phase    % convergence is reached when the amplitude of the phase correction falls below the threshold
        break
    end
    bkgd = (known_bkgd==1) | (abs(corrected_image)<(thresh_bkgd*max(max(abs(corrected_image(2:end,2:end))))));        % re-defines the background, remove the first line of the k-space in case of DC artefact

end


