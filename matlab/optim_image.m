function [p1,A1,I1,fval,exitflag] = optim_image(A,bkgd,ph0)
% corrects for phase encode artifacts due to random phase fluctuations
% between k-space lines
% the phase encode direction should be vertical in the image (along 
% columns)
% Author: Lionel Broche, August 2019
% license: LGPLv3
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

[l,c] = size(A);

% optimisation over the entire image in one bloc
options = optimoptions('fmincon','Algorithm','sqp',...
                         'MaxFunEvals',1e5,...
                         'MaxIter',50000,...
                         'Display','off',...
                         'TolX',1e-4,...
                         'TolFun',1e-8);
                     
% initialising variables
Aeq = zeros(2,c);   % fmincon satisfies the condition Aeq*x = beq
beq = zeros(1,2); 

% setting the conditions and solving the system
beq(1) = 0; Aeq(1,1:c) = 1; % The average phase correction is set to 0, as this parameter is free
beq(2) = 0; Aeq(2,1:c) = 1:c; % Set the first moment to zero to avoid image shifting
[p1,fval,exitflag] = fmincon(@(x) minabsim(x),ph0',[],[],Aeq,beq,-10*pi*ones(1,c), 10*pi*ones(1,c),[],options); % corrects for random phase on all lines

% corrects the image using the latest estimation of the phase error
p1 = linear_correction(p1)';
A1 = A.*(ones(l,1)*exp(-1i*p1));
I1 = (ifft2c(A1));

% Here is the main optimisation loop, it has to be as fast as possible.
    function s = minabsim(phi)
        Ac = A.*(ones(l,1)*exp(-1i*phi'));   % apply the correction in the k-space
        Ic = (abs(ifft2c(Ac)));              % get the corrected image
        s = sum(Ic(bkgd(:)));               % assess the amount of noise in the background
    end

end
