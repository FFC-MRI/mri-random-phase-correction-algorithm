function [dB,a,b] = linear_correction(dB)

% dB contains a lilnear term a*t+b that needs to be removed
% since linear variations of dB do not affect the image (too
% much). So we find a and b from dB:
% B = epsilon + a*n + b
% mean(B) = m1 = a*n*(n+1)/2 + b*n
%mean(cumsum(B)) = m2 = a*(n+1)*(n+2)/6 + b*(n+1)/2
%
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

sdB = cumsum(dB);
m1 = mean(dB);
m2 = mean(sdB);
n = length(dB);
a = (m1 - 2*m2/(n+1))/((n+1)/2 - (n+2)/3);
b = m1 - a*(n+1)/2;
dB = dB(:) - (a*(1:n)'+b);