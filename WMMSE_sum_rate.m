%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  MATLAB code to reproduce our work on DNN research for ICASSP 2017.
%  To run our code, Neuron Network Toolbox and Deep Learning Toolbox need to be installed first.
%  Code has been tested successfully on MATLAB 2016b prerelease platform.
%
%  References:
%  [1] Haoran Sun, Xiangyi Chen, Qingjiang Shi, Mingyi Hong and Xiao Fu.
%  "LEARNING TO OPTIMIZE: TRAINING DEEP NEURAL NETWORKS FOR WIRELESS RESOURCE MANAGEMENT."
%
%  version 1.0 -- September/2016
%  Written by Haoran Sun (hrsun AT iastate.edu)
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [p_opt sr_opt totIters] = WMMSE_sum_rate(p_int, H, Pmax, var_noise)

K = length(Pmax);
vnew = 0;
b = sqrt(p_int);
f = zeros(K, 1);
w = f;
for i=1:K
    f(i) = H(i,i)*b(i)/((H(i,:).^2)*(b.^2)+var_noise);
    w(i) = 1/(1-f(i)*b(i)*H(i,i));
    vnew = vnew + log2(w(i));
end

VV = [vnew];
iter = 0;

while(1)
    iter = iter+1;
    vold = vnew;
    for i=1:K
        btmp = w(i)*f(i)*H(i,i)/sum(w.*(f.^2).*(H(:,i).^2));
        b(i) = min(btmp, sqrt(Pmax(i))) + max(btmp, 0) - btmp;
    end
    
    vnew = 0;
    for i=1:K
        f(i) = H(i,i)*b(i)/((H(i,:).^2)*(b.^2)+var_noise);
        w(i) = 1/(1-f(i)*b(i)*H(i,i));
        vnew = vnew + log2(w(i));
    end
    
    VV = [VV vnew];
    if vnew-vold <= 1e-3 | iter>100
        break;
    end
end

totIters = iter;
p_opt = b.^2;
sr_opt = vnew;
return

