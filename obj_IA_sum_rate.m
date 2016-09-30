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

function y = obj_IA_sum_rate(H, p, var_noise)
K = length(p);
y = 0;
for i=1:K
    s = var_noise;
    for j=1:K
        if j~=i
            s = s+H(i,j)^2*p(j);
        end
    end
    y = y+log2(1+H(i,i)^2*p(i)/s);
end
return


