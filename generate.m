%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  MATLAB code to reproduce our work on DNN research for ICASSP 2017.
%  To run our code, Neuron Network Toolbox and Deep Learning Toolbox need to be installed first.
%  Code has been tested successfully on MATLAB 2016b platform.
%
%  References:
%  [1] Haoran Sun, Xiangyi Chen, Qingjiang Shi, Mingyi Hong and Xiao Fu.
%  "LEARNING TO OPTIMIZE: TRAINING DEEP NEURAL NETWORKS FOR WIRELESS RESOURCE MANAGEMENT."
%
%  version 1.0 -- September/2016
%  Written by Haoran Sun (hrsun AT iastate.edu)
%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function generate(K,num_H)
tic
Pm=1;
var_noise = 1;
rng('default');
X=zeros(K^2,num_H);
Y=zeros(K,num_H);
for loop = 1:num_H
    CH = 1/sqrt(2)*complex(randn(K,K), randn(K,K));
    H=abs(CH);
    temp_H = reshape(H,K^2,1);
    Pmax = Pm*ones(K,1);
    p_t = WMMSE_sum_rate(rand(K,1), H, Pmax, var_noise);
    X(:,loop)=temp_H;
    Y(:,loop)=p_t;
    if mod(loop,5000)==0
        fprintf('.');
    end
end
save(sprintf('Gussian_%d_%d.mat',num_H, K),'X','Y');
fprintf('Generate Done! \n');
toc
end
