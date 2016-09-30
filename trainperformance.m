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

function trainperformance(K,num_H)
ENCODE1=2*K;
neurons=2*K;
Pm=1;
var_noise = 1;
threshold=0.5;
rate_nn_sum=[];
rate_wmmse_sum=[];
load(sprintf('Gussian_%d_%d.mat',num_H, K));
func=str2func(sprintf('Gussianfit_%d_%d_%d_%d',K,num_H,ENCODE1,neurons));

for loop=1:num_H
    
    temp_H = X(:,loop);
    H=reshape(temp_H,K,K);
    Pmax = Pm*ones(K,1);
    
    p_nn=func(temp_H);
    p_nn=(p_nn-min(p_nn))./(max(p_nn)-min(p_nn));
    p_nn=Pm*(p_nn>threshold);
    rate_nn = obj_IA_sum_rate(H, p_nn, var_noise);
    rate_nn_sum=[rate_nn_sum rate_nn];
    
    rate_wmmse = obj_IA_sum_rate(H, Y(:,loop), var_noise);
    rate_wmmse_sum=[rate_wmmse_sum rate_wmmse];
    
end
fprintf('Training performance %.2f%%\n',sum(rate_nn_sum./rate_wmmse_sum)/num_H*100);

end
