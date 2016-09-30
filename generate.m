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

function generate(K,num_H)
Pm=1;
var_noise = 1;
rng('default');
X=[];
Y=[];
for loop = 1:num_H
    CH = 1/sqrt(2)*complex(randn(K,K), randn(K,K));
    H=abs(CH);
    temp_H = reshape(H,K^2,1);
    Pmax = Pm*ones(K,1);
    p_t = WMMSE_sum_rate(rand(K,1), H, Pmax, var_noise);
    X=[X temp_H];
    Y=[Y p_t];
    if mod(loop,5000)==0
        fprintf('.');
    end
end
save(sprintf('Gussian_%d_%d.mat',num_H, K),'X','Y');
fprintf('Generate Done! \n');
end


% clear
% clc
% clear all
% disp('####### PART A: Generate Training Data (speed up version) #######');
% K = 10;
% num_H = 50000;
% pack=1000;
% step=num_H/pack;
% Pm=1;
% rng('default');
% var_noise = 1;
% for FFF=1:step
%     X=[];
%     Y=[];
%     for loop = 1:pack
%         CH = 1/sqrt(2)*complex(randn(K,K), randn(K,K));
%         H=abs(CH);
%         temp_H = reshape(H,K^2,1);
%         Pmax = Pm*ones(K,1);
%         p_t = WMMSE_sum_rate(rand(K,1), H, Pmax, var_noise);
%         X=[X temp_H];
%         Y=[Y p_t];
%     end
%     save(sprintf('Gussian_%d_%d.mat',step, FFF),'X','Y');
% end
%
% Xcombine=[];
% Ycombine=[];
% for FFF=1:step
%     load(sprintf('Gussian_%d_%d.mat',step, FFF));
%     Xcombine=[Xcombine X];
%     Ycombine=[Ycombine Y];
% end
% X=Xcombine;
% Y=Ycombine;
% save(sprintf('Gussian_%d_%d.mat',num_H, K),'X','Y');
%
% for FFF=1:step
%     delete(sprintf('Gussian_%d_%d.mat',step, FFF));
% end
% fprintf('Generate Done! \n');

