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

% %%%%%% PART E: Plot Testing Performance %%%%%
disp('####### PART E: Plot Testing Performance #######');
figure(1)
cdfplot(rate_nn_sum)
hold on;
cdfplot(rate_wmmse_sum)
hold on;
cdfplot(rate_max_sum)
hold on;
cdfplot(rate_rand_sum)
hold on;
legend('DNN','WMMSE','Max Power','Random Power');
xlabel('rate');
ylabel('cumulative probability');
savefig(sprintf('DNN_CDF_%d_%d_%d_%d',K,num_H,ENCODE1,neurons));

