%  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  MATLAB code to reproduce our work on DNN research for ICASSP 2017.
%  Simply run "main.m", you will get the result for Gaussian IC case in section 4.3.
%  To get results for other sections, slightly modification may apply.
%  We also provide some pre-trained functions to show our results in Table. 1 & Table 2.
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


clc
clear
clear all
K = 10;
num_H = 50000;
disp('####### Generate Training Data #######');
generate(K,num_H);

disp('####### Train Deep Neural Network #######');
trainDNN(K,num_H);

disp('####### Evaluate Training Performance #######');
trainperformance(K,num_H);

disp('####### Evaluate Testing Performance #######');
testperformance(K,num_H)
