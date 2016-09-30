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

function trainDNN(K,num_H)

ENCODE1=2*K;
neurons=2*K;
load(sprintf('Gussian_%d_%d.mat',num_H, K));
if num_H>10000
    xTrain=X(:,1:10000);
    tTrain=Y(:,1:10000);
else
    xTrain=X;
    tTrain=Y;
end

var_noise=1;
fprintf('1st layer neurons = %d, 2nd layer neurons = %d, Training samples = %d\n',ENCODE1,neurons, num_H);
tic
rng('default')
autoenc1 = trainAutoencoder(xTrain,ENCODE1, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ShowProgressWindow',true, ...
    'ScaleData', false);
feat1 = encode(autoenc1,xTrain);

net = fitnet(neurons,'trainscg');
net.trainParam.showWindow=1;
net.trainParam.epochs=1000;
net = train(net,feat1,tTrain);

deepnet = stack(autoenc1,net);
deepnet.trainParam.showWindow=1;
deepnet.divideParam.trainRatio=90/100;
deepnet.divideParam.valRatio=10/100;
deepnet.divideParam.testRatio=0/100;
deepnet.trainParam.max_fail=25;
deepnet.trainParam.epochs=1000;
deepnet = train(deepnet,xTrain,tTrain);
xTrain=X;
tTrain=Y;
deepnet = train(deepnet,xTrain,tTrain);
genFunction(deepnet,sprintf('Gussianfit_%d_%d_%d_%d',K,num_H,ENCODE1,neurons));
time=toc;
fprintf('training time = %.2f s\n',time);

end


