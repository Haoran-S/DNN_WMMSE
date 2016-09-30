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

function testperformance(K,num_H)
ENCODE1=2*K;
neurons=2*K;
Pm=1;
var_noise = 1;
threshold=0.5;
sample=10000;
T1=0;
T2=0;
T3=0;
rate_nn_sum=[];
rate_wmmse_sum=[];
rate_max_sum=[];
rate_rand_sum=[];
func=str2func(sprintf('Gussianfit_%d_%d_%d_%d',K,num_H,ENCODE1,neurons));
func(ones(K^2,1));%pre-load DNN function

for loop=1:sample
    CH = 1/sqrt(2)*complex(randn(K,K), randn(K,K));
    H=abs(CH);
    temp_H = reshape(H,K^2,1);
    Pmax = Pm*ones(K,1);
    
    tic
    p_nn=func(temp_H);
    p_nn=(p_nn-min(p_nn))./(max(p_nn)-min(p_nn));
    p_nn=Pm*(p_nn>threshold);
    T1=T1+toc;
    rate_nn = obj_IA_sum_rate(H, p_nn, var_noise);
    rate_nn_sum=[rate_nn_sum rate_nn];
    
    tic
    p_wmmse = WMMSE_sum_rate(rand(K,1), H, Pmax, var_noise);
    T2=T2+toc;
    rate_wmmse = obj_IA_sum_rate(H, p_wmmse, var_noise);
    rate_wmmse_sum=[rate_wmmse_sum rate_wmmse];
    
    rate_max = obj_IA_sum_rate(H, Pmax, var_noise);
    rate_max_sum=[rate_max_sum rate_max];
    
    rate_rand = obj_IA_sum_rate(H, rand(K,1), var_noise);
    rate_rand_sum=[rate_rand_sum rate_rand];
end

fprintf('Testing performance: %.2f%% sum-rate in %.2f%% time\n',sum(rate_nn_sum./rate_wmmse_sum)/sample*100, T1/T2*100);


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


end
