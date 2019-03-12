clc;
clear;
close all;

%% Load Data

sData=load('PCA.mat');
dataCube=sData.dataPCA;                             %DATA PCA
% gpuCube=gpuArray(dataCube);
% disp('gpu load done...........');

% sData=load('../PCA_ICA/ICA.mat');
% dataCube=sData.dataICA;                           %DATA ICA
imgSize=size(dataCube);

%LOAD GROUND TRUTH DATA
sData=load('../../indian_pines_gt.mat');
dataGround=sData.indian_pines_gt;                   % INDIAN PINES GROUND TRUTH

x=zeros(imgSize(3),1); %Matriz Entrada RNA 
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
        if (dataGround(i,j)==2) || (dataGround(i,j)==3) || ...
                (dataGround(i,j)==5) || (dataGround(i,j)==6) || ...
                    (dataGround(i,j)==8) || (dataGround(i,j)==10) || ...
                        (dataGround(i,j)==11) || (dataGround(i,j)==12) || ...
                            (dataGround(i,j)==14)
        tempvect=dataCube(i,j,:);
        x(:,k)=tempvect(:);
        k=k+1;
        end
    end
end

x=x';

%% Create and Train Neural Gas Network

params.N = 100;

params.MaxIt = 5000;

params.tmax = 100;

params.epsilon_initial = 0.9;
params.epsilon_final = 0.4;

params.lambda_initial = 10;
params.lambda_final = 1;

params.T_initial = 5;
params.T_final = 10;

net = NeuralGasNetwork(x, params);
PlotResults3d(x, net.w, net.C);
save('ngNet','net');
disp('NG DONE..............');



