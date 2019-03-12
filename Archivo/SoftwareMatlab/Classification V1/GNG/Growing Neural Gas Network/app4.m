clc;
clear;
close all;

%% Load Data

data = load('iris');
X = data.iris(:,1:4);

%% Parameters

params.N = 100;

params.MaxIt = 50;

params.L = 100;

params.epsilon_b = 0.8;
params.epsilon_n = 0.01;

params.alpha = 0.5;
params.delta = 0.995;

params.T = 10;

net = GrowingNeuralGasNetwork(X, params, true);

disp('Done.....');