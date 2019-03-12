%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPML111
% Project Title: Growing Neural Gas Network in MATLAB
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

clc;
clear;
close all;

%% Load Data

data = load('spiral');
X = data.X;

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
