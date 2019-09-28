%RED NEURAL GAS GPU
clc;
clear;
close all;
%% GPU DETECT
devGpu=gpuDeviceCount;
for i=1:devGpu
    g = gpuDevice(i);
    fprintf(1,'Device %i has ComputeCapability %s \n', ...
            g.Index,g.ComputeCapability);
end
disp('gpu Device done.......');


%% CARGA DE DATOS
data=load('spiral');
X=gpuArray(data.X);
disp('Load data done.......');

%% DEFINICION DE PARAMETROS DE ENTRENAMIENTO
N = 100;                    %1
MaxIt = 80;                 %2
tmax = 8000;                %3
epsilon_initial = 0.9;      %4
epsilon_final = 0.4;        %5
lambda_initial = 10;        %6
lambda_final = 1;           %7
T_initial = 5;              %8
T_final = 10;               %9
tt = 0;                     %10


% INICIALIZACION DE VARIABLES
 nData = size(X,1);
 nDim = size(X,2);

 X = X(randperm(nData),:); %toma los datos de X de forma aleatoria

 Xmin = min(X);
 Xmax = max(X);
 for i=1:1:N
     a(:,i)=Xmax;
     b(:,i)=Xmin;
 end
 w=((rand(nDim,N).*(a-b))+b)'; %pesos neuronales w(GPU) aleatorios  

C = gpuArray(zeros(N, N)); %enlaces neuronales
t = gpuArray(zeros(N, N)); %tiempo de vida de los enlaces

disp('Init parameters done.......');

%% Main Loop
    for it = 1:MaxIt
        for l = 1:nData
            % Slect Input Vector
            x = X(l,:);

            % Competion and Ranking
            d = pdist2(x,w);
            [~, SortOrder] = sort(d);
            [~, n_ki]=sort(SortOrder);
            
            % Calculate Parameters
            epsilon = epsilon_initial*(epsilon_final/epsilon_initial)^(tt/tmax);
            lambda = lambda_initial*(lambda_final/lambda_initial)^(tt/tmax);
            T = T_initial*(T_final/T_initial)^(tt/tmax);

            % Adaptation
            w=w+((epsilon*exp(-n_ki/lambda))').*(x-w);

            tt = tt + 1;

            % Creating Links
            i = SortOrder(1);
            j = SortOrder(2);
            C(i,j) = 1;
            C(j,i) = 1;
            t(i,j) = 0;
            t(j,i) = 0;

            % Aging
            t(i,:) = t(i,:) + 1;
            t(:,i) = t(:,i) + 1;

            % Remove Old Links
            OldLinks = t(i,:)>T;
            C(i, OldLinks) = 0;
            C(OldLinks, i) = 0;
        end
    end
      
    %% Export Results
    net.w = w;
    net.C = C;
    net.t = t;
    PlotResults(x, w, C)
    disp('Training net done.......');