% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%SCRIPT DE PRUEBA%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear, clc, close all;
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% AA = [98.1196 89.9465 80.0938 85.7857 89.1242 83.9782 88.1429 87.3991 94.9615 97.2921 98.1408 97.1710 89.0000]
% stdAA = [1.2801 8.1526 6.7822 4.9480 6.2893 9.3607 5.5778 3.9070 4.8158 1.3779 0.6080 1.4812 1.6228]
% 
% meanAA = mean(AA)
% meanSTD = mean(stdAA)
% 
% OA = 92.6154
% stdOA = 5.1887
% Kappa = 90.7400
% stdK = 5.6790


%Esfera de radio unitario en coordenadas esféricas 
Az = linspace(0,2*pi,50); %phi
El = linspace(-pi,pi,50); %theta
r = ones(1,50);

%Construcción de malla
[El,Az] = meshgrid(El,Az);
r = meshgrid(r);

%La conversión se realiza después de construir la malla
[X,Y,Z] = sph2cart(Az,El,r);

mesh(X,Y,Z), axis equal
