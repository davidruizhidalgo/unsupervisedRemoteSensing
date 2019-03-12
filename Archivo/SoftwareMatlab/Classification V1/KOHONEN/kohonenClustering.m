% DATOS GENERADOS CON MAPA AUTO-ORGANIZADO DE KOHONEN
clear, clc, close all;
% LOAD INPUT DATA 
data=load('trainingDataSOM.mat');
dataIdx=load('idxDataCube.mat');
subtotalClass=17;
class=1:17;
% x=[data.dataClass{1},data.dataClass{2},data.dataClass{3}...
%     data.dataClass{4},data.dataClass{5},data.dataClass{6},data.dataClass{7}...
%     data.dataClass{8},data.dataClass{9},data.dataClass{10},data.dataClass{11}...
%     data.dataClass{12},data.dataClass{13},data.dataClass{14},data.dataClass{15}...
%     data.dataClass{16},data.dataClass{17}];
% 
% labels=[dataIdx.idxclass{1}(:,3);dataIdx.idxclass{2}(:,3);dataIdx.idxclass{3}(:,3);...
%     dataIdx.idxclass{4}(:,3);dataIdx.idxclass{5}(:,3);dataIdx.idxclass{6}(:,3);dataIdx.idxclass{7}(:,3);...
%     dataIdx.idxclass{8}(:,3);dataIdx.idxclass{9}(:,3);dataIdx.idxclass{10}(:,3);dataIdx.idxclass{11}(:,3);...
%     dataIdx.idxclass{12}(:,3);dataIdx.idxclass{13}(:,3);dataIdx.idxclass{14}(:,3);dataIdx.idxclass{15}(:,3);...
%     dataIdx.idxclass{16}(:,3);dataIdx.idxclass{17}(:,3)];

x=[data.dataClass{1}];

labels=[dataIdx.idxclass{1}(:,3)];


%LOAD SELFO ORGANIZED MAP
net=load('redSOM');
net=net.net;
y=net(x);   %Matriz de Salida RNA
indices=vec2ind(y);
Mapa=zeros(sqrt(size(net.IW{1,1},1)),sqrt(size(net.IW{1,1},1)));


for i=1:size(indices,2)
    fila=floor(indices(i)/sqrt(size(net.IW{1,1},1)))+1;
    columna=round(rem(indices(i),sqrt(size(net.IW{1,1},1))));        
    if columna==0 
        columna=sqrt(size(net.IW{1,1},1));
    end
    Mapa(fila,columna)=labels(i)+1;
end
Mapa=Mapa(1:sqrt(size(net.IW{1,1},1)),1:sqrt(size(net.IW{1,1},1)));
% imagesc(Mapa)
hexplot(Mapa);
disp('DONE ACTIVATION MAP');
