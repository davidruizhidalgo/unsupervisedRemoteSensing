% DATOS GENERADOS CON MAPA AUTO-ORGANIZADO DE KOHONEN
clear, clc, close all;
% LOAD INPUT DATA 
data=load('trainingDataSOM.mat');
dataIdx=load('idxDataCube.mat');

% x=[data.dataClass{3},data.dataClass{4},...
%     data.dataClass{6},data.dataClass{7},...
%     data.dataClass{9},data.dataClass{11},...
%     data.dataClass{12},data.dataClass{13},data.dataClass{15}]; %9 Class of vegetation
% 
% labels=[dataIdx.idxclass{3}(:,3);dataIdx.idxclass{4}(:,3);dataIdx.idxclass{6}(:,3);...
%     dataIdx.idxclass{7}(:,3);dataIdx.idxclass{9}(:,3);dataIdx.idxclass{11}(:,3);dataIdx.idxclass{12}(:,3);...
%     dataIdx.idxclass{13}(:,3);dataIdx.idxclass{15}(:,3)];

class=4;
x=[data.dataClass{class}];
labels=[dataIdx.idxclass{class}(:,3)];


%LOAD SELF ORGANIZED MAP
net=load('redSOM2');
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
    Mapa(fila,columna)=labels(i);
end
Mapa=Mapa(1:sqrt(size(net.IW{1,1},1)),1:sqrt(size(net.IW{1,1},1)));
% imagesc(Mapa)
hexplot(Mapa);
disp('DONE ACTIVATION MAP');