% REDUCCION DIMENSIONAL CON MAPA AUTO-ORGANIZADO DE KOHONEN
clear, clc, close all;
net=load('redSOM');
net=net.net;
wi=net.IW{1}; %Pesos capa ocualta de la Red Neuronal

%Cargar Datos Hiperespectrales
%LOAD PCA or ICA DATA
sData=load('../PCA_ICA/PCA.mat');
dataCube=sData.dataPCA;                             %DATA PCA
% sData=load('../PCA_ICA/ICA.mat');
% dataCube=sData.dataICA;                           %DATA ICA
imgSize=size(dataCube);

dataSOM=zeros(imgSize(1),imgSize(2),size(wi,1)); %Matriz Entrada RNA 

for i=1:imgSize(1)
    for j=1:imgSize(2)
        tempvect=dataCube(i,j,:);
        dataSOM(i,j,:)=wi*tempvect(:);
    end
end

imgSize=size(dataSOM);
x=zeros(imgSize(3),imgSize(1)*imgSize(2)); %Matriz Entrada K-MEANS
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
        tempvect=dataSOM(i,j,:);
        x(:,k)=tempvect(:);
        k=k+1;
    end
end

[idx,C] = kmeans(x',17);


k=1;
dataOut=zeros(imgSize(1),imgSize(2)); %Matriz Salida
for i=1:imgSize(1)
    for j=1:imgSize(2)
        dataOut(i,j)=idx(k);
        k=k+1;
    end
end
figure;
imagesc(dataOut);

% Show ground truth image
imgTh=load('../../indian_pines_gt.mat');
imgTh=imgTh.indian_pines_gt; figure;
imagesc(imgTh);    %IMAGEN DE CLASIFICACION DESEADA
title('Ground Truth'); axis off;


% save('datosSOM','dataSOM');
disp('PROCESS DONE REDUCTION!!!')









