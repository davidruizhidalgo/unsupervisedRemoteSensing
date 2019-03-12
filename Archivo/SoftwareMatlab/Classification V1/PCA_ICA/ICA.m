%ICA => INDEPENDENT COMPONENT ANALISYS. 
clear, clc, close all;
numICA=20; %Numero de Componentes Independientes
sData=load('../Indian_pines_corrected.mat');
dataCube=(10^-4).*sData.indian_pines_corrected; %Cargar datos INDIAN PINES
% sData=load('../Salinas_corrected.mat');
% dataCube=(10^-4).*sData.salinas_corrected; %Cargar datos SALINAS VALLEY

imgSize=size(dataCube);
dataMat=ones(imgSize(1)*imgSize(2),imgSize(3));
for i=1:imgSize(3)
    dataVect=dataCube(:,:,i)';
    dataMat(1:end,i)=dataVect(:);
end
% INDEPENDENT COMPONENT ANALISYS
% Performs independent component analysis (ICA) on the input
% data using the max-kurtosis ICA algorithm
% [Zica, W, T, mu] = kICA(dataMat');
% Performs independent component analysis (ICA) on the input
% data using the Fast ICA algorithm
[Zica, W, T, mu] = fastICA(dataMat','kurtosis',0); %'negentropy'

JICi=(1/12)*((1/size(Zica,2))*sum((Zica.^3),2)).^2+...
    (1/48)*(((1/size(Zica,2))*sum((Zica.^4),2))-3).^2;
[B,I] = sort(JICi,'descend');

%ORGANIZAR MATRIZ DE SALIDA 
dataICA=zeros(imgSize(1),imgSize(2),numICA);  %MATRIZ COMPONENTES PRINCIPALES
for i=1:numICA
figure
dataICA(:,:,i)=vec2mat(Zica(I(i),:),imgSize(2));
imagesc(dataICA(:,:,i));
title(strcat('Componente',32,num2str(i))); axis off;
end
% dataICA=spatialwindow(dataICA,3);

save('ICA','dataICA');
disp('PROCESS DONE !!!');




