% REDUCCION DIMENSIONAL CON MAPA AUTO-ORGANIZADO DE KOHONEN
clear, clc, close all;
net=load('redSOM_Reduccion');
net=net.net;
wi=net.IW{1}; %Pesos Asociados a Cada Entrada de la Red Neuronal

%Cargar Datos Hiperespectrales
sData=load('../../../../../dataSets/Indian_pines.mat');
dataCube=(10^0).*sData.indian_pines_corrected;
imgSize=size(dataCube);

dataSOM=zeros(imgSize(1),imgSize(2),size(wi,1)); %Matriz Entrada RNA 

for i=1:imgSize(1)
    for j=1:imgSize(2)
        tempvect=dataCube(i,j,:);
        dataSOM(i,j,:)=wi*tempvect(:);
    end
end

save('datosSOM','dataSOM');
disp('PROCESS DONE REDUCTION!!!')









