% REDUCCION DIMENSIONAL CON MAPA AUTO-ORGANIZADO DE KOHONEN
clear, clc, close all;
net=load('redSOM');
net=net.net;
wi=net.IW{1}; %Pesos Asociados a Cada Entrada de la Red Neuronal

sData=load('../../paviaU.mat');
dataCube=(10^-4).*sData.paviaU;
imgSize=size(dataCube);
sData=load('../../paviaU_gt.mat');
dataGround=sData.paviaU_gt;

x=zeros(imgSize(3),imgSize(1)*imgSize(2)); %Matriz Entrada RNA 
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
        if dataGround(i,j)~=0
            tempvect=dataCube(i,j,:);
            x(:,k)=tempvect(:);
            k=k+1;        
        end
    end
end
x=x(:,1:k-1);

dataSOM=wi*x;

save('datosSOM','dataSOM');
disp('PROCESS DONE REDUCTION!!!')
