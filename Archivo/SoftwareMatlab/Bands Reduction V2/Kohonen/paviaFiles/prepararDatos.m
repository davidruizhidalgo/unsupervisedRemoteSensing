%ENTRENAMIENTO MAPA AUTO-ORGANIZADO DE KOHONEN PARA REDUCCION DIMENSIONAL
clear, clc, close all;
sData=load('../../paviaU.mat');
dataCube=(10^-4).*sData.paviaU;
imgSize=size(dataCube);

sData=load('../../paviaU_gt.mat');
dataGround=sData.paviaU_gt;
imgSizeGT=size(dataGround);

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


Nneuronas=[3 3];
net=selforgmap(Nneuronas);
net.trainParam.epochs=500;
net=train(net,x);

% plotsomplanes(net);
save('redSOM','net');
disp('PROCESS DONE !!!')