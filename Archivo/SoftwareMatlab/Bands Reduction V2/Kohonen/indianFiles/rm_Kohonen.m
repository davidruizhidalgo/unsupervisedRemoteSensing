%ENTRENAMIENTO MAPA AUTO-ORGANIZADO DE KOHONEN PARA REDUCCION DIMENSIONAL
clear, clc, close all;
sData=load('../../indian_pines_corrected_0.mat');
dataCube=(10^3).*sData.dataCube;
imgSize=size(dataCube);

x=zeros(imgSize(3),imgSize(1)*imgSize(2)); %Matriz Entrada RNA 
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
        tempvect=dataCube(i,j,:);
        x(:,k)=tempvect(:);
        k=k+1;
    end
end

Nneuronas=[7 7];
net=selforgmap(Nneuronas);
net.trainParam.epochs=1500;
net=train(net,x);

% plotsomplanes(net);
save('redSOM_Reduccion','net');
disp('PROCESS DONE !!!')