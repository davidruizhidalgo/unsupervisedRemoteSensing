%ENTRENAMIENTO MAPA AUTO-ORGANIZADO DE KOHONEN PARA REDUCCION DIMENSIONAL
clear, clc, close all;
%LOAD PCA or ICA DATA
sData=load('../PCA_ICA/PCA.mat');
dataCube=sData.dataPCA;                             %DATA PCA
% sData=load('../PCA_ICA/ICA.mat');
% dataCube=sData.dataICA;                           %DATA ICA
imgSize=size(dataCube);

%LOAD GROUND TRUTH DATA
sData=load('../indian_pines_gt.mat');
dataGround=sData.indian_pines_gt;                   % INDIAN PINES GROUND TRUTH
% sData=load('../Salinas_gt.mat');
% dataGround=sData.salinas_gt;                      % SALINAS VALLEY GROUND TRUTH

x=zeros(imgSize(3),imgSize(1)*imgSize(2)); %Matriz Entrada RNA 
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
        tempvect=dataCube(i,j,:);
        x(:,k)=tempvect(:);
        k=k+1;
    end
end

Nneuronas=[30 30];
% net = competlayer(17);
net=selforgmap(Nneuronas,100,5,'hextop','dist');
net.trainParam.epochs=1500;
net=train(net,x);
yred=net(x);

%·ES NECESARIO APLICAR UN ALGORITMO JERARQUICO AGLOMERATIVO??????????

% indices=net(x);
indices=vec2ind(yred);
k=1;
dataOut=zeros(imgSize(1),imgSize(2)); %Matriz Salida
for i=1:imgSize(1)
    for j=1:imgSize(2)
        dataOut(i,j)=indices(k);
        k=k+1;
    end
end
 imagesc(dataOut);

% plotsomplanes(net);
save('redSOM','net');
disp('PROCESS DONE !!!')