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

x=zeros(imgSize(3),1); %Matriz Entrada RNA 
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
        if (dataGround(i,j)==2) || (dataGround(i,j)==3) || ...
                (dataGround(i,j)==5) || (dataGround(i,j)==6) || ...
                    (dataGround(i,j)==8) || (dataGround(i,j)==10) || ...
                        (dataGround(i,j)==11) || (dataGround(i,j)==12) || ...
                            (dataGround(i,j)==14)
        tempvect=dataCube(i,j,:);
        x(:,k)=tempvect(:);
        k=k+1;
        end
    end
end

Nneuronas=[30 30];
% net = competlayer(9);
net=selforgmap(Nneuronas,50,5,'hextop','dist');
net.trainParam.epochs=3000;
net=train(net,x);

% plotsomplanes(net);
save('redSOM2','net');
disp('PROCESS DONE !!!')