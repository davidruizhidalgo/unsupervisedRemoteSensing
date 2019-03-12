%PCA => PRINCIPAL COMPONENT ANALISYS. 
clear, clc, close all;
sData=load('../../paviaU.mat');
dataCube=(10^-4).*sData.paviaU;
imgSize=size(dataCube);

sData=load('../../paviaU_gt.mat');
dataGround=sData.paviaU_gt;
imgSizeGT=size(dataGround);

dataMat=zeros(imgSize(1)*imgSize(2),imgSize(3));
for i=1:imgSize(3)
    dataVect=dataCube(:,:,i)';
    dataMat(:,i)=dataVect(:);
end

%CALCULO COEFICIENTES PRINCIPALES
% [coeff,score,latent,tsquared,explained,mu] =pca(dataMat,'Algorithm','eig');
C=cov(dataMat);                     % matriz de covarianzas
[coeff,latent] = eig(C);            % eigenvectores e eigenvalores 
coeff=(flip(coeff'))';              % organizacion eigenvectores
latent=flip(diag(latent));          % organizacion eigenvalores
explained=100*latent/sum(latent);   % proporcion de varianza total por PC

% %RECUPERAR COMPONENTES PRINCIPALES
% se recuperan los coeficientes con proporcion de varianza mayor 
porPCA=0.99;      %Porcentaje de Varianza Minimo a Recuperar
dataPCA=zeros(imgSize(1),imgSize(2),4);  %COMPONENTES PRINCIPALES
for i=1:4
figure
Zf=dataMat*coeff(:,i);
dataPCA(:,:,i)=vec2mat(Zf,imgSize(2));
imagesc(dataPCA(:,:,i));
title(strcat('Componente',32,num2str(i))); axis off;
end

imgSizePCA=size(dataPCA);

x=zeros(imgSizePCA(3),imgSizePCA(1)*imgSizePCA(2)); %Matriz Entrada RNA 
k=1;
for i=1:imgSizePCA(1)
    for j=1:imgSizePCA(2)
        if (dataGround(i,j)~=0) 
            tempvect=dataPCA(i,j,:);
            x(:,k)=tempvect(:);
            k=k+1;        
        end
    end
end
x=x(:,1:k-1);

save('datosPCA','x');
disp('PROCESS DONE !!!')