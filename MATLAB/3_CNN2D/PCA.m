%PCA => PRINCIPAL COMPONENT ANALISYS. 
clear, clc, close all;
data = load('dataSets/Indian_pines.mat');
data = data.indian_pines_corrected;

%NORMALIZAR
for i=1:size(data,1)
    for j=1:size(data,2)
        mn = mean(data(i,j,:));
        data(i,j,:) = data(i,j,:)-mn;
        st = std(data(i,j,:));
        data(i,j,:) = data(i,j,:)/st;
    end
end

%PCA
imgSize=size(data);
dataMat=ones(imgSize(1)*imgSize(2),imgSize(3));
for i=1:imgSize(3)
    dataVect=data(:,:,i)';
    dataMat(1:end,i)=dataVect(:);
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
componentes = 6;
dataPCA=zeros(imgSize(1),imgSize(2),componentes);  %COMPONENTES PRINCIPALES
for i=1:componentes
figure
Zf=dataMat*coeff(:,i);
dataPCA(:,:,i)=vec2mat(Zf,imgSize(2));
imagesc(dataPCA(:,:,i));
title(strcat('Componente',32,num2str(i))); axis off;
end

disp('PROCESS DONE !!!')