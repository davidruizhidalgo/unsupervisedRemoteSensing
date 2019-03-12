%PCA => PRINCIPAL COMPONENT ANALISYS. 
clear, clc, close all;

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

%CALCULO COEFICIENTES PRINCIPALES
% [coeff,score,latent,tsquared,explained,mu] =pca(dataMat,'Algorithm','eig');
C=cov(dataMat);                     % matriz de covarianzas
[coeff,latent] = eig(C);            % eigenvectores e eigenvalores 
coeff=(flip(coeff'))';              % organizacion eigenvectores
latent=flip(diag(latent));          % organizacion eigenvalores
explained=100*latent/sum(latent);   % proporcion de varianza total por PC

% %RECUPERAR COMPONENTES PRINCIPALES
% se recuperan los coeficientes con proporcion de varianza mayor 
numPCA=0;
var=0;      
while var<93   %Porcentaje de Varianza Minimo a Recuperar
    numPCA=numPCA+1;
    var=sum(explained(1:numPCA));
end

dataPCA=zeros(imgSize(1),imgSize(2),numPCA);  %MATRIZ COMPONENTES PRINCIPALES
for i=1:numPCA
figure
Zf=dataMat*coeff(:,i);
dataPCA(:,:,i)=vec2mat(Zf,imgSize(2));
imagesc(dataPCA(:,:,i));
title(strcat('Componente',32,num2str(i))); axis off;
end
% dataPCA=spatialwindow(dataPCA,3);

save('PCA','dataPCA');
disp('PROCESS DONE !!!');












