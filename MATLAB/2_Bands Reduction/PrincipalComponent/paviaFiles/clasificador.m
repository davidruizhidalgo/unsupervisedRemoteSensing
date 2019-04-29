%Excluir datos del fondo
clear, clc, close all;

sData=load('datosPCA');
x=sData.x;
imgSize=size(x);

sData=load('../../paviaU_gt.mat');
dataGround=sData.paviaU_gt;
imgSizeGT=size(dataGround);

net=load('redRBF');
net=net.net;
y=sim(net,x);   %Matriz de Salida RNA
y=vec2ind(y);   %VECTOR de SAlida
disp('RNA DONE SIMULATION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!');


% IMAGENES DE SALIDA
imgOutPCA=zeros(imgSizeGT(1),imgSizeGT(2));
k=1;
for i=1:imgSizeGT(1)
    for j=1:imgSizeGT(2)
        if dataGround(i,j)~=0 
            imgOutPCA(i,j)=y(k);
            k=k+1;
        end
    end
end

figure;
imagesc(dataGround);    %IMAGEN DE CLASIFICACION DESEADA
title('Ground Truht'); axis off;

figure;
imagesc(imgOutPCA);   %IMAGEN DE CLASIFICACIÓN GENERADA SIN FILTRAR
title('PCA'); axis off;

% INDICE DE DESEMPEÑO
mdl = fitlm(double(dataGround(:)),imgOutPCA(:));
crSOM=mdl.Rsquared.Ordinary;
save('imgOutPCA','imgOutPCA');
disp('Correlación Datos PCA');
disp(crSOM);
disp('DONE CLASSIFICATION');
