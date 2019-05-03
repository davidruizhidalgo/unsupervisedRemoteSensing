%Excluir datos del fondo
clear, clc, close all;

sData=load('datosSOM');
x=sData.dataSOM;
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
imgOutSOM=zeros(imgSizeGT(1),imgSizeGT(2));
k=1;
for i=1:imgSizeGT(1)
    for j=1:imgSizeGT(2)
        if dataGround(i,j)~=0
            imgOutSOM(i,j)=y(k);
            k=k+1;
        end
    end
end

figure;
imagesc(dataGround);    %IMAGEN DE CLASIFICACION DESEADA
title('Imagen Deseada'); axis off;

figure;
imagesc(imgOutSOM);   %IMAGEN DE CLASIFICACIÓN GENERADA SIN FILTRAR
title('Imagen Resultante SOM'); axis off;

% INDICE DE DESEMPEÑO
mdl = fitlm(double(dataGround(:)),imgOutSOM(:));
crSOM=mdl.Rsquared.Ordinary;
save('imgOutSOM','imgOutSOM');
disp('Correlación Datos SOM');
disp(crSOM);
save('imgOutSOM','imgOutSOM');
disp('DONE CLASSIFICATION');
