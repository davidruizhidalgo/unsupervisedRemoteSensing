clear, clc, close all;

sData=load('../../paviaU_gt.mat');
dataGround=sData.paviaU_gt;
imgSizeGT=size(dataGround);

imgOutSOM=load('imgOutSOM.mat');
imgOutSOM=imgOutSOM.imgOutSOM;
imgSize=size(imgOutSOM);

%INICIO FILTRO PROCESAMIENTO FINAL
imgOut=zeros(imgSize(1),imgSize(2),9);
for i=1:imgSize(1)
    for j=1:imgSize(2)
         
        value=imgOutSOM(i,j);
        switch value
            case 1
                imgOut(i,j,1)=value;
            case 2
                imgOut(i,j,2)=value;
            case 3
                imgOut(i,j,3)=value;
            case 4
                imgOut(i,j,4)=value;
            case 5
                imgOut(i,j,5)=value;
            case 6
                imgOut(i,j,6)=value;
            case 7
                imgOut(i,j,7)=value;
            case 8
                imgOut(i,j,8)=value;
            case 9
                imgOut(i,j,9)=value;
            otherwise
        end
             
     end
end
% for k=1:9
% se = strel('disk',1);
% imgOut(:,:,k)=imdilate(imgOut(:,:,k),se);
% ImgMascara=imdilate(imgOut(:,:,k),se);
% imgOut(:,:,k)=imreconstruct(ImgMascara,imgOut(:,:,k));
% end
% imgOutSOM=zeros(imgSize(1),imgSize(2));
% for k=1:9
% imgOutSOM=imgOutSOM+imgOut(:,:,k);
% end
%FIN FILTRO DE PROCESAMIENTO FINAL


se = strel('cube',2);
% imgOut(:,:,k)=imdilate(imgOut(:,:,k),se);
% imgOutSOM=imdilate(imgOutSOM,se);
% imgOutSOM=imdilate(imgOutSOM,se);
% se = strel('cube',2);



figure;
imagesc(dataGround);    %IMAGEN DE CLASIFICACION DESEADA
title('Imagen Deseada'); axis off;

figure;
imagesc(imgOutSOM);   %IMAGEN DE CLASIFICACIÓN GENERADA SIN FILTRAR
title('Imagen Resultante SOM'); axis off;

% INDICE DE DESEMPEÑO
mdl = fitlm(double(dataGround(:)),imgOutSOM(:));
crSOM=mdl.Rsquared.Ordinary;
disp('Correlación Datos SOM');
disp(crSOM);