%IMAGEN EN COLOR REAL
clear, clc, close all;

% sData=load('Salinas_corrected.mat');
% dataCube=(10^3).*sData.salinas_corrected;
sData=load('indian_pines_corrected_0.mat');
dataCube=sData.dataCube;
imgSize=size(dataCube);

imgColor=zeros(imgSize(1),imgSize(2),3);
imgColor(:,:,1)=dataCube(:,:,8);
imgColor(:,:,2)=dataCube(:,:,14);
imgColor(:,:,3)=dataCube(:,:,26);


v_min=min(min(min(imgColor)));
v_max=max(max(max(imgColor)));

imgColor=(255/(v_max-v_min))*imgColor+((-255*v_min)/(v_max-v_min));


imgOut=uint8(imgColor);
image(imgOut);



% imgTh=load('indian_pines_gt.mat');
% imgTh=imgTh.indian_pines_gt; figure;
% imagesc(imgTh);    %IMAGEN DE CLASIFICACION DESEADA
% title('Imagen Deseada'); axis off;