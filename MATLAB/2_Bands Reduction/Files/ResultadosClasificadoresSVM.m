%RESULTADOS CLASIFICADORES SVM
clear, clc, close all;

imgTh=load('indian_pines_gt.mat');
imgTh=imgTh.indian_pines_gt; figure;
imagesc(imgTh);    %IMAGEN DE CLASIFICACION DESEADA
title('Imagen Deseada'); axis off;


imgOutDWT=load('out_svmDWT');
imgOutDWT=imgOutDWT.imgOutDWT;
figure;
imagesc(imgOutDWT); %IMAGEN RESULTANTE
title('Imagen Resultante Wavelet'); axis off;



imgOutPCA=load('out_svmPCA');
imgOutPCA=imgOutPCA.imgOutPCA;
figure;
imagesc(imgOutPCA); %IMAGEN RESULTANTE PCA
title('Imagen Resultante PCA'); axis off;


crPCA=corr2(imgTh,imgOutPCA);
crDWT=corr2(imgTh,imgOutDWT);
disp('Correlación Datos PCA');
disp(crPCA);
disp('Correlación Datos Wavelet');
disp(crDWT);
