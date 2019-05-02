%Accuracy per class
clear, clc, close all;
load('imgOutSOM.mat');
imgSize=size(imgOutSOM);

figure;
imagesc(imgOutSOM); %IMAGEN DE CLASIFICACIÓN FILTRADA
title('Imagen Resultante FILTRADA'); axis off;

imgTh=load('../../../dataSets/Indian_pines_gt.mat');
imgTh=imgTh.indian_pines_gt;

Accuracy=zeros(1,16);
for clase=1:16
x_class=0;
k=0;
for i=1:imgSize(1)
    for j=1:imgSize(2)
        if imgTh(i,j)==clase
           x_class(end+1)=imgOutSOM(i,j);
           k=k+1;
        end
    end
end

x_class=x_class(:,2:end);

Accuracy(clase)=sum(double(x_class==clase))/numel(x_class);
end

disp(Accuracy');
disp(sum(Accuracy)/16);