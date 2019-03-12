%Accuracy per class
clear, clc, close all;
load('imgOutDWT.mat');
imgSize=size(imgOutDWT);

figure;
imagesc(imgOutDWT); %IMAGEN DE CLASIFICACIÓN FILTRADA
title('Imagen Resultante'); axis off;

imgTh=load('../../paviaU_gt.mat');
imgTh=imgTh.paviaU_gt;

Accuracy=zeros(1,9);
for clase=1:9
x_class=0;
k=0;
for i=1:imgSize(1)
    for j=1:imgSize(2)
        if imgTh(i,j)==clase
           x_class(end+1)=imgOutDWT(i,j);
           k=k+1;
        end
    end
end

x_class=x_class(:,2:end);

Accuracy(clase)=sum(double(x_class==clase))/numel(x_class);
end

disp(Accuracy');
disp(sum(Accuracy)/7);

imgOut=zeros(imgSize(1),imgSize(2));
for i=1:imgSize(1)
    for j=1:imgSize(2)
        if imgTh(i,j)~=0 && imgTh(i,j)~=3 && imgTh(i,j)~=7  
            imgOut(i,j)=imgOutDWT(i,j);
        end
    end
end
figure
imagesc(imgOut)
title('Wavelet'); axis off;