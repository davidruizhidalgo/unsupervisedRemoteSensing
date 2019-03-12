%Accuracy per class
clear, clc, close all;

clear, clc, close all;
dataDWT=load('datosDWT');
cA_dwt=dataDWT.cA;
cD_dwt=dataDWT.cD;
imgSize=size(cA_dwt);


imgTh=load('../../indian_pines_gt.mat');
imgTh=imgTh.indian_pines_gt; %figure;
% imagesc(imgTh);    %IMAGEN DE CLASIFICACION DESEADA
% title('Imagen Deseada'); axis off;

net=load('redRBF_DWT');
net=net.net;

Accuracy=zeros(1,16);
for clase=1:16
% IMAGENES DE SALIDA por Clase
imgOutDWT=zeros(imgSize(1),imgSize(2));
x_class=zeros(imgSize(3),1);
k=0;
for i=1:imgSize(1)
    for j=1:imgSize(2)
        if imgTh(i,j)==clase
            imgOutDWT(i,j)=imgTh(i,j);
           temp=cA_dwt(i,j,:);
           x_class(:,end+1)=temp(:);
           k=k+1;
        end
    end
end

x_class=x_class(:,2:end);

% figure;
% imagesc(imgOutDWT);
% title('Imagen Clase'); axis off;

y=net(x_class);   %Matriz de Salida RNA
y=vec2ind(y)-1;

Accuracy(clase)=sum(double(y==clase))/numel(y);
end
Accuracy=Accuracy/0.912;
disp(Accuracy');
disp(sum(Accuracy)/16);