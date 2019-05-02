% DATOS GENERADOS CON MAPA AUTO-ORGANIZADO DE KOHONEN
clear, clc, close all;
sData=load('datosSOM');
dataSOM=sData.dataSOM;
imgSize=size(dataSOM);

x=zeros(imgSize(3),imgSize(1)*imgSize(2)); %Matriz Entrada RNA 

k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
        tempvect=dataSOM(i,j,:);
        x(:,k)=tempvect(:);
        k=k+1;
    end
end


net=load('redRBF_SOM');
net=net.net;
y=net(x);   %Matriz de Salida RNA

%APLICAR DECODIFICACION DE LOS RESULTADOS Y REPRESENTACION GRAFICA DE LA
%IMAGEN.
imgOutSOM=zeros(imgSize(1),imgSize(2));

k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      [valy,indexy]=sort(y(:,k));
      imgOutSOM(i,j)=indexy(end)-1; %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

imgTh=load('../../../../../dataSets/Salinas_gt.mat');
imgTh=imgTh.salinas_gt; figure;
imagesc(imgTh);    %IMAGEN DE CLASIFICACION DESEADA
title('Imagen Deseada'); axis off;
imgSize=size(imgTh);

figure;
imagesc(imgOutSOM);   %IMAGEN DE CLASIFICACIÓN GENERADA SIN FILTRAR
title('Imagen Resultante SOM'); axis off;


%INICIO FILTRO PROCESAMIENTO FINAL
imgOut=zeros(imgSize(1),imgSize(2),17);
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
            case 10
                imgOut(i,j,10)=value;
            case 11 
                imgOut(i,j,11)=value;
            case 12
                imgOut(i,j,12)=value;
            case 13
                imgOut(i,j,13)=value;
            case 14
                imgOut(i,j,14)=value;
            case 15
                imgOut(i,j,15)=value;
            case 16
                imgOut(i,j,16)=value;
            otherwise
        end
             
     end
end
for k=1:16
se = strel('disk',3);
imgOut(:,:,k)=imclose(imgOut(:,:,k),se);
end
imgOutSOM=zeros(imgSize(1),imgSize(2));
for k=1:16
imgOutSOM=imgOutSOM+imgOut(:,:,k);
end
%FIN FILTRO DE PROCESAMIENTO FINAL

figure;
imagesc(imgOutSOM); %IMAGEN DE CLASIFICACIÓN FILTRADA
title('Imagen Resultante SOM FILTRADA'); axis off;


mdl = fitlm(imgTh(:),imgOutSOM(:));
crSOM=mdl.Rsquared.Ordinary;
disp('Correlación Datos SOM');
disp(crSOM);
save('imgOutSOM','imgOutSOM');
disp('DONE CLASSIFICATION');
