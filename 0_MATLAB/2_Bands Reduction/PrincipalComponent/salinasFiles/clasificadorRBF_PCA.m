%CLASIFICADOR REDES RED BASE RADIAL redRBF_PCA
clear, clc, close all;
sData=load('datosPCA');
dataPCA=sData.dataPCA;
imgSize=size(dataPCA);


x=zeros(imgSize(3),imgSize(1)*imgSize(2)); %Matriz Entrada RNA 

k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
        tempvect=dataPCA(i,j,:);
        x(:,k)=tempvect(:);
        k=k+1;
    end
end


net=load('redRBF_PCA');
net=net.net;
y=net(x);   %Matriz de Salida RNA

%APLICAR DECODIFICACION DE LOS RESULTADOS Y REPRESENTACION GRAFICA DE LA
%IMAGEN.
imgOutPCA=zeros(imgSize(1),imgSize(2));

k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      [valy,indexy]=sort(y(:,k));
      imgOutPCA(i,j)=indexy(end)-1; %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

imgTh=load('../../Salinas_gt.mat');
imgTh=imgTh.salinas_gt; figure;
imagesc(imgTh);    %IMAGEN DE CLASIFICACION DESEADA
title('Imagen Deseada'); axis off;
imgSize=size(imgTh);

figure;
imagesc(imgOutPCA);   %IMAGEN DE CLASIFICACIÓN GENERADA SIN FILTRAR
title('Imagen Resultante PCA'); axis off;


%INICIO FILTRO PROCESAMIENTO FINAL
imgOut=zeros(imgSize(1),imgSize(2),17);
for i=1:imgSize(1)
    for j=1:imgSize(2)
         
        value=imgOutPCA(i,j);
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
imgOutPCA=zeros(imgSize(1),imgSize(2));
for k=1:16
imgOutPCA=imgOutPCA+imgOut(:,:,k);
end
%FIN FILTRO DE PROCESAMIENTO FINAL

figure;
imagesc(imgOutPCA); %IMAGEN DE CLASIFICACIÓN FILTRADA
title('Imagen Resultante PCA FILTRADA'); axis off;


mdl = fitlm(imgTh(:),imgOutPCA(:));
crPCA=mdl.Rsquared.Ordinary;
disp('Correlación Datos PCA');
disp(crPCA);
save('imgOutPCA','imgOutPCA');
disp('DONE CLASSIFICATION');
