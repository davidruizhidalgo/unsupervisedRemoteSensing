%CLASIFICACION SVM CON DATOS DWT

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

x=x';


svmData=load('svmPCA');
imgOutPCA=zeros(imgSize(1),imgSize(2),17);

%Clase 0
[Ypredict,scores] = predict(svmData.c0,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutPCA(i,j,1)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 1
[Ypredict,scores] = predict(svmData.c1,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutPCA(i,j,2)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 2
[Ypredict,scores] = predict(svmData.c2,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutPCA(i,j,3)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 3
[Ypredict,scores] = predict(svmData.c3,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutPCA(i,j,4)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 4
[Ypredict,scores] = predict(svmData.c4,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutPCA(i,j,5)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 5
[Ypredict,scores] = predict(svmData.c5,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutPCA(i,j,6)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 6
[Ypredict,scores] = predict(svmData.c6,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutPCA(i,j,7)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 7
[Ypredict,scores] = predict(svmData.c7,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutPCA(i,j,8)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 8
[Ypredict,scores] = predict(svmData.c8,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutPCA(i,j,9)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 9
[Ypredict,scores] = predict(svmData.c9,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutPCA(i,j,10)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 10
[Ypredict,scores] = predict(svmData.c10,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutPCA(i,j,11)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 11
[Ypredict,scores] = predict(svmData.c11,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutPCA(i,j,12)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 12
[Ypredict,scores] = predict(svmData.c12,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutPCA(i,j,13)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 13
[Ypredict,scores] = predict(svmData.c13,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutPCA(i,j,14)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 14
[Ypredict,scores] = predict(svmData.c14,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutPCA(i,j,15)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 15
[Ypredict,scores] = predict(svmData.c15,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutPCA(i,j,16)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 16
[Ypredict,scores] = predict(svmData.c16,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutPCA(i,j,17)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

imgOutPCA=imgOutPCA>0;
imgOutPCA_tmp=zeros(size(imgOutPCA,1),size(imgOutPCA,2));
for i=1:size(imgOutPCA,3)
  imgOutPCA_tmp=imgOutPCA_tmp+(i*imgOutPCA(:,:,i));  
end

imgOutPCA=imgOutPCA_tmp-1;


imgTh=load('../indian_pines_gt.mat');
imgTh=imgTh.indian_pines_gt; figure;
imagesc(imgTh);    %IMAGEN DE CLASIFICACION DESEADA
title('Imagen Deseada'); axis off;
figure;
imagesc(imgOutPCA); %IMAGEN RESULTANTE
title('Imagen Resultante PCA'); axis off;


save('out_svmPCA','imgOutPCA');
disp('DONE');


