%CLASIFICACION SVM CON DATOS DWT

clear, clc, close all;
dataDWT=load('datosDWT');
cA_dwt=dataDWT.cA;
cD_dwt=dataDWT.cD;
imgSize=size(cA_dwt);

x=zeros(imgSize(3),imgSize(1)*imgSize(2)); %Matriz Entrada RNA 

k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
        tempvect=cA_dwt(i,j,:);
        x(:,k)=tempvect(:);
        k=k+1;
    end
end

x=x';


svmData=load('svmDWT');
imgOutDWT=zeros(imgSize(1),imgSize(2),17);

%Clase 0
[Ypredict,scores] = predict(svmData.c0,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutDWT(i,j,1)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 1
[Ypredict,scores] = predict(svmData.c1,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutDWT(i,j,2)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 2
[Ypredict,scores] = predict(svmData.c2,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutDWT(i,j,3)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 3
[Ypredict,scores] = predict(svmData.c3,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutDWT(i,j,4)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 4
[Ypredict,scores] = predict(svmData.c4,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutDWT(i,j,5)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 5
[Ypredict,scores] = predict(svmData.c5,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutDWT(i,j,6)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 6
[Ypredict,scores] = predict(svmData.c6,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutDWT(i,j,7)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 7
[Ypredict,scores] = predict(svmData.c7,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutDWT(i,j,8)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 8
[Ypredict,scores] = predict(svmData.c8,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutDWT(i,j,9)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 9
[Ypredict,scores] = predict(svmData.c9,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutDWT(i,j,10)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 10
[Ypredict,scores] = predict(svmData.c10,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutDWT(i,j,11)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 11
[Ypredict,scores] = predict(svmData.c11,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutDWT(i,j,12)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 12
[Ypredict,scores] = predict(svmData.c12,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutDWT(i,j,13)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 13
[Ypredict,scores] = predict(svmData.c13,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutDWT(i,j,14)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 14
[Ypredict,scores] = predict(svmData.c14,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutDWT(i,j,15)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 15
[Ypredict,scores] = predict(svmData.c15,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutDWT(i,j,16)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end

%Clase 16
[Ypredict,scores] = predict(svmData.c16,x);
k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      imgOutDWT(i,j,17)=Ypredict(k); %CLASE DE VEGATACIÓN
      k=k+1;
    end
end


imgOutDWT=imgOutDWT>0;
imgOutDWT_tmp=zeros(size(imgOutDWT,1),size(imgOutDWT,2));
for i=1:size(imgOutDWT,3)
  imgOutDWT_tmp=imgOutDWT_tmp+(i*imgOutDWT(:,:,i)); 
end

imgOutDWT=imgOutDWT_tmp-1;

imgTh=load('indian_pines_gt.mat');
imgTh=imgTh.indian_pines_gt; figure;
imagesc(imgTh);    %IMAGEN DE CLASIFICACION DESEADA
title('Imagen Deseada'); axis off;
figure;
imagesc(imgOutDWT); %IMAGEN RESULTANTE
title('Imagen Resultante Wavelet'); axis off;


save('out_svmDWT','imgOutDWT');
disp('DONE');


