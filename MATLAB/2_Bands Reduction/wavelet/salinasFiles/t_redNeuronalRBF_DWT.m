%ENTRENAMIENTO RED BASE RADIAL CON DATOS WAVELET
clear, clc, close all;


trainData=load('datosEntrenamientoDWT.mat');

%MATRIZ DATOS DE ENTRADA
x=[trainData.dataClass0,trainData.dataClass1,trainData.dataClass2,trainData.dataClass3...
    trainData.dataClass4,trainData.dataClass5,trainData.dataClass6,trainData.dataClass7...
    trainData.dataClass8,trainData.dataClass9,trainData.dataClass10,trainData.dataClass11...
    trainData.dataClass12,trainData.dataClass13,trainData.dataClass14,trainData.dataClass15,trainData.dataClass16];

%MATRIZ DATOS DE SALIDA
y=zeros(17,size(x,2)); % 17 Clases de Vegetación
sizeIni=1; sizeEnd=size(trainData.dataClass0,2);
y(1,sizeIni:sizeEnd)=1; %Clase Cero
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass1,2)-1;
y(2,sizeIni:sizeEnd)=1; %Clase Uno
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass2,2)-1;
y(3,sizeIni:sizeEnd)=1; %Clase Dos
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass3,2)-1;
y(4,sizeIni:sizeEnd)=1; %Clase Tres
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass4,2)-1;
y(5,sizeIni:sizeEnd)=1; %Clase Cuatro
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass5,2)-1;
y(6,sizeIni:sizeEnd)=1; %Clase Cinco
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass6,2)-1;
y(7,sizeIni:sizeEnd)=1; %Clase Seis
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass7,2)-1;
y(8,sizeIni:sizeEnd)=1; %Clase Siete
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass8,2)-1;
y(9,sizeIni:sizeEnd)=1; %Clase Ocho
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass9,2)-1;
y(10,sizeIni:sizeEnd)=1; %Clase Nueve
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass10,2)-1;
y(11,sizeIni:sizeEnd)=1; %Clase Diez
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass11,2)-1;
y(12,sizeIni:sizeEnd)=1; %Clase Once
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass12,2)-1;
y(13,sizeIni:sizeEnd)=1; %Clase Doce
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass13,2)-1;
y(14,sizeIni:sizeEnd)=1; %Clase Trece
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass14,2)-1;
y(15,sizeIni:sizeEnd)=1; %Clase Catorce
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass15,2)-1;
y(16,sizeIni:sizeEnd)=1; %Clase Quince
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass16,2)-1;
y(17,sizeIni:sizeEnd)=1; %Clase Dieciseis

net=newpnn(x,y,0.7);
view(net);

save('redRBF_DWT','net');
disp('PROCESS DONE TRAINING NET !!!')