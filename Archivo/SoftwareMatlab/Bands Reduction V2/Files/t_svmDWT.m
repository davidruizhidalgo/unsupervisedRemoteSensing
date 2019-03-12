%SCRIPT ENTRENAMIENTO MAQUINA SOPORTE VECTORIAL
clear, clc, close all;

trainData=load('datosEntrenamientoDWT.mat');

%MATRIZ DATOS DE ENTRADA
x=[trainData.dataClass0,trainData.dataClass1,trainData.dataClass2,trainData.dataClass3...
    trainData.dataClass4,trainData.dataClass5,trainData.dataClass6,trainData.dataClass7...
    trainData.dataClass8,trainData.dataClass9,trainData.dataClass10,trainData.dataClass11...
    trainData.dataClass12,trainData.dataClass13,trainData.dataClass14,trainData.dataClass15,trainData.dataClass16];

x=x';

%MATRIZ DATOS DE SALIDA
%Train the SVM Classifier CLASS 0
y=-ones(1,size(x,1)); % 17 Clases de Vegetación
sizeIni=1; sizeEnd=size(trainData.dataClass0,2);
y(1,sizeIni:sizeEnd)=1; %Clase Cero 
y=y';
c0 = fitcsvm(x,y,'KernelFunction','rbf',...
    'BoxConstraint',Inf,'ClassNames',[1,-1]);
disp('c0 SVM DONE !!!!');

%Train the SVM Classifier CLASS 1
y=-ones(1,size(x,1)); % 17 Clases de Vegetación
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass1,2)-1;
y(sizeIni:sizeEnd)=1; %Clase Uno
y=y';
c1 = fitcsvm(x,y,'KernelFunction','rbf',...
    'BoxConstraint',Inf,'ClassNames',[1,-1]);
disp('c1 SVM DONE !!!!');

%Train the SVM Classifier CLASS 2
y=-ones(1,size(x,1)); % 17 Clases de Vegetación
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass2,2)-1;
y(sizeIni:sizeEnd)=1; %Clase Dos
y=y';
c2 = fitcsvm(x,y,'KernelFunction','rbf',...
    'BoxConstraint',Inf,'ClassNames',[1,-1]);
disp('c2 SVM DONE !!!!');

%Train the SVM Classifier CLASS 3
y=-ones(1,size(x,1)); % 17 Clases de Vegetación
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass3,2)-1;
y(sizeIni:sizeEnd)=1; %Clase Tres
y=y';
c3 = fitcsvm(x,y,'KernelFunction','rbf',...
    'BoxConstraint',Inf,'ClassNames',[1,-1]);
disp('c3 SVM DONE !!!!');

%Train the SVM Classifier CLASS 4
y=-ones(1,size(x,1)); % 17 Clases de Vegetación
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass4,2)-1;
y(sizeIni:sizeEnd)=1; %Clase Cuatro
y=y';
c4 = fitcsvm(x,y,'KernelFunction','rbf',...
    'BoxConstraint',Inf,'ClassNames',[1,-1]);
disp('c4 SVM DONE !!!!');

%Train the SVM Classifier CLASS 5
y=-ones(1,size(x,1)); % 17 Clases de Vegetación
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass5,2)-1;
y(sizeIni:sizeEnd)=1; %Clase Cinco
y=y';
c5 = fitcsvm(x,y,'KernelFunction','rbf',...
    'BoxConstraint',Inf,'ClassNames',[1,-1]);
disp('c5 SVM DONE !!!!');

%Train the SVM Classifier CLASS 6
y=-ones(1,size(x,1)); % 17 Clases de Vegetación
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass6,2)-1;
y(sizeIni:sizeEnd)=1; %Clase Seis
y=y';
c6 = fitcsvm(x,y,'KernelFunction','rbf',...
    'BoxConstraint',Inf,'ClassNames',[1,-1]);
disp('c6 SVM DONE !!!!');

%Train the SVM Classifier CLASS 7
y=-ones(1,size(x,1)); % 17 Clases de Vegetación
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass7,2)-1;
y(sizeIni:sizeEnd)=1; %Clase Siete
y=y';
c7 = fitcsvm(x,y,'KernelFunction','rbf',...
    'BoxConstraint',Inf,'ClassNames',[1,-1]);
disp('c7 SVM DONE !!!!');

%Train the SVM Classifier CLASS 8
y=-ones(1,size(x,1)); % 17 Clases de Vegetación
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass8,2)-1;
y(sizeIni:sizeEnd)=1; %Clase Ocho
y=y';
c8 = fitcsvm(x,y,'KernelFunction','rbf',...
    'BoxConstraint',Inf,'ClassNames',[1,-1]);
disp('c8 SVM DONE !!!!');

%Train the SVM Classifier CLASS 9
y=-ones(1,size(x,1)); % 17 Clases de Vegetación
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass9,2)-1;
y(sizeIni:sizeEnd)=1; %Clase Nueve
y=y';
c9 = fitcsvm(x,y,'KernelFunction','rbf',...
    'BoxConstraint',Inf,'ClassNames',[1,-1]);
disp('c9 SVM DONE !!!!');

%Train the SVM Classifier CLASS 10
y=-ones(1,size(x,1)); % 17 Clases de Vegetación
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass10,2)-1;
y(sizeIni:sizeEnd)=1; %Clase Diez
y=y';
c10 = fitcsvm(x,y,'KernelFunction','rbf',...
    'BoxConstraint',Inf,'ClassNames',[1,-1]);
disp('c10 SVM DONE !!!!');

%Train the SVM Classifier CLASS 11
y=-ones(1,size(x,1)); % 17 Clases de Vegetación
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass11,2)-1;
y(sizeIni:sizeEnd)=1; %Clase Once
y=y';
c11 = fitcsvm(x,y,'KernelFunction','rbf',...
    'BoxConstraint',Inf,'ClassNames',[1,-1]);
disp('c11 SVM DONE !!!!');

%Train the SVM Classifier CLASS 12
y=-ones(1,size(x,1)); % 17 Clases de Vegetación
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass12,2)-1;
y(sizeIni:sizeEnd)=1; %Clase Doce
y=y';
c12 = fitcsvm(x,y,'KernelFunction','rbf',...
    'BoxConstraint',Inf,'ClassNames',[1,-1]);
disp('c12 SVM DONE !!!!');

%Train the SVM Classifier CLASS 13
y=-ones(1,size(x,1)); % 17 Clases de Vegetación
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass13,2)-1;
y(sizeIni:sizeEnd)=1; %Clase Trece
y=y';
c13= fitcsvm(x,y,'KernelFunction','rbf',...
    'BoxConstraint',Inf,'ClassNames',[1,-1]);
disp('c13 SVM DONE !!!!');

%Train the SVM Classifier CLASS 14
y=-ones(1,size(x,1)); % 17 Clases de Vegetación
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass14,2)-1;
y(sizeIni:sizeEnd)=1; %Clase Catorce
y=y';
c14 = fitcsvm(x,y,'KernelFunction','rbf',...
    'BoxConstraint',Inf,'ClassNames',[1,-1]);
disp('c14 SVM DONE !!!!');

%Train the SVM Classifier CLASS 15
y=-ones(1,size(x,1)); % 17 Clases de Vegetación
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass15,2)-1;
y(sizeIni:sizeEnd)=1; %Clase Quince
y=y';
c15 = fitcsvm(x,y,'KernelFunction','rbf',...
    'BoxConstraint',Inf,'ClassNames',[1,-1]);
disp('c15 SVM DONE !!!!');

%Train the SVM Classifier CLASS 16
y=-ones(1,size(x,1)); % 17 Clases de Vegetación
sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass16,2)-1;
y(sizeIni:sizeEnd)=1; %Clase Dieciseis
y=y';
c16 = fitcsvm(x,y,'KernelFunction','rbf',...
    'BoxConstraint',Inf,'ClassNames',[1,-1]);
disp('c16 SVM DONE !!!!');



save('svmDWT','c0','c1','c2','c3','c4','c5','c6','c7','c8','c9','c10','c11',...
    'c12','c13','c14','c15','c16');

disp(':::::::::::::::::::PROCESS COMPLETED::::::::::::::::::');



