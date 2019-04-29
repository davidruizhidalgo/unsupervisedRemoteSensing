%DATOS ENTRENAMIENTO CON PCA. 
clear, clc, close all;
sData=load('../../Salinas_gt.mat');
dataGround=sData.salinas_gt;
imgSize=size(dataGround);

class0=zeros(1,3); %Clase de Vegetación Cero
class1=zeros(1,3); %Clase de Vegetación Uno
class2=zeros(1,3); %Clase de Vegetación Dos
class3=zeros(1,3); %Clase de Vegetación Tres
class4=zeros(1,3); %Clase de Vegetación Cuatro
class5=zeros(1,3); %Clase de Vegetación Cinco
class6=zeros(1,3); %Clase de Vegetación Seis
class7=zeros(1,3); %Clase de Vegetación Siete
class8=zeros(1,3); %Clase de Vegetación Ocho
class9=zeros(1,3); %Clase de Vegetación Nueve
class10=zeros(1,3); %Clase de Vegetación Diez
class11=zeros(1,3); %Clase de Vegetación Once
class12=zeros(1,3); %Clase de Vegetación Doce
class13=zeros(1,3); %Clase de Vegetación Trece
class14=zeros(1,3); %Clase de Vegetación Catorce
class15=zeros(1,3); %Clase de Vegetación Quince
class16=zeros(1,3); %Clase de Vegetación Dieciseis

%TOMAR DE LA IMAGEN (TABLA DE VERDAD)TODOS LOS ELEMENTOS 
%ASOCIADOS A CADA UNA DE LAS CLASES DE VEGETACIÓN 

for i=1:imgSize(1)
    for j=1:imgSize(2)
        val=dataGround(i,j);
        switch val
            case 0
               class0(end+1,:)= [i,j,val];
            case 1
                class1(end+1,:)= [i,j,val];
            case 2
                class2(end+1,:)= [i,j,val];
            case 3
                class3(end+1,:)= [i,j,val];
            case 4
                class4(end+1,:)= [i,j,val];
            case 5
                class5(end+1,:)= [i,j,val];
            case 6
                class6(end+1,:)= [i,j,val];
            case 7
                class7(end+1,:)= [i,j,val];
            case 8
                class8(end+1,:)= [i,j,val];
            case 9
                class9(end+1,:)= [i,j,val];
            case 10
                class10(end+1,:)= [i,j,val];
            case 11
                class11(end+1,:)= [i,j,val];
            case 12
                class12(end+1,:)= [i,j,val];
            case 13
                class13(end+1,:)= [i,j,val];
            case 14
                class14(end+1,:)= [i,j,val];
            case 15
                class15(end+1,:)= [i,j,val];
            case 16
                class16(end+1,:)= [i,j,val];
            otherwise
                disp('INVALID CLASS')
        end
    end
end

class0=class0(2:end,:); %Clase de Vegetación Cero
class1=class1(2:end,:); %Clase de Vegetación Uno
class2=class2(2:end,:); %Clase de Vegetación Dos
class3=class3(2:end,:); %Clase de Vegetación Tres
class4=class4(2:end,:); %Clase de Vegetación Cuatro
class5=class5(2:end,:); %Clase de Vegetación Cinco
class6=class6(2:end,:); %Clase de Vegetación Seis
class7=class7(2:end,:); %Clase de Vegetación Siete
class8=class8(2:end,:); %Clase de Vegetación Ocho
class9=class9(2:end,:); %Clase de Vegetación Nueve
class10=class10(2:end,:); %Clase de Vegetación Diez
class11=class11(2:end,:); %Clase de Vegetación Once
class12=class12(2:end,:); %Clase de Vegetación Doce
class13=class13(2:end,:); %Clase de Vegetación Trece
class14=class14(2:end,:); %Clase de Vegetación Catorce
class15=class15(2:end,:); %Clase de Vegetación Quince
class16=class16(2:end,:); %Clase de Vegetación Dieciseis.

%CARGAR CUBO DE DATOS ESPECTRALES
%[File,Path]=uigetfile({'*.*'},'CARGAR COMPONENTES PRINCIPALES');
%filename=strcat(Path,File);
sData=load('datosPCA');
dataPCA=sData.dataPCA;
imgSize=size(dataPCA);

porData=0.5;    %Porcentaje de Datos para Entrenamiento
dataClass0=zeros(imgSize(3),ceil(porData*size(class0,1)));     %Clase de Vegetación Cero
dataClass1=zeros(imgSize(3),ceil(porData*size(class1,1)));     %Clase de Vegetación Uno
dataClass2=zeros(imgSize(3),ceil(porData*size(class2,1)));     %Clase de Vegetación Dos
dataClass3=zeros(imgSize(3),ceil(porData*size(class3,1)));     %Clase de Vegetación Tres
dataClass4=zeros(imgSize(3),ceil(porData*size(class4,1)));     %Clase de Vegetación Cuatro
dataClass5=zeros(imgSize(3),ceil(porData*size(class5,1)));     %Clase de Vegetación Cinco
dataClass6=zeros(imgSize(3),ceil(porData*size(class6,1)));     %Clase de Vegetación Seis
dataClass7=zeros(imgSize(3),ceil(porData*size(class7,1)));     %Clase de Vegetación Siete
dataClass8=zeros(imgSize(3),ceil(porData*size(class8,1)));     %Clase de Vegetación Ocho
dataClass9=zeros(imgSize(3),ceil(porData*size(class9,1)));     %Clase de Vegetación Nueve
dataClass10=zeros(imgSize(3),ceil(porData*size(class10,1)));   %Clase de Vegetación Diez
dataClass11=zeros(imgSize(3),ceil(porData*size(class11,1)));   %Clase de Vegetación Once
dataClass12=zeros(imgSize(3),ceil(porData*size(class12,1)));   %Clase de Vegetación Doce
dataClass13=zeros(imgSize(3),ceil(porData*size(class13,1)));   %Clase de Vegetación Trece
dataClass14=zeros(imgSize(3),ceil(porData*size(class14,1)));   %Clase de Vegetación Catorce
dataClass15=zeros(imgSize(3),ceil(porData*size(class15,1)));   %Clase de Vegetación Quince
dataClass16=zeros(imgSize(3),ceil(porData*size(class16,1)));   %Clase de Vegetación Dieciseis.

%Clase de Vegetación Cero
dataToTrain=1:size(class0,1);
dataToTrain=dataToTrain(randperm(length(dataToTrain)));
j=1;
for i=1:size(dataClass0,2)
    k=dataToTrain(j);
    pxData=dataPCA(class0(k,1),class0(k,2),:);
    dataClass0(:,i)= pxData(:);
    j=j+1;  
end

%Clase de Vegetación Uno
dataToTrain=1:size(class1,1);
dataToTrain=dataToTrain(randperm(length(dataToTrain)));
j=1;
for i=1:size(dataClass1,2)
    k=dataToTrain(j);
    pxData=dataPCA(class1(k,1),class1(k,2),:);
    dataClass1(:,i)= pxData(:);
    j=j+1;  
end

%Clase de Vegetación Dos
dataToTrain=1:size(class2,1);
dataToTrain=dataToTrain(randperm(length(dataToTrain)));
j=1;
for i=1:size(dataClass2,2)
    k=dataToTrain(j);
    pxData=dataPCA(class2(k,1),class2(k,2),:);
    dataClass2(:,i)= pxData(:);
    j=j+1;  
end

%Clase de Vegetación Tres
dataToTrain=1:size(class3,1);
dataToTrain=dataToTrain(randperm(length(dataToTrain)));
j=1;
for i=1:size(dataClass3,2)
    k=dataToTrain(j);
    pxData=dataPCA(class3(k,1),class3(k,2),:);
    dataClass3(:,i)= pxData(:);
    j=j+1;  
end

%Clase de Vegetación Cuatro
dataToTrain=1:size(class4,1);
dataToTrain=dataToTrain(randperm(length(dataToTrain)));
j=1;
for i=1:size(dataClass4,2)
    k=dataToTrain(j);
    pxData=dataPCA(class4(k,1),class4(k,2),:);
    dataClass4(:,i)= pxData(:);
    j=j+1;  
end

%Clase de Vegetación Cinco
dataToTrain=1:size(class5,1);
dataToTrain=dataToTrain(randperm(length(dataToTrain)));
j=1;
for i=1:size(dataClass5,2)
    k=dataToTrain(j);
    pxData=dataPCA(class5(k,1),class5(k,2),:);
    dataClass5(:,i)= pxData(:);
    j=j+1;  
end

%Clase de Vegetación Seis
dataToTrain=1:size(class6,1);
dataToTrain=dataToTrain(randperm(length(dataToTrain)));
j=1;
for i=1:size(dataClass6,2)
    k=dataToTrain(j);
    pxData=dataPCA(class6(k,1),class6(k,2),:);
    dataClass6(:,i)= pxData(:);
    j=j+1;  
end

%Clase de Vegetación Siete
dataToTrain=1:size(class7,1);
dataToTrain=dataToTrain(randperm(length(dataToTrain)));
j=1;
for i=1:size(dataClass7,2)
    k=dataToTrain(j);
    pxData=dataPCA(class7(k,1),class7(k,2),:);
    dataClass7(:,i)= pxData(:);
    j=j+1;  
end

%Clase de Vegetación Ocho
dataToTrain=1:size(class8,1);
dataToTrain=dataToTrain(randperm(length(dataToTrain)));
j=1;
for i=1:size(dataClass8,2)
    k=dataToTrain(j);
    pxData=dataPCA(class8(k,1),class8(k,2),:);
    dataClass8(:,i)= pxData(:);
    j=j+1;  
end

%Clase de Vegetación Nueve
dataToTrain=1:size(class9,1);
dataToTrain=dataToTrain(randperm(length(dataToTrain)));
j=1;
for i=1:size(dataClass9,2)
    k=dataToTrain(j);
    pxData=dataPCA(class9(k,1),class9(k,2),:);
    dataClass9(:,i)= pxData(:);
    j=j+1;  
end

%Clase de Vegetación Diez
dataToTrain=1:size(class10,1);
dataToTrain=dataToTrain(randperm(length(dataToTrain)));
j=1;
for i=1:size(dataClass10,2)
    k=dataToTrain(j);
    pxData=dataPCA(class10(k,1),class10(k,2),:);
    dataClass10(:,i)= pxData(:);
    j=j+1;  
end

%Clase de Vegetación Once
dataToTrain=1:size(class11,1);
dataToTrain=dataToTrain(randperm(length(dataToTrain)));
j=1;
for i=1:size(dataClass11,2)
    k=dataToTrain(j);
    pxData=dataPCA(class11(k,1),class11(k,2),:);
    dataClass11(:,i)= pxData(:);
    j=j+1;  
end

%Clase de Vegetación Doce
dataToTrain=1:size(class12,1);
dataToTrain=dataToTrain(randperm(length(dataToTrain)));
j=1;
for i=1:size(dataClass12,2)
    k=dataToTrain(j);
    pxData=dataPCA(class12(k,1),class12(k,2),:);
    dataClass12(:,i)= pxData(:);
    j=j+1;  
end

%Clase de Vegetación Trece
dataToTrain=1:size(class13,1);
dataToTrain=dataToTrain(randperm(length(dataToTrain)));
j=1;
for i=1:size(dataClass13,2)
    k=dataToTrain(j);
    pxData=dataPCA(class13(k,1),class13(k,2),:);
    dataClass13(:,i)= pxData(:);
    j=j+1;  
end

%Clase de Vegetación Catorce
dataToTrain=1:size(class14,1);
dataToTrain=dataToTrain(randperm(length(dataToTrain)));
j=1;
for i=1:size(dataClass14,2)
    k=dataToTrain(j);
    pxData=dataPCA(class14(k,1),class14(k,2),:);
    dataClass14(:,i)= pxData(:);
    j=j+1;  
end

%Clase de Vegetación Quince
dataToTrain=1:size(class15,1);
dataToTrain=dataToTrain(randperm(length(dataToTrain)));
j=1;
for i=1:size(dataClass15,2)
    k=dataToTrain(j);
    pxData=dataPCA(class15(k,1),class15(k,2),:);
    dataClass15(:,i)= pxData(:);
    j=j+1;  
end

%Clase de Vegetación Dieciseis
dataToTrain=1:size(class16,1);
dataToTrain=dataToTrain(randperm(length(dataToTrain)));
j=1;
for i=1:size(dataClass16,2)
    k=dataToTrain(j);
    pxData=dataPCA(class16(k,1),class16(k,2),:);
    dataClass16(:,i)= pxData(:);
    j=j+1;  
end

save('datosEntrenamientoPCA','dataClass0','dataClass1','dataClass2',...
    'dataClass3','dataClass4','dataClass5','dataClass6','dataClass7',...
    'dataClass8','dataClass9','dataClass10','dataClass11','dataClass12',...
    'dataClass13','dataClass14','dataClass15','dataClass16','-v7.3');

disp('PROCESS DONE GET TRAINING DATA !!!')
