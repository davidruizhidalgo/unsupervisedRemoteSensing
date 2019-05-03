%DATOS ENTRENAMIENTO CON KOHONEN
clear, clc, close all;
sData=load('../../paviaU_gt.mat');
dataGround=sData.paviaU_gt;
imgSizeGT=size(dataGround);

% DATOS GENERADOS CON MAPA AUTO-ORGANIZADO DE KOHONEN
sData=load('datosPCA');
dataPCA=sData.x;
imgSize=size(dataPCA);

yind=zeros(1,imgSize(2));%Etiquetas de Clase 
classSize=zeros(1,9); %total muestras por clase 
k=1;
for i=1:imgSizeGT(1)
    for j=1:imgSizeGT(2)
        if (dataGround(i,j)~=0) 
           val=dataGround(i,j);
           yind(k)=val;
           classSize(val)=classSize(val)+1;
           k=k+1;
        end
    end
end

dataClass=cell(9,1); 

for i=1:imgSize(2)
    dataClass{yind(i)}(:,end+1)=dataPCA(:,i);
end

porData=0.5;    %Porcentaje de Datos para Entrenamiento
y=0;            %VECTOR salida RNA
trainClass=cell(9,1); 
for c=1:size(trainClass,1)
    trainClass{c}=zeros(imgSize(1),ceil(porData*size(dataClass{c},2)));  
    dataToTrain=1:classSize(c);
    dataToTrain=dataToTrain(randperm(length(dataToTrain)));
    for i=1:size(trainClass{c},2)
        trainClass{c}(:,i)=dataClass{c}(:,dataToTrain(i));
        y(end+1)=c;
    end   
end
y=y(2:end); 
y=full(ind2vec(y,9));   % MATRIZ Salida RNA AD-HOC 

x=[trainClass{1},trainClass{2},trainClass{3},trainClass{4},trainClass{5},...
    trainClass{6},trainClass{7},trainClass{8},trainClass{9}];%MATRIZ de ENTRADA RNA %*****


net=newpnn(x,y,0.4);
view(net);

save('redRBF','net');
disp('PROCESS TRAINING DATA DONE !!!!!!!!')