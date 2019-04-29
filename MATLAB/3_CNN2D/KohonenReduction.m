%KOHONEN DIMENSIONAL REDUCTION
clear, clc, close all;
Nneuronas=[5 5];        %Mapa Auto-Organizado de Kohonen
epochs = 5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IndianPines = 1 Salinas = 2 PaviaU = 3  Pavia = 4
[data,groundTh] = loadData(1);
imgSize=size(data);
k=1;
x=zeros(imgSize(3),imgSize(1)*imgSize(2)); %Matriz Entrada RNA 
for i=1:imgSize(1)
    for j=1:imgSize(2)
        if groundTh(i,j)~=0
            tempvect=data(i,j,:);
            x(:,k)=tempvect(:);
        end
        k=k+1;
    end
end

net=selforgmap(Nneuronas);
net.trainParam.epochs=epochs;
net=train(net,x);
wi=net.IW{1}; %Pesos Asociados a Cada Entrada de la Red Neuronal

dataSOM=zeros(imgSize(1),imgSize(2),size(wi,1)); %Reduccion Dimensional
for i=1:imgSize(1)
    for j=1:imgSize(2)
        tempvect=data(i,j,:);
        dataSOM(i,j,:)=wi*tempvect(:);
    end
end
save('netSOM_1','net');
save('dataSOM_1','dataSOM');
disp('PROCESS DONE 1 !!!')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IndianPines = 1 Salinas = 2 PaviaU = 3  Pavia = 4
[data,groundTh] = loadData(2);
imgSize=size(data);
k=1;
x=zeros(imgSize(3),imgSize(1)*imgSize(2)); %Matriz Entrada RNA 
for i=1:imgSize(1)
    for j=1:imgSize(2)
        if groundTh(i,j)~=0
            tempvect=data(i,j,:);
            x(:,k)=tempvect(:);
        end
        k=k+1;
    end
end

net=selforgmap(Nneuronas);
net.trainParam.epochs=epochs;
net=train(net,x);
wi=net.IW{1}; %Pesos Asociados a Cada Entrada de la Red Neuronal

dataSOM=zeros(imgSize(1),imgSize(2),size(wi,1)); %Reduccion Dimensional
for i=1:imgSize(1)
    for j=1:imgSize(2)
        tempvect=data(i,j,:);
        dataSOM(i,j,:)=wi*tempvect(:);
    end
end
save('netSOM_2','net');
save('dataSOM_2','dataSOM');
disp('PROCESS DONE 2 !!!')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% IndianPines = 1 Salinas = 2 PaviaU = 3  Pavia = 4
[data,groundTh] = loadData(3);
imgSize=size(data);
k=1;
x=zeros(imgSize(3),imgSize(1)*imgSize(2)); %Matriz Entrada RNA 
for i=1:imgSize(1)
    for j=1:imgSize(2)
        if groundTh(i,j)~=0
            tempvect=data(i,j,:);
            x(:,k)=tempvect(:);
        end
        k=k+1;
    end
end

net=selforgmap(Nneuronas);
net.trainParam.epochs=epochs;
net=train(net,x);
wi=net.IW{1}; %Pesos Asociados a Cada Entrada de la Red Neuronal

dataSOM=zeros(imgSize(1),imgSize(2),size(wi,1)); %Reduccion Dimensional
for i=1:imgSize(1)
    for j=1:imgSize(2)
        tempvect=data(i,j,:);
        dataSOM(i,j,:)=wi*tempvect(:);
    end
end
save('netSOM_3','net');
save('dataSOM_3','dataSOM');
disp('PROCESS DONE 3 !!!')