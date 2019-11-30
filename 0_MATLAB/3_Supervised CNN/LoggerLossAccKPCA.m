clear, clc, close all;
dataset = 'Urban';  % IndianPines  Salinas  PaviaU ...
numTest = 10;             % número de pruebas

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Prueba No.1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test = 'pcaCNN2D';         % pcaCNN2D kpcaInception
path = strcat('../../6_data Logger/',test,'/',dataset,'/logger_',dataset,'.txt');
data = load(path);
numEpochs = size(data,2);  % número de iteraciones
x=1:numEpochs;
loss=zeros(numTest,size(data,2)); j=1;
for i=1:4:4*numTest
   loss(j,:) = data(i,:);
   j = j+1;
end
std_loss = std(loss);
loss = sum(loss)/numTest;
acc=zeros(10,size(data,2)); j=1;
for i=3:4:4*numTest
   acc(j,:) = data(i,:);
   j = j+1;
end
std_acc = std(acc);
acc = sum(acc)/numTest;
subplot(2,1,1)
plot(x,loss,'-o'); hold on; grid on  
subplot(2,1,2)
plot(x,acc, '-o'); hold on; grid on 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Prueba No.2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test = 'pcaSCAE_v2';         % pcaCNN2D  pcaSCAE_v2 kpcaInception 
path = strcat('../../6_data Logger/',test,'/',dataset,'/logger_',dataset,'.txt');
data = load(path);
numEpochs = size(data,2);  % número de iteraciones
x=1:numEpochs;
loss=zeros(numTest,size(data,2)); j=1;
for i=1:4:4*numTest
   loss(j,:) = data(i,:);
   j = j+1;
end
std_loss = std(loss);
loss = sum(loss)/numTest;
acc=zeros(10,size(data,2)); j=1;
for i=3:4:4*numTest
   acc(j,:) = data(i,:);
   j = j+1;
end
std_acc = std(acc);
acc = sum(acc)/numTest;
subplot(2,1,1)
% plot(x(1:35),loss(1:35)+0.28,'-p'); hold on; grid on  
% subplot(2,1,2)
% plot(x(1:35),acc(1:35)-0.05, '-p'); hold on; grid on 
plot(x(1:35),loss(1:35),'-p'); hold on; grid on  
subplot(2,1,2)
plot(x(1:35),acc(1:35), '-p'); hold on; grid on 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Prueba No.3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test = 'kpcaInception';         % pcaCNN2D kpcaInception
path = strcat('../../6_data Logger/',test,'/',dataset,'/logger_',dataset,'.txt');
data = load(path);
numEpochs = size(data,2);  % número de iteraciones
x=1:numEpochs;
loss=zeros(numTest,size(data,2)); j=1;
for i=1:4:4*numTest
   loss(j,:) = data(i,:);
   j = j+1;
end
std_loss = std(loss);
loss = sum(loss)/numTest;
acc=zeros(10,size(data,2)); j=1;
for i=3:4:4*numTest
   acc(j,:) = data(i,:);
   j = j+1;
end
std_acc = std(acc);
acc = sum(acc)/numTest;
subplot(2,1,1)
plot(x,loss,'-x'); grid on
legend('PCA+CNN','PCA+SCAE','KPCA+Inception')
title('Función de Costo'); ylabel('Costo')
subplot(2,1,2)
plot(x,acc,'-x'); grid on
legend('PCA+CNN','PCA+SCAE','KPCA+Inception','Location','SouthEast')
title('Precisión Entrenamiento'); ylabel('Precisión'); xlabel('iteraciones')
