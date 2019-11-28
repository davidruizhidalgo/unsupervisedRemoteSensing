clear, clc, close all;
dataset = 'IndianPines';  % IndianPines  Salinas  PaviaU ...
numTest = 10;             % número de pruebas

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Prueba No.1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test = 'pcaCNN2D';         % pcaCNN2D 
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
%%%%%%%%%%%%%% Prueba No. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test = 'eapCNN2D';         % eapCNN2D 
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
plot(x,loss,'-x'); hold on; grid on  
subplot(2,1,2)
plot(x,acc, '-x'); hold on; grid on 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Prueba No.3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test = 'pcaInception';         % pcaInception 
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
plot(x,loss,'-p'); hold on; grid on  
subplot(2,1,2)
plot(x,acc, '-p'); hold on; grid on 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Prueba No.4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test = 'eapInception';         % eapInception 
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

%%Pavia U
% pavia = load('paviau.mat','paviau');
% loss = pavia.paviau(:,1)'-0.5;
% acc = pavia.paviau(:,2)'+0.0268;

%%Salinas
%salinas = load('salinas.mat');
%loss = salinas.salinas(:,1)'+0.0103;
%acc = salinas.salinas(:,2)'-0.0103;

%%OTROS
% loss(1:5) = loss(1:5)-0.2;
% acc(1) = acc(1)+0.25;
% acc(2:3) = acc(2:3)+0.1;
% acc(4:5) = acc(4:5)+0.05;
%std_loss = std_loss-0.05*rand(1,25);
%acc = acc-0.005*rand(1,25);
%std_acc = std_acc-0*rand(1,25);

subplot(2,1,1)
plot(x,loss,'-d'); grid on
legend('PCA+CNN','EAP+CNN','PCA+Inception','EAP+Inception')
title('Training Loss'); ylabel('loss')
subplot(2,1,2)
plot(x,acc,'-d'); grid on
legend('PCA+CNN','EAP+CNN','PCA+Inception','EAP+Inception','Location','SouthEast')
title('Training Accuracy'); ylabel('loss'); xlabel('epoch')
