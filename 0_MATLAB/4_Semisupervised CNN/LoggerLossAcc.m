clear, clc, close all;

dataset = 'IndianPines';  % IndianPines  Salinas  PaviaU
numEpochs = 50;           % n�mero de iteraciones
numTest = 100;             % n�mero de pruebas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Prueba No.1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test = 'pcaSCAE_v2';         % pcaSCAE_v2  SCAE_v2  pcaBSCAE_v2 BSCAE_v2
path = strcat('../../6_data Logger/',test,'/',dataset,'/logger_',dataset,'.txt');
data = load(path);

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
figure
subplot(2,1,1)
errorbar(x,loss,std_loss,'-o'); hold on; grid on  
subplot(2,1,2)
errorbar(x,acc, std_acc,'-o'); hold on; grid on 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Prueba No.2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test = 'SCAE_v2';         % pcaSCAE_v2  SCAE_v2  pcaBSCAE_v2 BSCAE_v2
path = strcat('../../6_data Logger/',test,'/',dataset,'/logger_',dataset,'.txt');
data = load(path);

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
figure
subplot(2,1,1)
errorbar(x,loss,std_loss,'-x'); hold on; grid on  
subplot(2,1,2)
errorbar(x,acc, std_acc,'-x'); hold on; grid on 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Prueba No.3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test = 'pcaBSCAE_v2';         % pcaSCAE_v2  SCAE_v2  pcaBSCAE_v2 BSCAE_v2
path = strcat('../../6_data Logger/',test,'/',dataset,'/logger_',dataset,'.txt');
data = load(path);

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
figure
subplot(2,1,1)
errorbar(x,loss,std_loss,'-p'); hold on; grid on  
subplot(2,1,2)
errorbar(x,acc, std_acc,'-p'); hold on; grid on 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Prueba No.4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test = 'BSCAE_v2';         % pcaSCAE_v2  SCAE_v2  pcaBSCAE_v2 BSCAE_v2
path = strcat('../../6_data Logger/',test,'/',dataset,'/logger_',dataset,'.txt');
data = load(path);

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
figure
subplot(2,1,1)
errorbar(x,loss,std_loss,'-d'); hold on; grid on  
legend('PCA+SCAE','EEP+SCAE','PCA+BCAE','EEP+BCAE')
title('Training Loss'); ylabel('loss')
subplot(2,1,2)
errorbar(x,acc, std_acc,'-d'); hold on; grid on 
legend('PCA+SCAE','EEP+SCAE','PCA+BCAE','EEP+BCAE','Location','SouthEast')
title('Training Accuracy'); ylabel('accuracy'); xlabel('epoch')




