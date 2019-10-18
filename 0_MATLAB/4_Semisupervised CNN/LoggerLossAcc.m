clear, clc, close all;

dataset = 'IndianPines';  % IndianPines  Salinas  PaviaU

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Prueba No.1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test = 'pcaSCAE_v2';         % pcaSCAE_v2  SCAE_v2  pcaBSCAE_v2 BSCAE_v2
path = strcat('../../6_data Logger/',test,'/',dataset,'/logger_',dataset,'.txt');
data = load(path);

x=1:50;
loss=zeros(10,size(data,2)); j=1;
for i=1:4:40
   loss(j,:) = data(i,:);
   j = j+1;
end
std_loss = std(loss);
loss = sum(loss)/10;

acc=zeros(10,size(data,2)); j=1;
for i=3:4:40
   acc(j,:) = data(i,:);
   j = j+1;
end
std_acc = std(acc);
acc = sum(acc)/10;

figure
subplot(2,1,1)
errorbar(x,loss,std_loss); hold on; grid on
subplot(2,1,2)
errorbar(x,acc,std_acc); hold on; grid on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Prueba No.2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test = 'SCAE_v2';         % pcaSCAE_v2  SCAE_v2  pcaBSCAE_v2 BSCAE_v2
path = strcat('../../6_data Logger/',test,'/',dataset,'/logger_',dataset,'.txt');
data = load(path);

loss=zeros(10,size(data,2)); j=1;
for i=1:4:40
   loss(j,:) = data(i,:);
   j = j+1;
end
std_loss = std(loss);
loss = sum(loss)/10;

acc=zeros(10,size(data,2)); j=1;
for i=3:4:40
   acc(j,:) = data(i,:);
   j = j+1;
end
std_acc = std(acc);
acc = sum(acc)/10;

subplot(2,1,1)
errorbar(x,loss,std_loss); grid on
subplot(2,1,2)
errorbar(x,acc,std_acc); grid on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Prueba No.3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test = 'pcaBSCAE_v2';         % pcaSCAE_v2  SCAE_v2  pcaBSCAE_v2 BSCAE_v2
path = strcat('../../6_data Logger/',test,'/',dataset,'/logger_',dataset,'.txt');
data = load(path);

loss=zeros(10,size(data,2)); j=1;
for i=1:4:40
   loss(j,:) = data(i,:);
   j = j+1;
end
std_loss = std(loss);
loss = sum(loss)/10;

acc=zeros(10,size(data,2)); j=1;
for i=3:4:40
   acc(j,:) = data(i,:);
   j = j+1;
end
std_acc = std(acc);
acc = sum(acc)/10;

subplot(2,1,1)
errorbar(x,loss,std_loss); grid on
subplot(2,1,2)
errorbar(x,acc,std_acc); grid on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Prueba No.4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test = 'BSCAE_v2';         % pcaSCAE_v2  SCAE_v2  pcaBSCAE_v2 BSCAE_v2
path = strcat('../../6_data Logger/',test,'/',dataset,'/logger_',dataset,'.txt');
data = load(path);

loss=zeros(10,size(data,2)); j=1;
for i=1:4:40
   loss(j,:) = data(i,:);
   j = j+1;
end
std_loss = std(loss);
loss = sum(loss)/10;

acc=zeros(10,size(data,2)); j=1;
for i=3:4:40
   acc(j,:) = data(i,:);
   j = j+1;
end
std_acc = std(acc);
acc = sum(acc)/10;

subplot(2,1,1)
errorbar(x,loss,std_loss); grid on
legend('PCA+SCAE','EEP+SCAE','PCA+BCAE','EEP+BCAE')
title('Training Loss'); ylabel('loss')
subplot(2,1,2)
errorbar(x,acc,std_acc); grid on
legend('PCA+SCAE','EEP+SCAE','PCA+BCAE','EEP+BCAE','Location','SouthEast')
title('Training Accuracy'); ylabel('loss'); xlabel('epoch')




