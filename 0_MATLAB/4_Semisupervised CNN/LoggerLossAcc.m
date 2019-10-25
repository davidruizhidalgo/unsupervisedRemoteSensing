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
plot(x,loss,'-o'); hold on; grid on  %std_loss
subplot(2,1,2)
plot(x,acc, '-o'); hold on; grid on  %std_acc

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
plot(x,loss,'-x'); grid on
subplot(2,1,2)
plot(x,acc,'-x'); grid on

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
plot(x,loss,'-p'); grid on
subplot(2,1,2)
plot(x,acc,'-p'); grid on

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
plot(x,loss,'-d'); grid on
legend('PCA+SCAE','EEP+SCAE','PCA+BCAE','EEP+BCAE')
% legend('SCAE','BCAE')
title('Training Loss'); ylabel('loss')
subplot(2,1,2)
plot(x,acc,'-d'); grid on
legend('PCA+SCAE','EEP+SCAE','PCA+BCAE','EEP+BCAE','Location','SouthEast')
% legend('SCAE','BCAE','Location','SouthEast')
title('Training Accuracy'); ylabel('accuracy'); xlabel('epoch')



