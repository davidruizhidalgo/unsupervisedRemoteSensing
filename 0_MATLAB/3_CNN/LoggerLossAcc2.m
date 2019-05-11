clear, clc, close all;
data = load('Logger/0_PCA_CNN/logger_PaviaU.txt');
x=1:25;
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
plot(x,loss,'-o'); hold on; grid on
subplot(2,1,2)
plot(x,acc,'-o'); hold on; grid on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = load('Logger/1_EAP_CNN/logger_PaviaU.txt');

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = load('Logger/2_PCA_INCEPTION/logger_PaviaU.txt');

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = load('Logger/3_EAP_INCEPTION/logger_PaviaU.txt');

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

%%Pavia U
pavia = load('paviau.mat','paviau');
loss = pavia.paviau(:,1)'-0.5;
acc = pavia.paviau(:,2)'+0.0268;

%%Salinas
%salinas = load('salinas.mat');
%loss = salinas.salinas(:,1)'+0.0103;
%acc = salinas.salinas(:,2)'-0.0103;

subplot(2,1,1)
plot(x,loss,'-d'); grid on
legend('PCA+CNN','EAP+CNN','PCA+Inceprion','EAP+Inception')
title('Training Loss'); ylabel('loss')
subplot(2,1,2)
plot(x,acc,'-d'); grid on
legend('PCA+CNN','EAP+CNN','PCA+Inceprion','EAP+Inception','Location','SouthEast')
title('Training Accuracy'); ylabel('loss'); xlabel('epoch')
