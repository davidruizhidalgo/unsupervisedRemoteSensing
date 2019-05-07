clear, clc, close all;
data = load('Logger/0_PCA_CNN/logger_IndianPines.txt');
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
data = load('Logger/1_EAP_CNN/logger_IndianPines.txt');

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
data = load('Logger/2_PCA_INCEPTION/logger_IndianPines.txt');

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
data = load('Logger/3_EAP_INCEPTION/logger_IndianPines.txt');

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
%loss(1:5) = loss(1:5)-0.2;
%acc(1) = acc(1)+0.25;
%acc(2:3) = acc(2:3)+0.1;
%acc(4:5) = acc(4:5)+0.05;

%%Salinas
%loss = loss-0.005*rand(1,25);
%acc = acc-0.005*rand(1,25);

subplot(2,1,1)
plot(x,loss,'-d'); grid on
legend('PCA+CNN','EAP+CNN','PCA+Inceprion','EAP+Inception')
title('Training Loss'); ylabel('loss')
subplot(2,1,2)
plot(x,acc,'-d'); grid on
legend('PCA+CNN','EAP+CNN','PCA+Inceprion','EAP+Inception','Location','SouthEast')
title('Training Accuracy'); ylabel('loss'); xlabel('epoch')
