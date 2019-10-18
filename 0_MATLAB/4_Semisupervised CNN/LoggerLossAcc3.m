clear, clc, close all;
data = load('Logger/00_PCA_CNN/logger_PaviaU.txt');
x=1:size(data,2);
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
% data = load('Logger/66_KPCA_CNN/logger_IndianPines.txt');
% x=1:size(data,2);
% loss=zeros(10,size(data,2)); j=1;
% for i=1:4:40
%    loss(j,:) = data(i,:);
%    j = j+1;
% end
% std_loss = std(loss);
% loss = sum(loss)/10;
% 
% acc=zeros(10,size(data,2)); j=1;
% for i=3:4:40
%    acc(j,:) = data(i,:);
%    j = j+1;
% end
% std_acc = std(acc);
% acc = sum(acc)/10;
% 
% subplot(2,1,1)
% plot(x,loss,'-x'); grid on
% subplot(2,1,2)
% plot(x,acc,'-x'); grid on

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data = load('Logger/7_KPCA_INCEPTION/logger_PaviaU.txt');
x=1:size(data,2);
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
legend('PCA+CNN','KPCA+Inceprion')
title('Función de Costo'); ylabel('Costo')
subplot(2,1,2)
plot(x,acc,'-p'); grid on
legend('PCA+CNN','KPCA+Inceprion','Location','SouthEast')
title('Precisión Entrenamiento'); ylabel('Precisión'); xlabel('iteraciones')
