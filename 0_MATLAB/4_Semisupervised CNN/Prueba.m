clear, clc, close all;
dataset = 'IndianPines';  % IndianPines  Salinas  PaviaU
numTest = 100;             % número de pruebas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Prueba No.1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test = 'BSCAE_v2';         % pcaSCAE_v2  SCAE_v2  pcaBSCAE_v2 BSCAE_v2
path = strcat('../../6_data Logger/',test,'/',dataset,'/logger_',dataset,'.txt');
data = load(path);
numEpochs = size(data,2);  % número de iteracione
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
errorbar(x,loss,std_loss,'-o'); hold on; grid on  
subplot(2,1,2)
errorbar(x,acc, std_acc,'-o'); hold on; grid on  