clear, clc, close all;
data2 = load('Logger/00_PCA_CNN/logger_IndianPines_TEST.txt');
% 00_PCA_CNN  6_KPCA_CNN  7_KPCA_INCEPTION 
% IndianPines  Salinas  PaviaU

OA = zeros(1,10); j=1;
for i=1:3:30
    OA(j) = data2(i,1);
    j=j+1;
end
OA_std = std(OA);
OA = sum(OA)/10;

AA = zeros(10,size(data2,2)); j=1;
for i=2:3:30
    AA(j,:) = data2(i,:);
    j = j+1;
end
SDT_AA = std(AA);
STD_AA_P = sum(SDT_AA)/(numel(SDT_AA)-1);
AA = sum(AA)/10;
AA_P = sum(AA)/(numel(AA)-1);

kappa = zeros(1,10); j=1;
for i=3:3:30
    kappa(j) = data2(i,1);
    j=j+1;
end
kappa_std = std(kappa); 
kappa = sum(kappa)/10;

disp('AA')
disp(AA')
disp('Desviacion AA')
disp(SDT_AA') 
disp('Promedio AA')
disp(AA_P);
disp('Promedio Desviacion AA')
disp(STD_AA_P);
disp('OA')
disp(OA);
disp('Desviacion OA')
disp(OA_std);
disp('Kappa')
disp(kappa);
disp('Desviacion Kappa')
disp(kappa_std);