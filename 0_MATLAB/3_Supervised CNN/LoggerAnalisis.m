clear, clc, close all;

dataset = 'IndianPines';       % IndianPines  Salinas  PaviaU SalinasA Pavia Urban
test = 'pcaSCAE_v2';     %  select folder in 6_logger: ex. pcaCNN2D pcaSCAE_v2  kpcaInception 
numTest = 10;               % número de pruebas

path = strcat('../../6_data Logger/',test,'/',dataset,'/logger_',dataset,'_TEST.txt');
data = load(path);

OA = zeros(1,numTest); j=1;
for i=1:3:3*numTest
    OA(j) = data(i,1);
    j=j+1;
end
OA_std = std(OA);
OA = sum(OA)/numTest;

AA = zeros(numTest,size(data,2)); j=1;
for i=2:3:3*numTest
    AA(j,:) = data(i,:);
    j = j+1;
end
SDT_AA = std(AA);
STD_AA_P = sum(SDT_AA)/(numel(SDT_AA)-1);
AA = sum(AA)/numTest;
AA_P = sum(AA)/(numel(AA)-1);

kappa = zeros(1,numTest); j=1;
for i=3:3:3*numTest
    kappa(j) = data(i,1);
    j=j+1;
end
kappa_std = std(kappa); 
kappa = sum(kappa)/numTest;

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