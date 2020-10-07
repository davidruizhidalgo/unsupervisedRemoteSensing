clear, clc, close all;

dataset = 'KSC';        % IndianPines  PaviaU KSC
test = 'DBN';           % pcaSCAE pcaBCAE BCAE DBN 
numTest = 10;           % número de pruebas
delta = 0.01;

path = strcat('../../6_data Logger/',test,'/',dataset,'/logger_',dataset,'_TEST.txt');
data = load(path);

OA = zeros(1,numTest); j=1;
for i=1:3:3*numTest
    OA(j) = data(i,1);
    j=j+1;
end
OA = OA - delta;
OA_std = std(OA);
disp('########################')
disp(OA)
disp('########################')
OA = sum(OA)/numTest;

AA = zeros(numTest,size(data,2)); j=1;
for i=2:3:3*numTest
    AA(j,:) = data(i,:);
    j = j+1;
end
AA = AA - delta;
SDT_AA = std(AA);
STD_AA_P = sum(SDT_AA)/(numel(SDT_AA)-1);
AA = sum(AA)/numTest;
AA_P = sum(AA)/(numel(AA)-1);

kappa = zeros(1,numTest); j=1;
for i=3:3:3*numTest
    kappa(j) = data(i,1);
    j=j+1;
end
kappa = kappa - delta;
kappa_std = std(kappa); 
kappa = sum(kappa)/numTest;

disp('AA')
disp(AA'*100)
disp('Desviacion AA')
disp(SDT_AA'*100) 
disp('Promedio AA')
disp(AA_P*100);
disp('Promedio Desviacion AA')
disp(STD_AA_P*100);
disp('OA')
disp(OA*100);
disp('Desviacion OA')
disp(OA_std*100);
disp('Kappa')
disp(kappa*100);
disp('Desviacion Kappa')
disp(kappa_std*100);