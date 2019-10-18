clear, clc, close all;

dataset = 'IndianPines';  % IndianPines  Salinas  PaviaU
test = 'SCAE_v2';         %  pcaSCAE_v2  SCAE_v2  pcaBSCAE_v2 BSCAE_v2

path = strcat('../../6_data Logger/',test,'/',dataset,'/logger_',dataset,'_TEST.txt');
data = load(path);

OA = zeros(1,10); j=1;
for i=1:3:30
    OA(j) = data(i,1);
    j=j+1;
end
OA_std = std(OA);
OA = sum(OA)/10;

AA = zeros(10,size(data,2)); j=1;
for i=2:3:30
    AA(j,:) = data(i,:);
    j = j+1;
end
SDT_AA = std(AA);
STD_AA_P = sum(SDT_AA)/(numel(SDT_AA)-1);
AA = sum(AA)/10;
AA_P = sum(AA)/(numel(AA)-1);

kappa = zeros(1,10); j=1;
for i=3:3:30
    kappa(j) = data(i,1);
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