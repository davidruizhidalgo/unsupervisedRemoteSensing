clear, clc, close all;

dataset = 'IndianPines';  % IndianPines  Salinas  PaviaU

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Prueba No.1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test = 'pcaSCAE_v2';         %  pcaSCAE_v2  SCAE_v2  pcaBSCAE_v2 BSCAE_v2

path = strcat('../../6_data Logger/',test,'/',dataset,'/logger_',dataset,'_TEST.txt');
data = load(path);
OA = zeros(1,10); j=1;
for i=1:3:30
    OA(j) = data(i,1);
    j=j+1;
end
OA_LR1 = sum(OA)/10;

path = strcat('../../6_data Logger/',test,'/',dataset,'/logger_',dataset,'_RIEM.txt');
data = load(path);
OA = zeros(1,10); j=1;
for i=1:3:30
    OA(j) = data(i,1);
    j=j+1;
end
OA_RC1 = sum(OA)/10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Prueba No.2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test = 'SCAE_v2';         %  pcaSCAE_v2  SCAE_v2  pcaBSCAE_v2 BSCAE_v2

path = strcat('../../6_data Logger/',test,'/',dataset,'/logger_',dataset,'_TEST.txt');
data = load(path);
OA = zeros(1,10); j=1;
for i=1:3:30
    OA(j) = data(i,1);
    j=j+1;
end
OA_LR2 = sum(OA)/10;

path = strcat('../../6_data Logger/',test,'/',dataset,'/logger_',dataset,'_RIEM.txt');
data = load(path);
OA = zeros(1,10); j=1;
for i=1:3:30
    OA(j) = data(i,1);
    j=j+1;
end
OA_RC2 = sum(OA)/10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Prueba No.3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test = 'pcaBSCAE_v2';         %  pcaSCAE_v2  SCAE_v2  pcaBSCAE_v2 BSCAE_v2

path = strcat('../../6_data Logger/',test,'/',dataset,'/logger_',dataset,'_TEST.txt');
data = load(path);
OA = zeros(1,10); j=1;
for i=1:3:30
    OA(j) = data(i,1);
    j=j+1;
end
OA_LR3 = sum(OA)/10;

path = strcat('../../6_data Logger/',test,'/',dataset,'/logger_',dataset,'_RIEM.txt');
data = load(path);
OA = zeros(1,10); j=1;
for i=1:3:30
    OA(j) = data(i,1);
    j=j+1;
end
OA_RC3 = sum(OA)/10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Prueba No.3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
test = 'BSCAE_v2';         %  pcaSCAE_v2  SCAE_v2  pcaBSCAE_v2 BSCAE_v2

path = strcat('../../6_data Logger/',test,'/',dataset,'/logger_',dataset,'_TEST.txt');
data = load(path);
OA = zeros(1,10); j=1;
for i=1:3:30
    OA(j) = data(i,1);
    j=j+1;
end
OA_LR4 = sum(OA)/10;

path = strcat('../../6_data Logger/',test,'/',dataset,'/logger_',dataset,'_RIEM.txt');
data = load(path);
OA = zeros(1,10); j=1;
for i=1:3:30
    OA(j) = data(i,1);
    j=j+1;
end
OA_RC4 = sum(OA)/10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%% Graficar %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
output = [OA_LR1 OA_RC1 ; OA_LR2 OA_RC2 ; OA_LR3 OA_RC3 ; OA_LR4 OA_RC4];
c = categorical({'pcaSCAE','eepSCAE','pcaBCAE', 'eepBCAE'});
bar(c, output)
ylabel('Accuracy'), grid on 


