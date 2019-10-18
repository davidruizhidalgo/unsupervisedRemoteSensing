%Resultados para diferentes niveles de EEP
clear, clc, close all;
%%%%%%%%%%%%%%%%%%%%%%%%AUTOENCODER CONVOLUCIONAL APILADO%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
L = [3 4 5 6 7 8 9];

OA_ip = [0.8507  0.8473	 0.8470   0.8350   0.8368     0.8342  0.8008] ;
OA_sv = [0.6412  0.6749  0.6621   0.6437   0.6582     0.6696  0.6901];
OA_pu = [0.9270  0.9503  0.9142   0.9180   0.8607     0.8371  0.7976];
OA_mean = (OA_ip+OA_sv+OA_pu)/3;

figure;
hold on; grid on;
plot(L,OA_ip,'--o','lineWidth',0.5);
plot(L,OA_sv,'--d','lineWidth',0.5);
plot(L,OA_pu,'--*','lineWidth',0.5);
plot(L,OA_mean,'--+','lineWidth',2.5);
%axis([1 16 0.86 1.0]);
legend('Indian Pines','Salinas', 'Pavia Unv','Mean', 'Location','northEast');
