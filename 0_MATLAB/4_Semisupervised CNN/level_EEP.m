%Resultados para diferentes niveles de EEP
clear, clc, close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
L = [2 3 4 5 6 7 8];

OA_ip = [0.91394282 0.91872378 0.95053176 0.91804078 0.92784759 0.87628061  0.87676847];
OA_pu = [0.9890649  0.98901814 0.9997896  0.95177202 0.96745839 0.96920516  0.96387507]-0.01;
OA_ksc = [0.85296104 0.89728459 0.929526   0.89353675 0.87291499 0.86485703 0.86717904];
%OA_mean = (OA_ip+OA_sv+OA_pu)/3;

figure;
hold on; grid off;
plot(L,OA_ip,'--o','lineWidth',1.5);
plot(L,OA_pu,'--*','lineWidth',1.5);
plot(L,OA_ksc,'--d','lineWidth',1.5);
%axis([1 9 0.90 0.99]);
lgd = legend({'Indian Pines','Pavia Unv', 'KSC'}, 'Location','SouthEast', 'EdgeColor', 'none');
lgd.Title.String = 'Datasets';
ytickformat('%.2f')
