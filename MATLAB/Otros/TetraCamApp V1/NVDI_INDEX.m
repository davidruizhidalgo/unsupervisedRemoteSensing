%NVDI INDEX NORMALIZED VEGETATION DIFFERENCE INDEX
%NVDI= (R800-R680)/(R800+R680)
clear all; clc;
[R490,R550,R680,R720,R800,R900]=LoadSpectralData();     %Cargar 6 Bandas Espectrales
NVDI=(R800-R680)./(R800+R680);                          %Calcular NVDI INDEX


ViewImage(R490,'GRAFICAR');        %Grafica Imagen Original Banda 490nm
figure;imagesc(NVDI);              %Grafica NVDI PseudoColor
axis off; colorbar;