%Entropia Analisys. 
clear, clc, close all;
                             
spectralBands=load('spectralBands.mat'); %spectralBands.wavelength Vector de Bandas Espectrales
sData=load('../indian_pines_corrected_0.mat');
dataCube=sData.dataCube;
imgSize=size(dataCube);                     %LOAD HYPERESPECTRAL DATA

EntropyArrR2=zeros(1,imgSize(3));
bp=waitbar(0,'CALCULANDO ENTROPIA ....'); 
for i=1:imgSize(3)
     band_one=dataCube(:,:,i);
     EntropyArrR2(i)=entropy(band_one); 
     waitbar(i/imgSize(3),bp); 
end

close(bp);

%Cargar Imagen Original
rgbImg=dataCube(:,:,1);
figure;
imagesc(rgbImg); axis off;  %Mostrar Imagen
%Grafica de la Varianza
figure;
plot(spectralBands.wavelength,EntropyArrR2); grid on;
% xlim([spectralBands.wavelength(1) spectralBands.wavelength(end)]);
xlabel('Longitud de Onda (nm)'); ylabel('Entropía de Banda');

