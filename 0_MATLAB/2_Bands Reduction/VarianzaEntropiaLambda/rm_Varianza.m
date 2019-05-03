%Variance Analisys. 
clear, clc, close all;
                             
spectralBands=load('spectralBands.mat'); %spectralBands.wavelength Vector de Bandas Espectrales
sData=load('../indian_pines_corrected_0.mat');
dataCube=(10^3).*sData.dataCube;
imgSize=size(dataCube);                     %LOAD HYPERESPECTRAL DATA

VarArrR2=zeros(1,imgSize(3));
bp=waitbar(0,'CALCULANDO VARIANZA ....'); 
for i=1:imgSize(3)
     band_one=dataCube(:,:,i);
     band_one=band_one(:);
     VarArrR2(i)=var(band_one,'omitnan');
    
    waitbar(i/imgSize(3),bp); 
end

close(bp);

%Cargar Imagen Original
rgbImg=dataCube(:,:,1);
figure;
imagesc(rgbImg); axis off;  %Mostrar Imagen
%Grafica de la Varianza
figure;
plot(spectralBands.wavelength,VarArrR2/1000); grid on;
% xlim([spectralBands.wavelength(1) spectralBands.wavelength(end)]);
xlabel('Longitud de Onda (nm)'); ylabel('Varianza de Banda (10^3)');
