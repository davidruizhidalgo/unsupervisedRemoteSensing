%Lambda vs Lambda Analisys. 
clear, clc, close all;
                             
spectralBands=load('spectralBands.mat'); %spectralBands.wavelength Vector de Bandas Espectrales
sData=load('../indian_pines_corrected_0.mat');
dataCube=(10^3).*sData.dataCube;
imgSize=size(dataCube);                     %LOAD HYPERESPECTRAL DATA


CoefMatR2=eye(imgSize(3));
bp=waitbar(0,'CALCULANDO COEFICIENTES R2...'); 
for i=1:imgSize(3)
     band_one=dataCube(:,:,i);
     band_one=band_one(:);
    for j=1:imgSize(3)
        if j>i
            %Coeficiente de Determinación R2
            band_two=dataCube(:,:,j);
            band_two=band_two(:);
            mdl = fitlm(band_one,band_two);
            CoefMatR2(i,j)=mdl.Rsquared.Ordinary;   
            CoefMatR2(j,i)=mdl.Rsquared.Ordinary; 
        end
    end
    waitbar(i/imgSize(3),bp); 
end

CoefMatR2=flip(CoefMatR2);
close(bp);

save('lambdaGraf','CoefMatR2');

%Cargar Imagen Original
rgbImg=dataCube(:,:,1);
figure;
imagesc(rgbImg); axis off;  %Mostrar Imagen
%Grafica Coeficientes R2
figure;
imagesc(CoefMatR2); axis off;
colorbar;



