function dataToView = ViewImage(data,plotimg)
% AJUSTAR DATOS PARA GRAFICAR IMAGEN .RAW 
% data = ViewImage(dataIn,'GRAFICAR') para visualisar Img OUT
% data = ViewImage(dataIn) para no visualizar Img OUT
dataToView=uint8((255/1024)*data);      %Ajustar el Rango de Salida
if nargin==1
    
else
    figure; imshow(dataToView);             %Mostrar Imagen   
    clc; plotimg=strcat(plotimg,' IMAGEN ORIGINAL'); disp(plotimg);
end
