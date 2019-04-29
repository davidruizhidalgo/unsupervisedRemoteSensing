function data = ReadRawImage(filename,plotimg)
% data = ReadRawImage('filename.RAW','GRAFICAR') para visualisar Img OUT
% data = ReadRawImage('filename.RAW') para no visualizar Img OUT
% Cargar Archivos de Imagenes .RAW 
% Camara Multiespectral TETRACAM MICRO-MCA6
% Bytes 0-3: Tamaño Total de la Imagen
% Byte 4:    Bits por Pixel
% Byte 5:    Etiqueta del Formato .RAW
% Bytes 6-7: Numero de Columnas de Pixeles
% Bytes 8-9: Numero de Filas de Pixeles
% Bytes 10-Tamaño_Img: Datos de los Pixeles 16 bits
% Bytes Tamaño_Img-(EOF-28): Datos GPS Cadenas $GGA $RMC
% Bytes (EOF-28)-EOF(end of file): 'EXPOSURE:%0.8ld seconds\n'
fin=fopen(filename,'r');                    %Abrir el Archivo .RAW
numelData=fread(fin, 1,'uint32');           %Tamaño Total de la Imagen 
bitsData=fread(fin, 1,'uint8');             %Numero de Bits por Pixel
format=fread(fin, 1,'uint8');               %Formato de la Imagen
col=fread(fin, 1,'uint16');                 %Numero de Columnas de la Img
row=fread(fin, 1,'uint16');                 %Numero de Filas de la Img
data=(fread(fin, [col row],'uint16'))';     %Datos de la Imagen 
%Bytes (tamaño de la imagen + 10)-(EOF - 28) datos del GPS. Cadenas $GGA y $RMC
%Últimos 28 bytes – cadena de exposición ASCII formateada: "EXPOSURE:%08ld uSeconds\n"
fclose(fin);                                %Cerrar el Archivo .RAW
if nargin==1
    
else
    dataToView=uint8((255/1024)*data);      %Ajustar el Rango de Salida
    figure; imshow(dataToView);             %Mostrar Imagen   
    clc; plotimg=strcat(plotimg,' IMAGEN ORIGINAL'); disp(plotimg);
end