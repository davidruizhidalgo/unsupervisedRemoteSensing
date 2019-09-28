%%%%%%%%%%%%%%%%%%%%%%CARGAR IMAGEN .RAW %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc;
[File Path]=uigetfile({'*.raw'},'Abrir Imagen');
filename=strcat(Path,File);
fin=fopen(filename,'r');                    %Abrir el Archivo .RAW
numelData=fread(fin, 1,'uint32');           %Tama�o Total de la Imagen 
bitsData=fread(fin, 1,'uint8');             %Numero de Bits por Pixel
format=fread(fin, 1,'uint8');               %Formato de la Imagen
col=fread(fin, 1,'uint16');                 %Numero de Columnas de la Img
row=fread(fin, 1,'uint16');                 %Numero de Filas de la Img
data=(fread(fin, [col row],'uint16'))';     %Datos de la Imagen 
%Bytes (tama�o de la imagen + 10)-(EOF - 28) datos del GPS. Cadenas $GGA y $RMC
%�ltimos 28 bytes � cadena de exposici�n ASCII formateada: "EXPOSURE:%08ld uSeconds\n"
fclose(fin);                                %Cerrar el Archivo .RAW
dataToView=uint8((255/1024)*data);          %Ajustar el Rango de Salida
imshow(dataToView);                         %Mostrar Imagen
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%