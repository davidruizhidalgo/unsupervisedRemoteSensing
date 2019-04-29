function [R490,R550,R680,R720,R800,R900] = LoadSpectralData()
%Cargar Informacion Multiespectral 
%R490 -> 490nm banda de azul
%R550 -> 550nm banda de verde
%R680 -> 680nm banda de rojo
%R720 -> 720nm banda de frontera del rojo
%R800 -> 800nm banda de infrarojo cercano
%R900 -> 900nm banda de infrarojo cercano
R490=ReadRawImage(strcat(pwd,'\Images\','TTC00083.RAW'));
R550=ReadRawImage(strcat(pwd,'\Images\','TTC10083.RAW'));
R680=ReadRawImage(strcat(pwd,'\Images\','TTC20083.RAW'));
R720=ReadRawImage(strcat(pwd,'\Images\','TTC30083.RAW'));
R800=ReadRawImage(strcat(pwd,'\Images\','TTC40083.RAW'));
R900=ReadRawImage(strcat(pwd,'\Images\','TTC50083.RAW'));

