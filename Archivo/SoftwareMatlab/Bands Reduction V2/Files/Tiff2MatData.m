% TIFF IMAGE 2 MATLAB DATA SPECTRAL CUBE 

clear, clc, close all;
[File,Path]=uigetfile({'*.*'},'Cargar Imagen');
filename=strcat(Path,File);
%sData=load(filename);% Matlab Files
%dataCube=eval(strcat('sData.',File(1:end-4)));
dataCube=imread(filename); %TIFF Images
dataCube = double(dataCube) ./ 10e3; % Normalize to proper reflectance units.
imgSize=size(dataCube);

ypos=0;
width=[1 imgSize(2)];
height=[1 imgSize(1)];
File=File(1:end-4);

saveFileName=strcat(File,'_',num2str(ypos));
save(saveFileName,'File','ypos','width','height','dataCube','-v7.3');