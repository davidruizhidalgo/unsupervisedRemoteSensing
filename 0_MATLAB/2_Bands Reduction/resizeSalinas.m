%REMUESTREO IMAGEN SALINAS
clear, clc, close all;

sData=load('C:\Users\Ruiz Desk\Downloads\Datos Investigacion\Aviris Imagery Set\salinas_scene_data\Salinas_corrected.mat');
dataCube=sData.salinas_corrected;
imgSize=size(dataCube);

salinas_corrected=zeros(imgSize(1)/2,floor(imgSize(2)/2),imgSize(3));
f=1;c=1;
for i=1:imgSize(1)/2
    for j=1:floor(imgSize(2)/2)
        salinas_corrected(i,j,:)=dataCube(f,c,:);
        c=c+2;
    end
    c=1;
    f=f+2;
end

save('Salinas_corrected.mat','salinas_corrected');


sData=load('C:\Users\Ruiz Desk\Downloads\Datos Investigacion\Aviris Imagery Set\salinas_scene_data\Salinas_gt.mat');
dataCube=sData.salinas_gt;
imgSize=size(dataCube);

salinas_gt=zeros(imgSize(1)/2,floor(imgSize(2)/2));
f=1;c=1;
for i=1:imgSize(1)/2
    for j=1:floor(imgSize(2)/2)
        salinas_gt(i,j,:)=dataCube(f,c);
        c=c+2;
    end
    c=1;
    f=f+2;
end

save('Salinas_gt.mat','salinas_gt');

imagesc(salinas_gt);






