clear, clc, close all;

imgTh=load('indian_pines_gt.mat');
imgTh=imgTh.indian_pines_gt; figure;
imagesc(imgTh);    %IMAGEN DE CLASIFICACION DESEADA
title('Imagen Deseada'); axis off;
imgSize=size(imgTh);

imgOutSOM=load('out_rbfSOM');
imgOutSOM=imgOutSOM.imgOutSOM;
figure;
imagesc(imgOutSOM); %IMAGEN SOM
title('Imagen SOM'); axis off;


imgOut=zeros(imgSize(1),imgSize(2),17);
for i=1:imgSize(1)
    for j=1:imgSize(2)
         
        value=imgOutSOM(i,j);
        switch value
            case 1
                imgOut(i,j,1)=value;
            case 2
                imgOut(i,j,2)=value;
            case 3
                imgOut(i,j,3)=value;
            case 4
                imgOut(i,j,4)=value;
            case 5
                imgOut(i,j,5)=value;
            case 6
                imgOut(i,j,6)=value;
            case 7
                imgOut(i,j,7)=value;
            case 8
                imgOut(i,j,8)=value;
            case 9
                imgOut(i,j,9)=value;
            case 10
                imgOut(i,j,10)=value;
            case 11 
                imgOut(i,j,11)=value;
            case 12
                imgOut(i,j,12)=value;
            case 13
                imgOut(i,j,13)=value;
            case 14
                imgOut(i,j,14)=value;
            case 15
                imgOut(i,j,15)=value;
            case 16
                imgOut(i,j,16)=value;
            otherwise
        end
             
     end
end



for k=1:16
se = strel('disk',3);
imgOut(:,:,k)=imclose(imgOut(:,:,k),se);
end


imgOut2=zeros(imgSize(1),imgSize(2));
for k=1:16
imgOut2=imgOut2+imgOut(:,:,k);
end
figure;
imagesc(imgOut2);

mdl = fitlm(imgTh(:),imgOut2(:));
crSOM=mdl.Rsquared.Ordinary;
disp('Correlación Datos SOM');
disp(crSOM);

