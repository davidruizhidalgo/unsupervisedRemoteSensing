%WTF => WAVELET ANALISYS FOR SPECTRAL DIMENSION REDUCTION. 
clear, clc, close all;
level=2;
waveletMadre='bior6.8';
sData=load('../../indian_pines_corrected.mat');
dataCube=(10^-3).*sData.indian_pines_corrected;
imgSize=size(dataCube);

dwtmode('per');
tempvect=dataCube(1,1,:);
[c,l]=wavedec(tempvect(:),level,waveletMadre);

cA=zeros(imgSize(1),imgSize(2),l(1));
cD=zeros(imgSize(1),imgSize(2),l(2));


for i=1:imgSize(1)
    for j=1:imgSize(2)
        tempvect=dataCube(i,j,:);
        %[cA(i,j,:),cD(i,j,:)]=dwt(tempvect(:),waveletMadre,'mode','per');
        [c,l]=wavedec(tempvect(:)',level,waveletMadre);
        cA(i,j,:)=c(1:l(1));
        cD(i,j,:)=c(l(1)+1:l(2)+l(1));
    end
   
end

save('datosDWT','cA','cD');
disp('PROCESS DONE !!!')