%WTF => WAVELET ANALISYS FOR SPECTRAL DIMENSION REDUCTION. 
clear, clc, close all;
level=5;
waveletMadre='bior6.8';
sData=load('../../paviaU.mat');
dataCube=(10^-4).*sData.paviaU;
imgSize=size(dataCube);

sData=load('../../paviaU_gt.mat');
dataGround=sData.paviaU_gt;
imgSizeGT=size(dataGround);

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

imgSizeDWT=size(cA);

x=zeros(imgSizeDWT(3),imgSizeDWT(1)*imgSizeDWT(2)); %Matriz Entrada RNA 
k=1;
for i=1:imgSizeDWT(1)
    for j=1:imgSizeDWT(2)
        if dataGround(i,j)~=0
            tempvect=cA(i,j,:);
            x(:,k)=tempvect(:);
            k=k+1;        
        end
    end
end
x=x(:,1:k-1);

save('datosDWT','x');
disp('PROCESS DONE !!!')