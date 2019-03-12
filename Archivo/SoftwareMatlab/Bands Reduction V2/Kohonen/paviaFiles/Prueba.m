clear, clc, close all;

imgTh=load('../../paviaU_gt.mat');
imgTh=imgTh.paviaU_gt;
imgSize=size(imgTh);
figure
imagesc(imgTh)
title('Ground Truth Original'); axis off;


imgOut=zeros(imgSize(1),imgSize(2));
for i=1:imgSize(1)
    for j=1:imgSize(2)
        if imgTh(i,j)~=0 && imgTh(i,j)~=3 && imgTh(i,j)~=7  
            imgOut(i,j)=imgTh(i,j);
        end
    end
end
figure
imagesc(imgOut)
title('Ground Truth'); axis off;
