%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%DIVISION OF IMAGES TO GET A TEST BED TO TRY A MOSAICING PROCESS%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; clc;
[imgPrub,map]=imread('ImgPrueba.jpg'); % Image read
orgDim=size(imgPrub);% Original image dimentions
longDes=100;% Desired longitude of sub-images in pixels
superPst=0.4;% Percentage of sub-images superposition desired
percentage=(1-superPst)*longDes;% Desired superposition

offset_y=1;
wb=waitbar(0,'Please Wait...');
for i=1:((orgDim(1)-longDes)/percentage)+1
    offset_x=1; 
    waitbar(i,wb);
    for j=1:((orgDim(2)-longDes)/percentage)+1
    newImg=imgPrub(offset_y:(offset_y-1)+longDes,...
        offset_x:(offset_x-1)+longDes,1:orgDim(end));% Division of images
    imwrite(newImg,strcat(int2str(i),'-',int2str(j),'.jpg'));
    offset_x=j*percentage;
    end
    offset_y=i*percentage;
end

h=msgbox('Operation Completed',...
         'Success','custom',imgPrub,map);


