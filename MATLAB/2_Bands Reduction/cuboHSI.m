%GRaficar Cubo Hiperespectral . 
clear, clc, close all;
sData=load('Indian_pines_corrected.mat');
dataCube=sData.indian_pines_corrected;
imgSize=size(dataCube);

% for i=1:imgSize(1)
%     for j=1:imgSize(2)
%         tempvec=dataCube(i,j,:);
%         dataCube(i,j,:)=flip(tempvec(:));
%     end
% end

dataCube(:,:,200)=dataCube(:,:,30);

D = dataCube;
D(D==0)=nan;
h = slice(D, [], [], 1:size(D,3));
set(h, 'EdgeColor','none', 'FaceColor','interp')
alpha(.5)
colormap jet
axis image
axis ij