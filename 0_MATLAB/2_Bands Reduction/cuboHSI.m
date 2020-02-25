%GRaficar Cubo Hiperespectral . 
clear, clc, close all;
sData=load('KSC.mat');
dataCube=sData.KSC;
imgSize=size(dataCube);

% for i=1:imgSize(1)
%     for j=1:imgSize(2)
%         tempvec=dataCube(i,j,:);
%         dataCube(i,j,:)=flip(tempvec(:));
%     end
% end

for i=1:imgSize(1)
    for j=1:imgSize(2)
        for k=1:imgSize(3)
            if dataCube(i,j,k) > 1000
                dataCube(i,j,k) = 702;
            end
        end
    end
end

dataCube(:,:,176)=dataCube(:,:,30);

% D = zeros(size(dataCube,1),size(dataCube,2),176);
% D(:,:,1:20) = dataCube(:,:,11:30);
% D(:,:,21:40) = dataCube(:,:,11:30);
% D(:,:,41:60) = dataCube(:,:,11:30);
% D(:,:,61:80) = dataCube(:,:,11:30);
% D(:,:,81:100) = dataCube(:,:,11:30);
% D(:,:,101:120) = dataCube(:,:,15:34);
D =dataCube;
D(D==0)=nan;
h = slice(D, [], [], 1:size(D,3));
set(h, 'EdgeColor','none', 'FaceColor','interp')
alpha(.2)
colormap jet
axis image
axis ij