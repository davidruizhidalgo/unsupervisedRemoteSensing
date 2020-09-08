%GRaficar Cubo Hiperespectral . 
clear, clc, close all;
sData=load('../../../dataSets/PaviaU.mat');
dataCube=sData.paviaU;
imgSize=size(dataCube);

for i=1:imgSize(1)
    for j=1:imgSize(2)
        tempvec=dataCube(i,j,:);
        dataCube(i,j,:)=tempvec(:);
    end
end

% for i=1:imgSize(1)
%     for j=1:imgSize(2)
%         for k=1:imgSize(3)
%             if dataCube(i,j,k) > 1000
%                 dataCube(i,j,k) = 702;
%             end
%         end
%     end
% end
dataCube(:,:,imgSize(3))=dataCube(:,:,20);

D =dataCube;
D(D==0)=nan;
h = slice(D, [], [], 1:size(D,3));
set(h, 'EdgeColor','none', 'FaceColor','interp')
alpha(.4)
colormap jet
axis image
axis ij


% Get axis handle
axh = h(1).Parent;    %alternative: axh = gca; 
xlabel(axh, 'x pixel')
ylabel(axh, 'y pixel')
zlabel(axh, 'Spectral Bands')
xlim(axh,[1 imgSize(2)])
ylim(axh,[1 imgSize(1)])
zlim(axh,[1 imgSize(3)])
