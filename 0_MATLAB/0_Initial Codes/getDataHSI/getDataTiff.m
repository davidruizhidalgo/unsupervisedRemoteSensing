%get datatiff
clc;
clear all;

imageOut = imread('DC Mall/DC.tif');
sizeCol = size(imageOut,1);
sizeRow = size(imageOut,2);
figure(1)
imshow(imageOut(:,:,60))

[image_gt, map] = imread('DC Mall/GT.tif');
groundTruth = ind2rgb(image_gt, map);
figure(2)
imshow(groundTruth)

% index =1;
% values = zeros(1,size(map,1));
% for i=1:sizeRow
%     for j=1:sizeRow
%         values(image_gt(i,j)+1) = values(image_gt(i,j)+1)+1;
%     end
% end
% 
% save('DC.mat','imagenOut');
% save('DC_gt.mat','imagenOut_gt')


