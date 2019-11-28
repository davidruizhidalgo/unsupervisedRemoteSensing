%programas de prueba para levantamiento de imagenes HSI
clc;
clear all;
%data = load('Urban/Urban_F210.mat');
% data = load('samson/samson_1.mat');
data = load('jasper/jasperRidge2_F224_2.mat'); 
sizeCol = data.nCol;
sizeRow = data.nRow;
imagen = data.Y;

%IMAGE DATA
index =1;
imagenOut = zeros(sizeRow,sizeCol,size(imagen,1));
for i=1:sizeRow
    for j=1:sizeRow
        imagenOut(j,i,:) = imagen(:,index);
        index = index +1;
    end
end
figure(1)
imagesc(imagenOut(:,:,50))

%IMAGE GROUNDTHRUTH
index =1;
% groundTruth = load('Urban/end6_groundTruth.mat'); 
% groundTruth = load('samson/end3.mat'); 
groundTruth = load('jasper/end4.mat');
groundTruth = groundTruth.A;
[value, groundTruth] = max(groundTruth);
imagenOut_gt = zeros(sizeRow,sizeCol);
for i=1:sizeRow
    for j=1:sizeRow
        imagenOut_gt(j,i) = groundTruth(index);
        index = index +1;
    end
end
figure(2)
imagesc(imagenOut_gt)

% save('Urban210.mat','imagenOut');
% save('Urban210_gt.mat','imagenOut_gt')
% save('samson.mat','imagenOut');
% save('samson_gt.mat','imagenOut_gt')
save('Jasper.mat','imagenOut');
save('Jasper_gt.mat','imagenOut_gt')
