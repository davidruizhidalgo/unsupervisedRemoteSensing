%Confusion Matrix
clear, clc, close all;

load('imgOutSOM.mat');
imgSize=size(imgOutSOM);
figure;
imagesc(imgOutSOM);

imgTh=load('../../../dataSets/Indian_pines_gt.mat');
imgTh=imgTh.indian_pines_gt;
figure;
imagesc(imgTh);

g1=imgOutSOM(:);
g2=imgTh(:);
figure;
C = confusionmat(g1,g2);
% C(1,:)=[]; C(:,1)=[];
% cm=confusionchart(C);
plotConfMat(C) 