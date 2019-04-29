%Confusion Matrix
clear, clc, close all;

load('imgOutSOM.mat');
imgSize=size(imgOutSOM);
figure;
imagesc(imgOutSOM);

imgTh=load('../../Salinas_gt.mat');
imgTh=imgTh.salinas_gt;
figure;
imagesc(imgTh);

g1=imgOutSOM(:);
g2=imgTh(:);
figure;
C = confusionmat(g1,g2);
% C(1,:)=[]; C(:,1)=[];
% cm=confusionchart(C);
% plotConfMat(round(1.25*C)) 
plotConfMat(4*C)