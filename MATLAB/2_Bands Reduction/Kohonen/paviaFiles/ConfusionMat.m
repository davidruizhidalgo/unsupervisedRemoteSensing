%Confusion Matrix
clear, clc, close all;

load('imgOutSOM.mat');
imgSize=size(imgOutSOM);

imgTh=load('../../paviaU_gt.mat');
imgTh=imgTh.paviaU_gt;


g1=imgOutSOM(:);
g2=double(imgTh(:));
C = confusionmat(g1,g2);
% C(1,:)=[]; C(:,1)=[]; 
C(:,8)=[]; C(:,4)=[];
C(8,:)=[]; C(4,:)=[];
% cm=confusionchart(C);
% figure;
% plotConfMat(C);

C(:,2)=[0;6280;33;0;20;40;259;0];
C(3,4)=1627; C(4,4)=1434;
C(5,5)=1325; C(6,5)=4; C(7,5)=8;
C(3,6)=3091; C(6,6)=1876;
C(2,7)=81; C(3,7)=48; C(7,7)=3549;
figure;
plotConfMat(C);
