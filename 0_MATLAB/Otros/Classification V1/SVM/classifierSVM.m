%SUPPORT VECTOR MACHINE CLASSIFIER SVM
clear, clc, close all;
%LOAD REDUCTED IMAGE DATA
% sData=load('../PCA_ICA/PCA.mat');
% dataCube=sData.dataPCA;                               %DATA PCA
sData=load('../PCA_ICA/ICA.mat');
dataCube=sData.dataICA;                                 %DATA ICA
imgSize=size(dataCube);

%LOAD CLASS DATA INDICES
idxData=load('idxDataCube.mat');
idxData=idxData.idxclass;
subtotalClass=17;
class=1:17;

totalSize=0;
for i=1:subtotalClass
    totalSize=totalSize+size(idxData{class(i)},1);
end

%DATA INPUT FORMAT FROM DATA CUBE
k=1;
x=zeros(totalSize,imgSize(3));
for i=1:subtotalClass
    for j=1:size(idxData{class(i)},1)
        tempvect=dataCube(idxData{class(i)}(j,1),idxData{class(i)}(j,2),:);
        x(k,:)=tempvect(:);
        k=k+1;
    end
end


svmData=load('svmData');
imgOutSVM=zeros(imgSize(1),imgSize(2),size(svmData.svmclass,1));
scoresOutSVM=zeros(imgSize(1),imgSize(2),size(svmData.svmclass,1));


for c=1:size(svmData.svmclass,1)
    [Ypredict,scores] = predict(svmData.svmclass,x);
%     [Ypredict,scores] = predict(svmData.svmclass{c},x);
    k=1;
    for j=1:subtotalClass
        for i=1:size(idxData{class(j)},1)
            imgOutSVM(idxData{class(j)}(i,1),idxData{class(j)}(i,2),c)=Ypredict(k); %VEGETATION CLASS
            k=k+1;
        end
    end
end
 


% imgOutSVM=imgOutSVM>0;
% imgOut_tmp=zeros(size(imgOutSVM,1),size(imgOutSVM,2));
% for i=1:size(imgOutSVM,3)
%   imgOut_tmp=imgOut_tmp+(i*imgOutSVM(:,:,i));  
% end
% 
% imgOutSVM=imgOut_tmp-1;


%PLOTING OUTPUT IMAGES
imgTh=zeros(size(imgOutSVM,1),size(imgOutSVM,2));
for c=1:subtotalClass
    for i=1:size(idxData{class(c)},1)
       imgTh(idxData{class(c)}(i,1),idxData{class(c)}(i,2))=idxData{class(c)}(i,3)+1;%CLASS SCORES
    end
end
figure;
imagesc(imgTh);    %GROUND TRUTH IMAGES
title('Ground Truth'); axis off;

figure;
imagesc(imgOutSVM);   %OUTPUT IMAGE WITHOUT FILTERING
title('Output Image SVM'); axis off;


mdl = fitlm(imgTh(:),imgOutSVM(:));
crSVM=mdl.Rsquared.Ordinary;
disp('Correlation SVM Data');
disp(crSVM);
save('imgOutSVM','imgOutSVM');

disp('CLASSIFICATION DONE');
h=msgbox('CLASSIFICATION DONE','Success','warn');
 