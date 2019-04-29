%TRAINING SUPPORT VECTOR MACHINE SVM 9 Class of vegetation
clear, clc, close all;
trainData=load('trainingDataSVM.mat');
subtotalClass=9;
class=[3 4 6 7 9 11 12 13 15];
%INPUT DATA MATRIX
x=[trainData.dataClass{3},trainData.dataClass{4},...
    trainData.dataClass{6},trainData.dataClass{7},...
    trainData.dataClass{9},trainData.dataClass{11},...
    trainData.dataClass{12},trainData.dataClass{13},trainData.dataClass{15}]; %9 Class of vegetation
x=x';

%OUTPUT DATA MATRIX
sizeEnd=0;
y=zeros(1,size(x,1)); % dimension of input vector
for i=1:subtotalClass
    sizeIni=sizeEnd+1; sizeEnd=sizeIni+size(trainData.dataClass{class(i)},2)-1;
    y(sizeIni:sizeEnd)=class(i); 
end

y=y';%Put in true class elements 
t = templateSVM('Standardize',1,'KernelFunction','gaussian');
svmclass = fitcecoc(x,y,'Learners',t,'FitPosterior',1,...
    'ClassNames',[3, 4, 6, 7, 9, 11, 12, 13, 15]);


save('svmData_2','svmclass');
disp('Training SVM: DONE !!!')
h=msgbox('Training SVM: DONE','Success','warn');
