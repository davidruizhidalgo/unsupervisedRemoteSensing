%TRAINING SUPPORT VECTOR MACHINE SVM 17 Class of vegetation

clear, clc, close all;
trainData=load('trainingDataSVM.mat');
subtotalClass=17;
class=1:17;
x=[trainData.dataClass{1},trainData.dataClass{2},trainData.dataClass{3}...
    trainData.dataClass{4},trainData.dataClass{5},trainData.dataClass{6},trainData.dataClass{7}...
    trainData.dataClass{8},trainData.dataClass{9},trainData.dataClass{10},trainData.dataClass{11}...
    trainData.dataClass{12},trainData.dataClass{13},trainData.dataClass{14},trainData.dataClass{15}...
    trainData.dataClass{16},trainData.dataClass{17}];


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
    'ClassNames',[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 ,16 ,17]);


save('svmData','svmclass');
disp('Training SVM: DONE !!!')
h=msgbox('Training SVM: DONE','Success','warn');