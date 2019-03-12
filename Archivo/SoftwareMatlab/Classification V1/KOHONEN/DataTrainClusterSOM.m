%DATOS ENTRENAMIENTO CON KOHONEN
clear, clc, close all;
porData=1;                                      %PERCENTAGE OF TRAINING DATA
totalClass=17;
%LOAD PCA or ICA DATA
sData=load('../PCA_ICA/PCA.mat');
dataCube=sData.dataPCA;                             %DATA PCA
% sData=load('../PCA_ICA/ICA.mat');
% dataCube=sData.dataICA;                           %DATA ICA
imgSize=size(dataCube);

%LOAD GROUND TRUTH DATA
sData=load('../indian_pines_gt.mat');
dataGround=sData.indian_pines_gt;                   % INDIAN PINES GROUND TRUTH
% sData=load('../Salinas_gt.mat');
% dataGround=sData.salinas_gt;                      % SALINAS VALLEY GROUND TRUTH

%TAKING FROM GROUND TRUTH ALL CLASS ELEMENTS    
    %CREATING CLASS VECTORS
    idxclass=cell(totalClass,1);

    for i=1:imgSize(1)
        for j=1:imgSize(2)
        val=dataGround(i,j);
            idxclass{val+1}(end+1,:)=[i,j,val]; %Class Vegetation Index
        end
    end
            
%TAKING A PERCENTAGE OF DATA TO TRAINING
dataClass=cell(totalClass,1);
for i=1:totalClass
    dataClass{i}=zeros(imgSize(3),ceil(porData*size(idxclass{i},1))); %Creating Vectors
end
%Taking Data
trainImg=zeros(imgSize(1),imgSize(2));
for c=1:totalClass
    dataToTrain=1:size(idxclass{c},1);
    dataToTrain=dataToTrain(randperm(length(dataToTrain))); % Mixing class elements
    for i=1:size(dataClass{c},2)
        k=dataToTrain(i);
        pxData=dataCube(idxclass{c}(k,1),idxclass{c}(k,2),:);
        dataClass{c}(:,i)= pxData(:); % Data Class c
        trainImg(idxclass{c}(k,1),idxclass{c}(k,2))=idxclass{c}(k,3);
    end
end

figure;
imagesc(dataGround);

figure;
imagesc(trainImg);


save('idxDataCube','idxclass');
save('trainingDataSOM','dataClass');

disp('Get training data: DONE!!!');
h=msgbox('Get training data: DONE','Success','warn');
