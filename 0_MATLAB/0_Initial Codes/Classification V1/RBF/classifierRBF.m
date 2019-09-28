%RBF CLASSIFIER netRBF

%LOAD ORIGINAL IMAGE DATA
sData=load('../../datosSOM_indianPines.mat');
dataCube=sData.dataSOM;
% sData=load('../../indian_pines_corrected_0.mat');
% dataCube=sData.dataCube;
imgSize=size(dataCube);

x=zeros(imgSize(3),imgSize(1)*imgSize(2)); %Input Matrix to RNA 

k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
        tempvect=dataCube(i,j,:);
        x(:,k)=tempvect(:);
        k=k+1;
    end
end


net=load('netRBF');
net=net.net;
y=net(x);   %Matriz de Salida RNA

%DECODIFICATION OF OUTPUT DATA
imgOutRBF=zeros(imgSize(1),imgSize(2));

k=1;
for i=1:imgSize(1)
    for j=1:imgSize(2)
      [valy,indexy]=sort(y(:,k));
      imgOutRBF(i,j)=indexy(end)-1; %VEGETATION CLASS
      k=k+1;
    end
end

imgTh=load('../../indian_pines_gt.mat');
imgTh=imgTh.indian_pines_gt; figure;
imagesc(imgTh);    %GROUND TRUTH IMAGES
title('Ground Truth'); axis off;

figure;
imagesc(imgOutRBF);   %OUTPUT IMAGE WITHOUT FILTERING
title('Output Image RBF'); axis off;




mdl = fitlm(imgTh(:),imgOutRBF(:));
crRBF=mdl.Rsquared.Ordinary;
disp('Correlation RBF Data');
disp(crRBF);
save('imgOutRBF','imgOutRBF');

disp('CLASSIFICATION DONE');
h=msgbox('CLASSIFICATION DONE','Success','warn');
