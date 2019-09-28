function sData=spatialwindow(data,mask)
dataSize=size(data);
sData=zeros(dataSize(1),dataSize(2),dataSize(3)*mask*mask);

%Generacion Matriz Ampliada 
w=(mask-1)/2;
tempData=zeros(dataSize(1)+2*w,dataSize(2)+2*w,dataSize(3));
tempData(w+1:dataSize(1)+w,w+1:dataSize(2)+w,:)=data;
tempData(:,1:w,:)=tempData(:,w+1:2*w,:);
tempData(:,end-w+1:end,:)=tempData(:,dataSize(2)+1:dataSize(2)+w,:);
tempData(1:w,:,:)=tempData(w+1:2*w,:,:);
tempData(end-w+1:end,:,:)=tempData(dataSize(1)+1:dataSize(1)+w,:,:);

%Nuevo conjunto de datos con MASK
for i=1:dataSize(1)
    for j=1:dataSize(2)
        xi=i;xf=i+2*w;yi=j;yf=j+2*w; % ORGANIZAR ESTOS INDICES
        tempVect=tempData(xi:xf,yi:yf,:);
        sData(i,j,:)=tempVect(:);
    end
end

end