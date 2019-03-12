clear, clc, close all;

Nneuronas=[10 10];
datosauxiris = load('-ascii','iris_data2.txt');
datosiris = datosauxiris(:,1:4)';
%red = competlayer(3);
red = selforgmap(Nneuronas);
red.trainParam.epochs=1500;
red=train(red,datosiris);
yred=sim(red, datosiris);
indices=vec2ind(yred);
Mapa=zeros(10,10);


for (i=1:50)
    fila=floor(indices(i)/10)+1;
 
    columna=round(rem(indices(i),10));        
    if columna==0 
        columna=10;
    end;
    Mapa(fila,columna)=1;
end;
 
for (i=51:100)
    fila=floor(indices(i)/10)+1;
    columna=round(rem(indices(i),10));  
    if columna==0 
        columna=10;
    end;    
    Mapa(fila,columna)=2;
end;

for (i=101:150)
    fila=floor(indices(i)/10)+1;
    columna=round(rem(indices(i),10));  
    if columna==0 
        columna=10;
    end;    
    Mapa(fila,columna)=3;
end;
imagesc (Mapa(1:10,1:10)); figure(gcf)

    % MWE %
    
hexplot(Mapa(1:10,1:10));





