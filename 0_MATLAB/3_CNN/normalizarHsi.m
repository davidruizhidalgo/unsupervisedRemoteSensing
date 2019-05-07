%NORMALIZAR DATOS DE IMAGEN HSI
function data = normalizarHsi(data_input)
data = data_input;
for i=1:size(data,1)
    for j=1:size(data,2)
        mn = sum(data(i,j,:))/size(data,3);
        data(i,j,:) = data(i,j,:)-mn;
        st = std(data(i,j,:));
        data(i,j,:) = data(i,j,:)/st;
    end
end
end