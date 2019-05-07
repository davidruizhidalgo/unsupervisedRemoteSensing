%CARGAR DATOS DE IMAGEN HSI
function [data,groundTh] = loadData(dataSet)
switch dataSet
    case 1
        data = load('../../../dataSets/Indian_pines.mat');
        data = data.indian_pines_corrected;
        data = normalizarHsi(data);
        groundTh=load('../../../dataSets/Indian_pines_gt.mat');
        groundTh = groundTh.indian_pines_gt;
    case 2
        data = load('../../../dataSets/Salinas.mat');
        data = data.salinas_corrected;
        data = normalizarHsi(data);
        groundTh=load('../../../dataSets/Salinas_gt.mat');
        groundTh = groundTh.salinas_gt;
    case 3
        data = load('../../../dataSets/PaviaU.mat');
        data = data.paviaU;
        data = normalizarHsi(data);
        groundTh=load('../../../dataSets/PaviaU_gt.mat');
        groundTh = groundTh.paviaU_gt;
    case 4
        data = load('../../../dataSets/Pavia.mat');
        data = data.pavia;
        data = normalizarHsi(data);
        groundTh=load('../../../dataSets/Pavia_gt.mat');
        groundTh = groundTh.pavia_gt;
    otherwise
        disp('ERROR... Imposible cargar datos')
end
end