clear, clc, close all;

%PRI
erdasdata=load('C:\Users\Ruiz Desk\Downloads\Datos Investigacion\Aviris Imagery Set\sdatosErAS\pri.asc');
erdasIndex=vec2mat(erdasdata(1:813*813,3),813);
spectralIndex=load('C:\Users\Ruiz Desk\Downloads\Datos Investigacion\Aviris Imagery Set\sdatosErAS\PRI.mat');
spectralIndex=spectralIndex.spectralIndex;
mdl_pri = fitlm(spectralIndex(:),erdasIndex(:));
disp(mdl_pri.Rsquared.Ordinary);