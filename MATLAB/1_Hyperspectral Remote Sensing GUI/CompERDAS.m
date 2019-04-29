%COMPARACION INDICES ESPECTRALES CON ERDEAS%
clear, clc, close all;

%NDVI
erdasdata=load('C:\Users\Ruiz Desk\Downloads\Datos Investigacion\Aviris Imagery Set\f130803t01p00r11_refl\sdatosErDAS\ndvi.asc');
erdasIndex=vec2mat(erdasdata(1:813*813,3),813);
spectralIndex=load('C:\Users\Ruiz Desk\Downloads\Datos Investigacion\Aviris Imagery Set\f130803t01p00r11_refl\sdatosErDAS\NDVI.mat');
spectralIndex=spectralIndex.spectralIndex;
mdl_ndvi = fitlm(spectralIndex(:),erdasIndex(:));
disp(mdl_ndvi.Rsquared.Ordinary);

%GNVDI
erdasdata=load('C:\Users\Ruiz Desk\Downloads\Datos Investigacion\Aviris Imagery Set\f130803t01p00r11_refl\sdatosErDAS\gndvi.asc');
erdasIndex=vec2mat(erdasdata(1:813*813,3),813);
spectralIndex=load('C:\Users\Ruiz Desk\Downloads\Datos Investigacion\Aviris Imagery Set\f130803t01p00r11_refl\sdatosErDAS\GNDVI.mat');
spectralIndex=spectralIndex.spectralIndex;
mdl_gndvi = fitlm(spectralIndex(:),erdasIndex(:));
disp(mdl_gndvi.Rsquared.Ordinary);


%RNVDI
erdasdata=load('C:\Users\Ruiz Desk\Downloads\Datos Investigacion\Aviris Imagery Set\f130803t01p00r11_refl\sdatosErDAS\rndvi.asc');
erdasIndex=vec2mat(erdasdata(1:813*813,3),813);
spectralIndex=load('C:\Users\Ruiz Desk\Downloads\Datos Investigacion\Aviris Imagery Set\f130803t01p00r11_refl\sdatosErDAS\RNDVI.mat');
spectralIndex=spectralIndex.spectralIndex;
mdl_rndvi = fitlm(spectralIndex(:),erdasIndex(:));
disp(mdl_rndvi.Rsquared.Ordinary);


%WBI
erdasdata=load('C:\Users\Ruiz Desk\Downloads\Datos Investigacion\Aviris Imagery Set\f130803t01p00r11_refl\sdatosErDAS\wbi.asc');
erdasIndex=vec2mat(erdasdata(1:813*813,3),813);
spectralIndex=load('C:\Users\Ruiz Desk\Downloads\Datos Investigacion\Aviris Imagery Set\f130803t01p00r11_refl\sdatosErDAS\WBI.mat');
spectralIndex=spectralIndex.spectralIndex;
mdl_wbi = fitlm(spectralIndex(:),erdasIndex(:));
disp(mdl_wbi.Rsquared.Ordinary);


%NWBI_1
erdasdata=load('C:\Users\Ruiz Desk\Downloads\Datos Investigacion\Aviris Imagery Set\f130803t01p00r11_refl\sdatosErDAS\nwbi_1.asc');
erdasIndex=vec2mat(erdasdata(1:813*813,3),813);
spectralIndex=load('C:\Users\Ruiz Desk\Downloads\Datos Investigacion\Aviris Imagery Set\f130803t01p00r11_refl\sdatosErDAS\NWBI_1.mat');
spectralIndex=spectralIndex.spectralIndex;
mdl_nwbi_1 = fitlm(spectralIndex(:),erdasIndex(:));
disp(mdl_nwbi_1.Rsquared.Ordinary);


%NWBI_2
erdasdata=load('C:\Users\Ruiz Desk\Downloads\Datos Investigacion\Aviris Imagery Set\f130803t01p00r11_refl\sdatosErDAS\nwbi_2.asc');
erdasIndex=vec2mat(erdasdata(1:813*813,3),813);
spectralIndex=load('C:\Users\Ruiz Desk\Downloads\Datos Investigacion\Aviris Imagery Set\f130803t01p00r11_refl\sdatosErDAS\NWBI_2.mat');
spectralIndex=spectralIndex.spectralIndex;
mdl_nwbi_2 = fitlm(spectralIndex(:),erdasIndex(:));
disp(mdl_nwbi_2.Rsquared.Ordinary);


%NWBI_3
erdasdata=load('C:\Users\Ruiz Desk\Downloads\Datos Investigacion\Aviris Imagery Set\f130803t01p00r11_refl\sdatosErDAS\nwbi_3.asc');
erdasIndex=vec2mat(erdasdata(1:813*813,3),813);
spectralIndex=load('C:\Users\Ruiz Desk\Downloads\Datos Investigacion\Aviris Imagery Set\f130803t01p00r11_refl\sdatosErDAS\NWBI_3.mat');
spectralIndex=spectralIndex.spectralIndex;
mdl_nwbi_3 = fitlm(spectralIndex(:),erdasIndex(:));
disp(mdl_nwbi_3.Rsquared.Ordinary);


%NWBI_4
erdasdata=load('C:\Users\Ruiz Desk\Downloads\Datos Investigacion\Aviris Imagery Set\f130803t01p00r11_refl\sdatosErDAS\nwbi_4.asc');
erdasIndex=vec2mat(erdasdata(1:813*813,3),813);
spectralIndex=load('C:\Users\Ruiz Desk\Downloads\Datos Investigacion\Aviris Imagery Set\f130803t01p00r11_refl\sdatosErDAS\NWBI_4.mat');
spectralIndex=spectralIndex.spectralIndex;
mdl_nwbi_4 = fitlm(spectralIndex(:),erdasIndex(:));
disp(mdl_nwbi_4.Rsquared.Ordinary);


%PRI
erdasdata=load('C:\Users\Ruiz Desk\Downloads\Datos Investigacion\Aviris Imagery Set\f130803t01p00r11_refl\sdatosErDAS\pri.asc');
erdasIndex=vec2mat(erdasdata(1:813*813,3),813);
spectralIndex=load('C:\Users\Ruiz Desk\Downloads\Datos Investigacion\Aviris Imagery Set\f130803t01p00r11_refl\sdatosErDAS\PRI.mat');
spectralIndex=spectralIndex.spectralIndex;
mdl_pri = fitlm(spectralIndex(:),erdasIndex(:));
disp(mdl_pri.Rsquared.Ordinary);


%SIMPLERADIO
erdasdata=load('C:\Users\Ruiz Desk\Downloads\Datos Investigacion\Aviris Imagery Set\f130803t01p00r11_refl\sdatosErDAS\simpleradio.asc');
erdasIndex=vec2mat(erdasdata(1:813*813,3),813);
spectralIndex=load('C:\Users\Ruiz Desk\Downloads\Datos Investigacion\Aviris Imagery Set\f130803t01p00r11_refl\sdatosErDAS\SIMPLERADIO.mat');
spectralIndex=spectralIndex.spectralIndex;
mdl_simpleradio = fitlm(spectralIndex(:),erdasIndex(:));
disp(mdl_simpleradio.Rsquared.Ordinary);


%R1000_R1100
erdasdata=load('C:\Users\Ruiz Desk\Downloads\Datos Investigacion\Aviris Imagery Set\f130803t01p00r11_refl\sdatosErDAS\r1000_r1100.asc');
erdasIndex=vec2mat(erdasdata(1:813*813,3),813);
spectralIndex=load('C:\Users\Ruiz Desk\Downloads\Datos Investigacion\Aviris Imagery Set\f130803t01p00r11_refl\sdatosErDAS\R1000_R1100.mat');
spectralIndex=spectralIndex.spectralIndex;
mdl_r1000_r1100 = fitlm(spectralIndex(:),erdasIndex(:));
disp(mdl_r1000_r1100.Rsquared.Ordinary);


%R940_R960
erdasdata=load('C:\Users\Ruiz Desk\Downloads\Datos Investigacion\Aviris Imagery Set\f130803t01p00r11_refl\sdatosErDAS\r940_r960.asc');
erdasIndex=vec2mat(erdasdata(1:813*813,3),813);
spectralIndex=load('C:\Users\Ruiz Desk\Downloads\Datos Investigacion\Aviris Imagery Set\f130803t01p00r11_refl\sdatosErDAS\R940_R960.mat');
spectralIndex=spectralIndex.spectralIndex;
mdl_r940_r960 = fitlm(spectralIndex(:),erdasIndex(:));
disp(mdl_r940_r960.Rsquared.Ordinary);




