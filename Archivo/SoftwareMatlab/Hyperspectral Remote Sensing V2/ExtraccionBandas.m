clear, clc, close all;
[File,Path]=uigetfile({'*.*'},'Cargar Datos');
filename=strcat(Path,File);


% Read in the reflectance data.
ypos=0;       %Salto de Posicion en el Archivo
bands=  224;  %Bands
Samples= 813; %Total Image Width
Lines= 3794;   %Total Image Height  
width = [1 Samples];                %Data Width   
height= [ypos+1 ypos+Samples];        %Data Height
data_type = 'uint16';
interleave = 'bip';
dataCube = multibandread(filename, [Lines Samples bands], data_type, 0, interleave, 'ieee-le',...
            {'Row', 'Range', height}, {'Column', 'Range', width}, ...
            {'Band', 'Range', [1 bands]} );

% Normalize to proper reflectance units.
% dataCube = dataCube ./ 10e3;

% Datos de una Banda Especifica
wavelength = [ 365, 375, 385, 394, 404, 414, 423, 433, 443, 453, 462, 472, 482, 491, 501,...
    511, 521, 530, 540, 550, 560, 569, 579, 589, 599, 608, 618, 628, 638, 647, 657, 667, 655,...
    665, 675, 684, 694, 704 , 714, 724, 733, 743, 753, 763, 772, 782, 792, 802, 811, 821, 831, 840, 850,...
    860, 870, 879, 889, 899, 908, 918, 928, 937, 947 , 957, 966, 976, 985, 995, 1005, 1014, 1024, 1034, 1043,...
    1053, 1062, 1072, 1082, 1091, 1101, 1110, 1120, 1129, 1139, 1148, 1158, 1168, 1177, 1187, 1196, 1206, 1215,...
    1225, 1234, 1244, 1253, 1263, 1253, 1263, 1273, 1283, 1293, 1303, 1313, 1323, 1333, 1343, 1353, 1363, 1373,...
    1382, 1392, 1402, 1412, 1422, 1432, 1442, 1452 , 1462, 1472, 1482, 1492, 1502, 1512, 1522, 1532, 1542, 1552 , 1562,...
    1572, 1582, 1592, 1602, 1612, 1622, 1632, 1642, 1652, 1662, 1672, 1682 , 1692, 1702, 1712, 1722 , 1732, 1742, 1752,...
    1762, 1772, 1782, 1792, 1802, 1811, 1821, 1831, 1841, 1851, 1861, 1871, 1872, 1866, 1876, 1887 , 1897, 1907, 1917, 1927,...
    1937, 1947, 1957, 1967, 1977, 1987, 1997, 2007, 2017, 2027, 2037, 2047, 2057, 2067, 2077, 2087, 2097, 2107, 2117, 2127,...
    2137, 2147, 2157, 2167, 2177 , 2187 , 2197 , 2207 , 2217 , 2227 , 2237 , 2247 , 2257 , 2267 , 2277 , 2287 , 2297 , 2307 ,...
    2317 , 2327 , 2337 , 2347 , 2357 , 2367 , 2377 , 2386 , 2396 , 2406 , 2416. , 2426 , 2436 , 2446 , 2456 , 2466 , 2476 , 2486 , 2496 ];
     
bandNumber=365;
[mdata, isdata]=min(abs(wavelength-bandNumber));
dataBand=dataCube(:,:,isdata);
imagesc(dataBand)

saveFileName=strcat(File,'_',num2str(ypos));
save(saveFileName,'File','ypos','width','height','dataCube','-v7.3');