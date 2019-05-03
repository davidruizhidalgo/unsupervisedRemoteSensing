function [newIndexData, proInfo]=evalNewIndex(stringVarArr,indexString,data,wavelength)


try
for i=1:length(stringVarArr)
    vname=matlab.lang.makeValidName(stringVarArr{i,1});
    if stringVarArr{i,1}(1) == 'R' && stringVarArr{i,1}(2) ~= 'N'
        band=str2double(stringVarArr{i,1}(2:end));
        eval([vname '=spectralBand(data,wavelength,band);']);       
    else
        switch stringVarArr{i,1}
            case 'NWI1'
                R970= spectralBand(data,wavelength,970);
                R900= spectralBand(data,wavelength,900);
                NWI1= NWI_1_idx(R970,R900);
            case 'NWI2'
                R970= spectralBand(data,wavelength,970);
                R850= spectralBand(data,wavelength,850);
                NWI2= NWI_2_idx(R970,R850);
            case 'NWI3'
                R970= spectralBand(data,wavelength,970);
                R880= spectralBand(data,wavelength,880);
                NWI3= NWI_3_idx(R970,R880);
            case 'NWI4'
                R970= spectralBand(data,wavelength,970);
                R920= spectralBand(data,wavelength,920);
                NWI4= NWI_4_idx(R970,R920);
            case 'WBI'
                R900= spectralBand(data,wavelength,900);
                R970= spectralBand(data,wavelength,970);
                WBI= WBI_idx(R900,R970);
            case 'NVDI'
                R800= spectralBand(data,wavelength,800);
                R680= spectralBand(data,wavelength,680);
                NVDI= NVDI_idx(R800,R680);
            case 'RNVDI'
                R780= spectralBand(data,wavelength,780);
                R670= spectralBand(data,wavelength,670);
                RNVDI= RNVDI_idx(R780,R670);
            case 'GNVDI'
                R780= spectralBand(data,wavelength,780);
                R550= spectralBand(data,wavelength,550);
                GNVDI= GNVDI_idx(R780,R550);
            case 'PRI'
                R531= spectralBand(data,wavelength,531);
                R570= spectralBand(data,wavelength,570);
                PRI= PRI_idx(R531,R570);
            case 'SR'
                R900= spectralBand(data,wavelength,900);
                R680= spectralBand(data,wavelength,680);
                SR= reflectanceRatio(R900,R680);
            otherwise
        end
    end
end

newIndexData=eval(indexString);
proInfo='¡¡¡¡¡INDEX CREATION SUCESSFUL!!!!!';
catch
newIndexData=0;
proInfo='ERROR, INVALID EQUATION. PROCESS FAIL';  
end

end