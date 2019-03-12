%Normalized Difference Vegetation Index (NVDI)
function spectralIndex= NVDI_idx(R800,R680)

spectralIndex=(R800-R680)./(R800+R680);  


end
