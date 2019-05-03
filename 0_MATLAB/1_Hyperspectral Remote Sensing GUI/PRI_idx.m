%Photosintetical Reflectance Index (PRI)
function spectralIndex= PRI_idx(R531,R570)

spectralIndex=(R531-R570)./(R531+R570);  


end
