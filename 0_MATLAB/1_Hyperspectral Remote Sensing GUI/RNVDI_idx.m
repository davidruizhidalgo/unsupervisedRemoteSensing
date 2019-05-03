%Red Normalized Difference Vegetation Index (RNVDI)
function spectralIndex= RNVDI_idx(R780,R670)

spectralIndex=(R780-R670)./(R780+R670);  


end
