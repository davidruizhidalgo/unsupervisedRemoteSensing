%Normalized Water Index 2 (NWI-2)
function spectralIndex= NWI_2_idx(R970,R850)

spectralIndex=(R970-R850)./(R970+R850);  


end
