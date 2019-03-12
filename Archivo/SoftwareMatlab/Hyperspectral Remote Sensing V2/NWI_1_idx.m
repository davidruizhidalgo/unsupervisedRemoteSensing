%Normalized Water Index 1 (NWI-1)
function spectralIndex= NWI_1_idx(R970,R900)

spectralIndex=(R970-R900)./(R970+R900);  


end
