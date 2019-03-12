%Normalized Water Index 3 (NWI-3)
function spectralIndex= NWI_3_idx(R970,R880)

spectralIndex=(R970-R880)./(R970+R880);  


end
