function hexplot(dataMat)

    l=size(dataMat,1);
    b=size(dataMat,2);
    xhex=[0 1 2 2 1 0]; % x-coordinates of the vertices
    yhex=[2 3 2 1 0 1]; % y-coordinates of the vertices
    figure;
    for i=1:b
        j=i-1;
        for k=1:l
            m=k-1;
            patch((xhex+mod(k,2))+2*j,yhex+2*m,dataMat(k,i)) % make a hexagon at [2i,2j]
            hold on
        end
    end
    axis equal;
    axis ij
    axis off;

end