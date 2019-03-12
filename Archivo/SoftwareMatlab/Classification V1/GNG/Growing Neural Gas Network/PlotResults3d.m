function PlotResults3d(x, w, C)
    N = size(w,1);
    plot3(x(:,1),x(:,2),x(:,3),'.');
    hold on;
        
    for i=1:N-1
        for j=i:N
            if C(i,j)==1
                plot3([w(i,1);w(j,1)],[w(i,2);w(j,2)],[w(i,3);w(j,3)],'r','LineWidth',2);
            end
        end
    end
    plot3(w(:,1),w(:,2),w(:,3),'ko','MarkerFaceColor','y','MarkerSize',10);
    hold off;
    axis equal;
    grid on;
end