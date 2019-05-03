%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPML111
% Project Title: Growing Neural Gas Network in MATLAB
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

function PlotResults(X, w, C)

    N = size(w,1);

    plx=size(X,2);
    if plx==2
    plot(X(:,1),X(:,2),'.');
    hold on;
    end
    
    for i=1:N-1
        for j=i:N
            if C(i,j)==1
                plot([w(i,1) w(j,1)],[w(i,2) w(j,2)],'r','LineWidth',2);
                hold on;
            end
        end
    end
    plot(w(:,1),w(:,2),'ko','MarkerFaceColor','y','MarkerSize',10);
    hold off;
    axis equal;
    grid on;
    
end