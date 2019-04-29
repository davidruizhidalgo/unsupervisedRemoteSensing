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

Choices = {'Example 1 (Circles)', 'Example 2 (Jain)', 'Example 3 (Spiral)'};

ANSWER = questdlg('Select the example to run.', ...
                  'Growing Neural Gass', ...
                  Choices{1}, Choices{2}, Choices{3}, ...
                  Choices{1});

if strcmpi(ANSWER, Choices{1})
    app1;
    return;
end

if strcmpi(ANSWER, Choices{2})
    app2;
    return;
end

if strcmpi(ANSWER, Choices{3})
    app3;
    return;
end
