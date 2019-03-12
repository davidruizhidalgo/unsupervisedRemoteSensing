function  choice=choosedialog

    d = dialog('Position',[800 700 325 125],'Name','Favorite Indexes Editor');
    txt = uicontrol('Parent',d, 'Style','text','Position',[0 80 200 40],...
           'String','Select an Option');
       
% Create three radio buttons in the button group.
bg = uibuttongroup('Parent',d,'Visible','off',...
                  'Position',[.15 .3 .75 .5]);
r1 = uicontrol(bg,'Style',...
                  'radiobutton',...
                  'String','Add Last Created Index',...
                  'Position',[5 30 500 30],...
                  'HandleVisibility','off');

r2 = uicontrol(bg,'Style',...
                  'radiobutton',...
                  'String','Delete Created Index',...
                  'Position',[5 5 500 30],...
                  'HandleVisibility','off');
              
              
bg.Visible = 'on';

       
    btn = uicontrol('Parent',d,'Position',[130 10 70 25],...
           'String','Accept', 'Callback',@bselection);
       
    choice = 'Add Last Created Index';
    
    uiwait(d);
        
      function bselection(source,callbackdata)
                    choice = bg.SelectedObject.String;
                    delete(gcf);
      end
   
  
end