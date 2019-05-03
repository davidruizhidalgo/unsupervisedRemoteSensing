function  [expression]=indexDialog

    d = dialog('Position',[800 700 325 125],'Name','My Index');
    txt = uicontrol('Parent',d, 'Style','text','Position',[0 80 200 40],...
           'String','Write the New Index Expression');
       
    edittxt = uicontrol('Parent',d, 'Style','edit','Position',[30 50 260 40],...
           'String','');      
       
      
    btnAccept = uicontrol('Parent',d,'Position',[80 10 70 25],...
           'String','Accept', 'Callback',@baccept);
       
    btnHelp = uicontrol('Parent',d,'Position',[170 10 70 25],...
           'String','Help', 'Callback',@bhelp);
           
    uiwait(d);
        
      function baccept(source,callbackdata)
                    expression= edittxt.String;
                    delete(gcf);
      end
  
  
      function bhelp(source,callbackdata)
                    
          helpdlg(['Enter the index equations following the next rules:',char(10)...
              ,char(10),'-The spectral band number must be written like R+band. For example, spectral band 880 must be written like R880.'...
              ,char(10),'-The equations editor accept expressions in the same way than Matlab® prompt editor.'...
              ,char(10),'-The user is able to use anyone of the implemented spectral indexes like NWI1, NWI2, NWI3, NWI4, WBI, NVDI, RNVDI, GNVDI, PRI and SR'...
              ,char(10)],'Equation Editor');
          
      end
   
  
end