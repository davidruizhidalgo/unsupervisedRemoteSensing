function  choice=listboxdialog(stringList)

    d = dialog('Position',[800 700 325 125],'Name','Favorite Indexes Editor');
    txt = uicontrol('Parent',d, 'Style','text','Position',[0 80 200 40],...
           'String','Select an Option to Delete');
       

   lbox=  uicontrol('Parent',d, 'Style','list','Position',[25 40 275 60],...
           'String',stringList);         
              


       
    btn = uicontrol('Parent',d,'Position',[130 10 70 25],...
           'String','Delete', 'Callback',@bselection);
       
    
    uiwait(d);
        
      function bselection(source,callbackdata)
                    choice =lbox.Value;
                    delete(gcf);
      end
   
  
end