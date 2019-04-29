function stringArray=scanStringFunction(indexString)

j=1;
stringVar='';
for i=1:length(indexString)
    if indexString(i)~='+' && indexString(i)~='-' && ...
            indexString(i)~='*' && indexString(i)~='/' && ...
            indexString(i)~='.' && indexString(i)~='(' && indexString(i)~=')'

        stringVar=strcat(stringVar,indexString(i));
    else
        if i~=1 && ~isempty(stringVar)
            if stringVar(1)~='0' && stringVar(1)~='1' && ...
                    stringVar(1)~='2' && stringVar(1)~='3' && ...
                    stringVar(1)~='4' && stringVar(1)~='5' && ...
                    stringVar(1)~='6' && stringVar(1)~='7' && ...
                    stringVar(1)~='8' && stringVar(1)~='9' 
                
                stringArray{j,1}=stringVar;
                j=j+1;
            end
         stringVar='';
        end
    end
end

if ~isempty(stringVar)
    if stringVar(1)~='0' && stringVar(1)~='1' && ...
                    stringVar(1)~='2' && stringVar(1)~='3' && ...
                    stringVar(1)~='4' && stringVar(1)~='5' && ...
                    stringVar(1)~='6' && stringVar(1)~='7' && ...
                    stringVar(1)~='8' && stringVar(1)~='9' 
        stringArray{j,1}=stringVar;
    end
end

end