function varargout = HyperSpectralRemoteSensing(varargin)
% HYPERSPECTRALREMOTESENSING MATLAB code for HyperSpectralRemoteSensing.fig
%      HYPERSPECTRALREMOTESENSING, by itself, creates a new HYPERSPECTRALREMOTESENSING or raises the existing
%      singleton*.
%
%      H = HYPERSPECTRALREMOTESENSING returns the handle to a new HYPERSPECTRALREMOTESENSING or the handle to
%      the existing singleton*.
%
%      HYPERSPECTRALREMOTESENSING('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in HYPERSPECTRALREMOTESENSING.M with the given input arguments.
%
%      HYPERSPECTRALREMOTESENSING('Property','Value',...) creates a new HYPERSPECTRALREMOTESENSING or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before HyperSpectralRemoteSensing_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to HyperSpectralRemoteSensing_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help HyperSpectralRemoteSensing

% Last Modified by GUIDE v2.5 17-Jan-2017 10:25:01

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @HyperSpectralRemoteSensing_OpeningFcn, ...
                   'gui_OutputFcn',  @HyperSpectralRemoteSensing_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before HyperSpectralRemoteSensing is made visible.
function HyperSpectralRemoteSensing_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to HyperSpectralRemoteSensing (see VARARGIN)

% Choose default command line output for HyperSpectralRemoteSensing
handles.output = hObject;

% Update handles structure
guidata(hObject, handles);

% UIWAIT makes HyperSpectralRemoteSensing wait for user response (see UIRESUME)
% uiwait(handles.figure1);
 cla (handles.axes3, 'reset');
 axis(handles.axes3,'off');
 lgImg=imread('univalle.jpg');
 axes(handles.axes3); imshow(lgImg);

 cla (handles.axes1, 'reset');
 grid(handles.axes1,'on');grid(handles.axes1,'minor');
 axis(handles.axes1,'off');
 %iniImage=imread(strcat(pwd,'\Files\AppImages\','IniImage.jpg'));
 %axes(handles.axes1); imshow(iniImage);
 
stringConsola={'';'';'';'';'';'';'';'';'';'';'';'';'';'';'';''};
pointerConsola=1;
GuiHandle = ancestor(hObject, 'figure');
    %Mensaje en Consola
    [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'App Ready...',pointerConsola,GuiHandle);
    
    %CARGAR INDICES ESPECTRALES CREADOS POR EL USUARIO
    load myFavoriteIndexList;
    set(handles.createdIndexMenu,'String',myFavoriteIndexList(:,1));
    
handles.myFavoriteIndexList=myFavoriteIndexList;    
handles.pointerConsola=pointerConsola;
handles.stringConsola=stringConsola;
guidata(hObject,handles); 
 
% --- Outputs from this function are returned to the command line.
function varargout = HyperSpectralRemoteSensing_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;

% --- Executes on button press in LOADDATA_1.
function loadData1_Callback(hObject, eventdata, handles)
% hObject    handle to loadData1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of loadData1
pointerConsola=handles.pointerConsola;
stringConsola=handles.stringConsola;
GuiHandle = ancestor(hObject, 'figure');


% wavelength = [ 365, 375, 385, 394, 404, 414, 423, 433, 443, 453, 462, 472, 482, 491, 501,...
%     511, 521, 530, 540, 550, 560, 569, 579, 589, 599, 608, 618, 628, 638, 647, 657, 667, 655,...
%     665, 675, 684, 694, 704 , 714, 724, 733, 743, 753, 763, 772, 782, 792, 802, 811, 821, 831, 840, 850,...
%     860, 870, 879, 889, 899, 908, 918, 928, 937, 947 , 957, 966, 976, 985, 995, 1005, 1014, 1024, 1034, 1043,...
%     1053, 1062, 1072, 1082, 1091, 1101, 1110, 1120, 1129, 1139, 1148, 1158, 1168, 1177, 1187, 1196, 1206, 1215,...
%     1225, 1234, 1244, 1253, 1263, 1253, 1263, 1273, 1283, 1293, 1303, 1313, 1323, 1333, 1343, 1353, 1363, 1373,...
%     1382, 1392, 1402, 1412, 1422, 1432, 1442, 1452 , 1462, 1472, 1482, 1492, 1502, 1512, 1522, 1532, 1542, 1552 , 1562,...
%     1572, 1582, 1592, 1602, 1612, 1622, 1632, 1642, 1652, 1662, 1672, 1682 , 1692, 1702, 1712, 1722 , 1732, 1742, 1752,...
%     1762, 1772, 1782, 1792, 1802, 1811, 1821, 1831, 1841, 1851, 1861, 1871, 1872, 1866, 1876, 1887 , 1897, 1907, 1917, 1927,...
%     1937, 1947, 1957, 1967, 1977, 1987, 1997, 2007, 2017, 2027, 2037, 2047, 2057, 2067, 2077, 2087, 2097, 2107, 2117, 2127,...
%     2137, 2147, 2157, 2167, 2177 , 2187 , 2197 , 2207 , 2217 , 2227 , 2237 , 2247 , 2257 , 2267 , 2277 , 2287 , 2297 , 2307 ,...
%     2317 , 2327 , 2337 , 2347 , 2357 , 2367 , 2377 , 2386 , 2396 , 2406 , 2416. , 2426 , 2436 , 2446 , 2456 , 2466 , 2476 , 2486 , 2496 ];



wavelength=load('spectralBands.mat');
wavelength=wavelength.wavelength;

bands=224;
[File,Path]=uigetfile({'*.*'},'Cargar Datos');
filename=strcat(Path,File);

%Mensaje en Consola
    [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'Loading Hiperspectral Data...',pointerConsola,GuiHandle);

try
    
sData=load(filename);
samples=sData.width(2);  %The number of samples (pixels) per image line for each band.
lines = samples;        %The number of lines per image for each band.
data=sData.dataCube;        %CARGAR DATOS
    
%Cargar Imagen Original
rgbImg=imread(strcat(Path,sData.File,'_RGB.jpeg'));
axes(handles.axes1); 
iniPoint=sData.ypos;
image(rgbImg(iniPoint+1:iniPoint+lines+1,:,:)); axis off;  %Mostrar Imagen

%Actualizar Variables de la Interfaz
handles.bands=bands;
handles.samples=samples;
handles.lines=lines;
handles.data=data;
handles.wavelength=wavelength;

%Mensaje en Consola
    [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'Loading Hiperspectral Data Done...',pointerConsola,GuiHandle);
catch
    [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'ERROR, Select a Spectral Imagery Folder!!!!',pointerConsola,GuiHandle);  
end
handles.pointerConsola=pointerConsola;
handles.stringConsola=stringConsola;
guidata(hObject,handles);  

% --- Executes GENERATE AND CREATE KNOWN INDEXS
function knownGenerate_Callback(hObject, eventdata, handles)
% hObject    handle to knownGenerate (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of knownGenerate

%Variables de la Interfaz
pointerConsola=handles.pointerConsola;
stringConsola=handles.stringConsola;
GuiHandle = ancestor(hObject, 'figure');
try
bands=handles.bands;
samples=handles.samples;
lines=handles.lines;
data=handles.data;
wavelength=handles.wavelength;
fun =get(handles.knownMenu,'Value');
saveIndex=get(handles.saveIndexCheck,'Value');
spectralIndex=0;
switch fun
    case 1 %--Select the Vegetation Index...
    %Mensaje en Consola
    [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'SELECT A SPECTRAL INDEX...',pointerConsola,GuiHandle);
    %case 2 %--Show Hiperspectral Cube
    case 3 %--Normalized Water Index 1 (NWI-1)
      R970= spectralBand(data,wavelength,970);
      R900= spectralBand(data,wavelength,900);
      spectralIndex= NWI_1_idx(R970,R900);
      axes(handles.axes1); 
      imagesc(spectralIndex,[-1 1]);  colorbar; axis off;  %Mostrar Imagen
      %Mensaje en Consola
        [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'Normalized Water Index 1 (NWI1)',pointerConsola,GuiHandle);
    case 4 %--Normalized Water Index 2 (NWI-2)
      R970= spectralBand(data,wavelength,970);
      R850= spectralBand(data,wavelength,850);
      spectralIndex= NWI_2_idx(R970,R850);
      axes(handles.axes1); 
      imagesc(spectralIndex,[-1 1]);  colorbar;axis off;  %Mostrar Imagen
      %Mensaje en Consola
        [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'Normalized Water Index 2 (NWI2)',pointerConsola,GuiHandle);
    case 5 %--Normalized Water Index 3 (NWI-3)
      R970= spectralBand(data,wavelength,970);
      R880= spectralBand(data,wavelength,880);
      spectralIndex= NWI_3_idx(R970,R880);
      axes(handles.axes1); 
      imagesc(spectralIndex,[-1 1]);  colorbar;axis off;  %Mostrar Imagen
      %Mensaje en Consola
        [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'Normalized Water Index 3 (NWI3)',pointerConsola,GuiHandle);
    case 6 %--Normalized Water Index 4 (NWI-4)
      R970= spectralBand(data,wavelength,970);
      R920= spectralBand(data,wavelength,920);
      spectralIndex= NWI_4_idx(R970,R920);
      axes(handles.axes1); 
      imagesc(spectralIndex,[-1 1]);  colorbar;axis off;  %Mostrar Imagen
      %Mensaje en Consola
        [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'Normalized Water Index 4 (NWI4)',pointerConsola,GuiHandle);
    case 7 %--Ratio of Reflectance 1000 and 1100 nm
      R1000= spectralBand(data,wavelength,1000);
      R1100= spectralBand(data,wavelength,1100);
      spectralIndex= reflectanceRatio(R1000,R1100);
      axes(handles.axes1); 
      imagesc(spectralIndex,[0 1]);  colorbar;axis off;  %Mostrar Imagen
      %Mensaje en Consola
        [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'Ratio of Reflectance 1000 and 1100 nm',pointerConsola,GuiHandle);
    case 8 %--Ratio of Reflectance 940 and 960 nm
      R940= spectralBand(data,wavelength,940);
      R960= spectralBand(data,wavelength,960);
      spectralIndex= reflectanceRatio(R940,R960);
      axes(handles.axes1); 
      imagesc(spectralIndex,[0 1]);  colorbar;axis off;  %Mostrar Imagen
      %Mensaje en Consola
        [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'Ratio of Reflectance 940 and 960 nm',pointerConsola,GuiHandle);
    case 9 %--Water Band Index (WBI)
      R900= spectralBand(data,wavelength,900);
      R970= spectralBand(data,wavelength,970);
      spectralIndex= WBI_idx(R900,R970);
      axes(handles.axes1); 
      imagesc(spectralIndex,[0 1]);  colorbar;axis off;  %Mostrar Imagen
      %Mensaje en Consola
        [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'Water Band Index (WBI)',pointerConsola,GuiHandle);
    case 10 %--Normalized Difference Vegetation Index (NVDI)
      R800= spectralBand(data,wavelength,800);
      R680= spectralBand(data,wavelength,680);
      spectralIndex= NVDI_idx(R800,R680);
      axes(handles.axes1); 
      imagesc(spectralIndex,[-1 1]);  colorbar;axis off;  %Mostrar Imagen
      %Mensaje en Consola
        [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'Normalized Difference Vegetation Index (NVDI)',pointerConsola,GuiHandle);
    case 11 %--Red Normalized Difference Vegetation Index (RNVDI)
      R780= spectralBand(data,wavelength,780);
      R670= spectralBand(data,wavelength,670);
      spectralIndex= RNVDI_idx(R780,R670);
      axes(handles.axes1); 
      imagesc(spectralIndex,[-1 1]);  colorbar;axis off;  %Mostrar Imagen
      %Mensaje en Consola
        [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'Red Normalized Difference Vegetation Index (RNVDI)',pointerConsola,GuiHandle);
    case 12 %--Green Normalized Difference Vegetation Index (GNVDI)
      R780= spectralBand(data,wavelength,780);
      R550= spectralBand(data,wavelength,550);
      spectralIndex= RNVDI_idx(R780,R550);
      axes(handles.axes1); 
      imagesc(spectralIndex,[-1 1]);  colorbar;axis off;  %Mostrar Imagen
      %Mensaje en Consola
        [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'Green Normalized Difference Vegetation Index (GNVDI)',pointerConsola,GuiHandle);
    case 13 %--Ratio of Reflectance 940/960 nm and NVDI
      R940= spectralBand(data,wavelength,940);
      R960= spectralBand(data,wavelength,960);
      numIndex= reflectanceRatio(R940,R960);
      R800= spectralBand(data,wavelength,800);
      R680= spectralBand(data,wavelength,680);
      nvdiIndex=NVDI_idx(R800,R680);
      spectralIndex= reflectanceRatio(numIndex,nvdiIndex);
      axes(handles.axes1); 
      imagesc(spectralIndex,[-1 1]);  colorbar;axis off;  %Mostrar Imagen
      %Mensaje en Consola
        [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'Ratio of Reflectance 940/960 nm and NVDI',pointerConsola,GuiHandle);
    case 14 %--Ratio of Reflectance WBI and NVDI
      R900= spectralBand(data,wavelength,900);
      R970= spectralBand(data,wavelength,970);
      wbiIndex= WBI_idx(R900,R970);
      R800= spectralBand(data,wavelength,800);
      R680= spectralBand(data,wavelength,680);
      nvdiIndex=NVDI_idx(R800,R680);
      spectralIndex= reflectanceRatio(wbiIndex,nvdiIndex);
      axes(handles.axes1); 
      imagesc(spectralIndex,[-1 1]);  colorbar;axis off;  %Mostrar Imagen
      %Mensaje en Consola
        [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'Ratio of Reflectance WBI and NVDI',pointerConsola,GuiHandle);     
    case 15 %--Photosintetical Reflectance Index (PRI)
      R531= spectralBand(data,wavelength,531);
      R570= spectralBand(data,wavelength,570);
      spectralIndex= PRI_idx(R531,R570);
      axes(handles.axes1); 
      imagesc(spectralIndex,[-1 1]);  colorbar;axis off;  %Mostrar Imagen
      %Mensaje en Consola
        [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'Photosintetical Reflectance Index (PRI)',pointerConsola,GuiHandle);
    case 16 %--Simple Ratio (SR)
      R900= spectralBand(data,wavelength,900);
      R680= spectralBand(data,wavelength,680);
      spectralIndex= reflectanceRatio(R900,R680);
      axes(handles.axes1); 
      imagesc(spectralIndex,[0 1]);  colorbar;axis off;  %Mostrar Imagen
      %Mensaje en Consola
        [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'Simple Ratio (SR)',pointerConsola,GuiHandle);
    otherwise
    %Mensaje en Consola
    [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        '¡¡¡¡ FUNCTION UNDER CONSTRUCTION !!!',pointerConsola,GuiHandle);    
end



if saveIndex==1
    if length(spectralIndex)>1
        %OPCION GRABAR IMAGEN Y DATOS GENERADOS
        opc=questdlg('¿Do you want to save the index?','Exit','Yes','No','No');
        if strcmp(opc,'Yes')
            prompt={'Enter a file name'};
            name = 'Save Data Index';
            defaultans = {'myIndex'};
            options.Interpreter = 'tex';
            saveFile = inputdlg(prompt,name,[1 40],defaultans,options);
            saveFileName=strcat(saveFile,'.mat'); fileName=saveFileName{1};
            save(fileName,'spectralIndex');
        end
    end
end
%Actualizar Variables de la Interfaz
handles.bands=bands;
handles.samples=samples;
handles.lines=lines;
handles.data=data;
handles.wavelength=wavelength;
 
catch
    [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'ERROR, Select a Valid Spectral Imagery Folder!!!!',pointerConsola,GuiHandle);  
    
end
handles.pointerConsola=pointerConsola;
handles.stringConsola=stringConsola;
guidata(hObject,handles);


% --- Executes on button press in CREATE NEW INDEX.
function createNewIndex_Callback(hObject, eventdata, handles)
% hObject    handle to createNewIndex (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of createNewIndex
pointerConsola=handles.pointerConsola;
stringConsola=handles.stringConsola;
GuiHandle = ancestor(hObject, 'figure');

try
bands=handles.bands;
samples=handles.samples;
lines=handles.lines;
data=handles.data;
wavelength=handles.wavelength;

    [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'CREATING NEW SPECTRAL INDEX....',pointerConsola,GuiHandle);   
    
    indexString=indexDialog;
    stringVarArr=scanStringFunction(indexString);
    
    [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        strcat('myIndex=',indexString),pointerConsola,GuiHandle);  
    
    [newIndexData, proInfo]=evalNewIndex(stringVarArr,indexString,data,wavelength);
    dataLim=[0 1];
     if length(newIndexData)>1
        xdlg = inputdlg('Enter [min max] index limit:','Index Limit', [1 50],{'[0 1]'});
        dataLim = str2num(xdlg{:}); 
        %NEW INDEX DATA
        handles.newIndexData=newIndexData;
        %NEW INDEX VARIABLES
        handles.indexString=indexString;
        handles.stringVarArr=stringVarArr;
     end
        imagesc(newIndexData,dataLim);  colorbar;axis off;  %Mostrar Imagen
    newIndexData
    [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        proInfo,pointerConsola,GuiHandle);  

%Actualizar Variables de la Interfaz
handles.bands=bands;
handles.samples=samples;
handles.lines=lines;
handles.data=data;
handles.wavelength=wavelength;

catch
        [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'ERROR, Select a Valid Spectral Imagery Folder!!!!',pointerConsola,GuiHandle);   
    
end
handles.pointerConsola=pointerConsola;
handles.stringConsola=stringConsola;
guidata(hObject,handles); 

% --- Executes on button press in EDIT FAVORITES
function editFavorites_Callback(hObject, eventdata, handles)
% hObject    handle to editFavorites (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of editFavorites

myFavoriteIndexList=handles.myFavoriteIndexList;
pointerConsola=handles.pointerConsola;
stringConsola=handles.stringConsola;
GuiHandle = ancestor(hObject, 'figure');

try
    choice=choosedialog;
    %ADD LAST CREATED INDEX
    if choice(1)=='A'
        %Mensaje en Consola
    [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'Creating New Spectral Index...',pointerConsola,GuiHandle);
        %NEW INDEX VARIABLES
        indexString=handles.indexString;
        
        %INTRODUCE NEW INDEX NAME
        prompt={'Enter a New Index Name'};
            name = 'New Index Name';
            defaultans = {'My Spectral New Index'};
            options.Interpreter = 'tex';
            indexName = inputdlg(prompt,name,[1 40],defaultans,options);
            myFavoriteIndexList{end+1,1}=indexName{1};
            myFavoriteIndexList{end,2}=indexString;
            
            set(handles.createdIndexMenu,'String',myFavoriteIndexList(:,1));
            handles.myFavoriteIndexList=myFavoriteIndexList;
            save('myFavoriteIndexList.mat','myFavoriteIndexList');  
            
            %Mensaje en Consola
    [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'NEW SPECTRAL INDEX HAS BEEN CREATED...',pointerConsola,GuiHandle);
    %Mensaje en Consola
    [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        indexName{1},pointerConsola,GuiHandle);
       %Mensaje en Consola
    [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        indexString,pointerConsola,GuiHandle);
    
       
    end
    %DELET CREATED INDEX
    if choice(1)=='D'
        
        choiceDel=listboxdialog(myFavoriteIndexList(2:end,1));
         
        %Mensaje en Consola
                 [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        myFavoriteIndexList{choiceDel+1,1},pointerConsola,GuiHandle);
             [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        myFavoriteIndexList{choiceDel+1,2},pointerConsola,GuiHandle);
           
           %BORRAR ELEMENTO DE LA LISTA
        myFavoriteIndexList(choiceDel+1,:)=[];
    
            set(handles.createdIndexMenu,'String',myFavoriteIndexList(:,1));
            set(handles.createdIndexMenu,'Visible','on');
            handles.myFavoriteIndexList=myFavoriteIndexList;
            save('myFavoriteIndexList.mat','myFavoriteIndexList');  
       [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'SPECTRAL INDEX HAS BEEN DELETED...',pointerConsola,GuiHandle);
        
    end
    
catch
    
       %Mensaje en Consola
    [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'ERROR, NEW SPECTRAL INDEX HAS NOT BEEN CREATED',pointerConsola,GuiHandle);
    
end

handles.myFavoriteIndexList=myFavoriteIndexList;
handles.pointerConsola=pointerConsola;
handles.stringConsola=stringConsola;
guidata(hObject,handles); 

% --- Executes on button press in SAVE NEW INDEX
function saveNewIndex_Callback(hObject, eventdata, handles)
% hObject    handle to saveNewIndex (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of saveNewIndex
pointerConsola=handles.pointerConsola;
stringConsola=handles.stringConsola;
GuiHandle = ancestor(hObject, 'figure');
try
    newIndexData=handles.newIndexData;
    prompt={'Enter a file name'};
            name = 'Save New Data Index';
            defaultans = {'myNewDataIndex'};
            options.Interpreter = 'tex';
            saveFile = inputdlg(prompt,name,[1 40],defaultans,options);
            saveFileName=strcat(saveFile,'.mat'); fileName=saveFileName{1};
            save(fileName,'newIndexData');
                 %Mensaje en Consola
    [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'SUCCESFULL, New Spectral Data Has Been Saved!!!',pointerConsola,GuiHandle);
    
catch
     %Mensaje en Consola
    [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'ERROR, NEW SPECTRAL INDEX HAS NOT BEEN CREATED',pointerConsola,GuiHandle);
end

handles.pointerConsola=pointerConsola;
handles.stringConsola=stringConsola;
guidata(hObject,handles); 

% --- Executes on button press in PLOT NEW INDEX.
function plotNewIndex_Callback(hObject, eventdata, handles)
% hObject    handle to plotNewIndex (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of plotNewIndex

%Variables de la Interfaz
pointerConsola=handles.pointerConsola;
stringConsola=handles.stringConsola;
GuiHandle = ancestor(hObject, 'figure');
try
myFavoriteIndexList=handles.myFavoriteIndexList;
bands=handles.bands;
samples=handles.samples;
lines=handles.lines;
data=handles.data;
wavelength=handles.wavelength;
fun =get(handles.createdIndexMenu,'Value');
switch fun
    case 1 %--Select the Vegetation Index...
    %Mensaje en Consola
    [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'SELECT A SPECTRAL INDEX...',pointerConsola,GuiHandle);
    
    otherwise
        
        stringArr=scanStringFunction(myFavoriteIndexList{fun,2});   
    [indexData, proInfo]=evalNewIndex(stringArr,myFavoriteIndexList{fun,2},data,wavelength);
    dataLim=[0 1];
     if length(indexData)>1
        xdlg = inputdlg('Enter [min max] index limit:','Index Limit', [1 50],{'[-1 1]'});
        dataLim = str2num(xdlg{:}); 
     end
        imagesc(indexData,dataLim);  colorbar;axis off;  %Mostrar Imagen
    
    [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        myFavoriteIndexList{fun,1},pointerConsola,GuiHandle);  

           
end


%Actualizar Variables de la Interfaz
handles.myFavoriteIndexList=myFavoriteIndexList;
handles.bands=bands;
handles.samples=samples;
handles.lines=lines;
handles.data=data;
handles.wavelength=wavelength;
catch
    [pointerConsola,stringConsola]=printTerminal(stringConsola,...
        'ERROR, Select a Valid Spectral Imagery Folder!!!!',pointerConsola,GuiHandle);  
end
handles.pointerConsola=pointerConsola;
handles.stringConsola=stringConsola;
guidata(hObject,handles);
   
    % --- Executes on button press in AcercaDe.
function AcercaDe_Callback(hObject, eventdata, handles)
% hObject    handle to AcercaDe (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of AcercaDe
msgbox(['HYPERSPECTRAL REMOTE SENSIG OF VEGETATION',char(10),...
                  '                                 Univalle - 2016'],'About App...');

% --- Executes on button press in Exit.
function Exit_Callback(hObject, eventdata, handles)
% hObject    handle to Exit (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of Exit
opc=questdlg('¿Do you want to close the app?','Exit','Yes','No','No');
if strcmp(opc,'No')
return;
end
try
clear,clc,close all
catch
    clear,clc,close all;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% --- Executes during object creation, after setting all properties.
function knownMenu_CreateFcn(hObject, eventdata, handles)
% hObject    handle to knownMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end
% --- knownMenu_Callback
function knownMenu_Callback(hObject, eventdata, handles)
% hObject    handle to knownMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns knownMenu contents as cell array
%        contents{get(hObject,'Value')} returns selected item from knownMenu

% --- Executes on selection change in popupmenu4.
function popupmenu4_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu4 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu4


% --- Executes during object creation, after setting all properties.
function popupmenu4_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu4 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end



% --- Executes on selection change in createdIndexMenu.
function createdIndexMenu_Callback(hObject, eventdata, handles)
% hObject    handle to createdIndexMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns createdIndexMenu contents as cell array
%        contents{get(hObject,'Value')} returns selected item from createdIndexMenu


% --- Executes during object creation, after setting all properties.
function createdIndexMenu_CreateFcn(hObject, eventdata, handles)
% hObject    handle to createdIndexMenu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on selection change in popupmenu6.
function popupmenu6_Callback(hObject, eventdata, handles)
% hObject    handle to popupmenu6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hints: contents = cellstr(get(hObject,'String')) returns popupmenu6 contents as cell array
%        contents{get(hObject,'Value')} returns selected item from popupmenu6


% --- Executes during object creation, after setting all properties.
function popupmenu6_CreateFcn(hObject, eventdata, handles)
% hObject    handle to popupmenu6 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: popupmenu controls usually have a white background on Windows.
%       See ISPC and COMPUTER.
if ispc && isequal(get(hObject,'BackgroundColor'), get(0,'defaultUicontrolBackgroundColor'))
    set(hObject,'BackgroundColor','white');
end


% --- Executes on button press in saveIndexCheck.
function saveIndexCheck_Callback(hObject, eventdata, handles)
% hObject    handle to saveIndexCheck (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Hint: get(hObject,'Value') returns toggle state of saveIndexCheck
