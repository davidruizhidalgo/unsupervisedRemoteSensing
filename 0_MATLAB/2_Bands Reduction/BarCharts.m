% Bar Charts of the Manuscript
clc; clear; close all;
% %Figure 12
% X = categorical({'3x3','5x5','7x7', '10x10', '13x13'});
% X = reordercats(X,{'3x3','5x5','7x7', '10x10', '13x13'});
% data = [91 91.5 92.5 90 86];
% bar(X,data,0.4)
% ylim([80 94]);
% text(1:length(data),data,num2str(data'),'vert','bottom','horiz','center'); 
% grid on;
% xlabel('Self-Organized Map Size (MxM)')
% ylabel('% Accuracy')

% %Figure 13
% data = [90.4 91.2 89.4 78.7];
% bar(data,0.4)
% ylim([70 95]);
% text(1:length(data),data,num2str(data'),'vert','bottom','horiz','center'); 
% grid on;
% xlabel('Wavelet Decomposition Level')
% ylabel('% Accuracy')

% %Figure 16
% X = categorical({'3x3','5x5','7x7', '10x10', '13x13'});
% X = reordercats(X,{'3x3','5x5','7x7', '10x10', '13x13'});
% data = [89.3 89.7 91.2 89.2 88.9];
% bar(X,data,0.4)
% ylim([85 92]);
% text(1:length(data),data,num2str(data'),'vert','bottom','horiz','center'); 
% grid on;
% xlabel('Self-Organized Map Size (MxM)')
% ylabel('% Accuracy')

% %Figure 17
% data = [88.27 88.79 89.76 89.14 88.61];
% bar(data,0.4)
% ylim([85 90]);
% text(1:length(data),data,num2str(data'),'vert','bottom','horiz','center'); 
% grid on;
% xlabel('Wavelet Decomposition Level')
% ylabel('% Accuracy')

% %Figure 20
% X = categorical({'3x3','5x5','7x7', '8x8', '9x9'});
% X = reordercats(X,{'3x3','5x5','7x7', '8x8', '9x9'});
% data = [75.1 78.9 80.5 81.9 81.6];
% bar(X,data,0.4)
% ylim([70 85]);
% text(1:length(data),data,num2str(data'),'vert','bottom','horiz','center'); 
% grid on;
% xlabel('Self-Organized Map Size (MxM)')
% ylabel('% Accuracy')

%Figure 21
data = [64.3 64.1 68.2 69.7 65.8];
bar(data,0.4)
ylim([60 71]);
text(1:length(data),data,num2str(data'),'vert','bottom','horiz','center'); 
grid on;
xlabel('Wavelet Decomposition Level')
ylabel('% Accuracy')
