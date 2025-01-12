%Generate Import files similar to the ones used in the Mathemathica code
function Import()

filedir = pwd;                      % Location of .dat files
filenames = getFileList(filedir);   % List of files to open

openFiles();

%Returns a list with the name of the files on the filedir folder
function names = getFileList(varargin)
    dirfiles = dir(fullfile(filedir,'*Average.*'));
    dirfiles([dirfiles.isdir]) = [];
    [~,index1] = sort([dirfiles.datenum],'ascend');
    names = {dirfiles(index1).name}';
end

% Opens files and creates E1 and E2 2D matrix and Pumpindex .dat files
function openFiles
    tic;
    file = fullfile(filedir, filenames{1});
    data_struct = importdata(file);
    data = data_struct.data;
    display(data)
	m = length(data);           % m rows
    n = length(filenames);      % n columns
    
	E1_2D = zeros(m, 1+n);      % Initialize array m x 1+n
    E1_2D(:,1) = data(:,1);     % Copy time axis
    E2_2D = E1_2D;
    
    E1_2D(:,2) = data(:,2);     % Copy E1 axis
    E2_2D(:,2) = data(:,3);     % Copy E2 axis

    dlmwrite('Pumpindex.dat', (1:n)', 'newline','pc')  % Pump Index
    if exist('PumpTimes.dat', 'file') == 0
        dlmwrite('PumpTimes.dat', (-3:1:(n-4))', 'newline','pc')  % Pump Times
    end
    for i = 2 : n
        file = fullfile(filedir, filenames{i});
        data_struct = importdata(file);
        data = data_struct.data;

        E1_2D(:,i+1) = data(:,2);
        E2_2D(:,i+1) = data(:,3);
    end
    dlmwrite('E1.dat', E1_2D, 'delimiter', '\t', 'newline','pc')
    dlmwrite('E2.dat', E2_2D, 'delimiter', '\t', 'newline','pc')
    toc;

end

end