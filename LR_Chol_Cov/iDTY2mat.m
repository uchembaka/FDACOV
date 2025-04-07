function funmat = iDTY2mat(data)
IDs = unique(data(:,1));
n = length(IDs);
grid = unique(data(:,2));
M = length(grid);
funmat = nan(n, M);

for i = 1:n
    iind = find(data(:,1) == IDs(i));
    idx = ismember(grid, data(iind,2));
    funmat(i,idx) = data(iind, 3);
end

end