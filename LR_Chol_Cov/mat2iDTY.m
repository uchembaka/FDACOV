function iDTY = mat2iDTY(funmat, grid)
[n,M] = size(funmat);
if nargin < 2
    grid = linspace(0,1,M)';
else
    if isrow(grid)
        grid = grid';
    end
end
iDTY =  [];
for i = 1:n
    ind = ~isnan(funmat(i,:));
    mi = sum(ind);
    iDTY = [iDTY; [repmat(i, mi,1), grid(ind), funmat(i,ind)']];
end
end