function [z,xyind, xy, SiCell] = get_data_vec(data_mat,grid, mu, bycol, covmat)

if nargin < 4 || isempty(bycol)
    bycol = 1;
end

if nargin < 5 || isempty(covmat)
    covmat = 0;
end

gs = length(grid);

nlpts = (gs^2+gs)/2;
z =  zeros(nlpts, 1);
full_ind = 1:nlpts;
n0 = zeros(nlpts, 1);
if bycol
    xy = vech_loc(grid, 'L');
else
    xy = vech_row_loc(grid, 'L');
end
if covmat
    SiCell = cell(1,1);
    SiCell{1} = data_mat;
    xyind = 1:nlpts;
    z = vech(data_mat);
    xyind = xyind(~isnan(z));
else
    n = size(data_mat, 1);
    SiCell = cell(n,1);
    for i = 1:n
        yi = data_mat(i,:);
        ind = ~ismissing(yi);
        yi = yi(ind);
        ti = grid(ind);
        SiCell{i} = raw_covi(yi, ti, mu, grid);
        if bycol
            zi = vech(SiCell{i});
            xyi = vech_loc(ti, 'L');
        else 
            zi = vech_row(SiCell{i});
            xyi = vech_row_loc(ti, 'L');
        end
        ind = full_ind(ismember(xy, xyi));
        n0(ind) = n0(ind)+1;
        z(ind) = z(ind)+zi;
    end
    
    ind = n0 ~= 0;
    n0 = n0(ind);
    z = z(ind);
    z = z./n0;
    xyind = 1:nlpts;
    xyind = xyind(ind);
end
end

function vech = vech(mat)
    n = size(mat, 2);
    vech = mat(tril(ones(n),0)==1);
end

function covi = raw_covi(yi, ti, mu, grid)
    ti_ind = ismember(grid , ti);
    yic = yi - mu(ti_ind)';
    covi = kron(yic, yic');
end

function [loc] = vech_loc(grid, or)
    if nargin < 2
        or = 'L';
    end
    g = length(grid);
    x = [];
    y = [];
    if strcmp(or, 'L')
        for i = 1:g
            x = [x; grid(i:g)];
            y = [y; ones(g-i+1, 1)*grid(i)];
        end
    elseif strcmp(or, 'U')
        for i = 1:g
            x = [x; grid(1:i)'];
            y = [y; ones(i, 1)*grid(i)];
        end
    else
        error('Orientation is either U or L');
    end
    loc = table(x, y);
end

function vechR = vech_row(mat)
    nr = size(mat, 2);
    vechR = (nr^2+nr)/2;
    r = 1;
    for i = 1:nr
        for j = 1:i
            vechR(r) = mat(i,j);
            r = r+1;
        end
    end
    vechR = vechR';
end

function loc = vech_row_loc(grid, or)
    if nargin < 2
        or = 'L';
    end
    gs = length(grid);
    x = []; y = [];
    if strcmp(or, 'L')
        for i = 1:gs
            x = [x; repelem(grid(i), i)'];
            y = [y; grid(1:i)];
        end
    elseif strcmp(or, 'U')
        for i = 1:gs
            x = [x; grid(1:i)];
            y = [y; repelem(grid(i), i)'];
        end
    else
        error('Orientation is either U or L');
    end
    loc = table(x, y);
end
