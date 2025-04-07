function vech = vech(mat)
    n = size(mat, 2);
    vech = mat(tril(ones(n),0)==1);
end
