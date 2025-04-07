function [R, cvec, r, GCV, lambda] = fitR_spLLK(data, SiCell, S, grid, r, p, basis, lambda, penmat, omega, W, options )

M = length(grid);

%rank selection
if isempty(r) || length(r) > 1
    disp('rank selection ...');
    if isempty(r)
        r = 1:round(0.2*M);
    end
     mrsse = zeros(length(r),1);
     prev_mrsse = 0;
    for i = 1:length(r)
        cvec0 = abs(rand(p*r(i)+1,1));
        cvec = fminunc(@loss, cvec0, options, SiCell, data,r(i), p, basis, penmat, 0, W);
        Shat = get_Shat(cvec,data,basis, r(i), p);
        mrsse(i) = rmse(Shat, S, 'all');
        curr_mrsse = mrsse(i);
        if abs(prev_mrsse - curr_mrsse) <= omega || i == length(r)
            r = r(i);
            break
        else
            prev_mrsse = mrsse(i);
        end 
    end
    disp(['r = ',num2str(r)]);
end

LLKGcv = [];
cvec0 = abs(rand(p*r+1,1));
cvec0 = fminunc(@loss, cvec0, options, SiCell, data,r, p, basis, penmat, 0, W); %init
cvec = [];

%lambda selection
if isempty(lambda) || length(lambda) > 1

    if isempty(gcp('nocreate'))
        parpool;
    end
    disp('lambda selection ...');
    if isempty(lambda)
        lambda = 10.^linspace(-6, 1, 8);
    end
    cvecGCV = cell(length(lambda), 1);


    parfor i = 1:length(lambda)
        [cvec, f] = fminunc(@loss_LLK, cvec0, options, SiCell, data,r, p, basis, penmat, lambda(i));
        trH = traceH(data, basis, penmat, lambda(i), r);
        LLKGcv(i) = f./((1 - (trH/length(S))).^2);
        cvecGCV{i} = cvec;
    end
    minGCV = knee_pt(LLKGcv, 1:length(lambda))-1;
    lam = lambda(minGCV);
    cvec = cvecGCV{minGCV};
    if lam == lambda(end)
        warning('Last value in lambda grid selected')
    end
    lambda = lam;
    disp(['lambda_index = ',num2str(minGCV)]);
    
end


if isempty(cvec)
    cvec = fminunc(@loss_LLK, cvec0, options, SiCell, data,r, p, basis, penmat, lambda);
end
[cvec,f] = fminunc(@loss_LLK, cvec, options, SiCell, data,r, p, basis, penmat, lambda); % LLK
R = get_R(cvec, basis, M, r, p);
if isempty(LLKGcv)
    trH = traceH(data, basis, penmat, lambda, r);
    GCV = f./((1 - (trH/length(S))).^2);
else
    GCV = LLKGcv;
end

end





%Loss function 
function [f, g]= loss(cvec, SiCell, data, r, p, basis, P, lambda, W)
n = length(unique(data(:,1)));
indB = cumsum([1, repelem(p,r)]);
Pen = 0; f=0; m2 = 0;
if nargout > 1
    g = zeros(p*r+1,1);
    ind_c = cumsum([0, repelem(p,r)]);
end


for i = 1:n
    iind = find(data(:,1) == i);
    mi = length(iind);
    ti = data(iind,2);
    Bi = eval_basis(ti, basis);
    Ri = zeros(mi, r);
    wi = W{i};
    for j = 1:r
        C = cvec(indB(j):j*p);
        Ri(:,j) = Bi*C;
    end
    sig2 = exp(cvec(end));
    Sihat = Ri*Ri' + sig2*eye(mi);
    vSihat = vech(Sihat);
    vSi = vech(SiCell{i});
    Resi = wi*(vSi-vSihat);
    fi = sum(Resi.^2);
    f = f+fi/mi;
    m2 = m2 + mi;

    if nargout > 1
        gi = zeros(p*r+1,1);
        vech_ind = vech(tril(reshape(1:mi^2, mi, mi)));
        for j = 1:r
            RTDRDc = kron(Bi', Ri(:,j)');
            RDRTDc = kron(Ri(:,j)', Bi');
            Dc_diff = - RDRTDc - RTDRDc;
            vech_Dc = Dc_diff(:,vech_ind)*wi;
            gi((ind_c(j)+1):ind_c(j+1)) = 2.*(vech_Dc*Resi);
        end
        vIsig2 = vech(tril(sig2*eye(mi)));
        vIsig2 = wi*-vIsig2;
        gi(end) = 2.*(sum(Resi.*vIsig2));
        g = g+gi/mi;
    end


end

fullR = mat2cell(repmat(P, r,1),repelem(p,r), p);
fullR = blkdiag(fullR{:});
f = f + lambda.*cvec(1:end-1)'*fullR*cvec(1:end-1);
if nargout > 1
    g(1:end-1)  = g(1:end-1) + lambda.*(2.*fullR)*cvec(1:end-1);
end

end

%Loss function 
function [f, g]= loss_LLK(cvec, SiCell, data, r, p, basis, P, lambda)
n = length(unique(data(:,1)));
indB = cumsum([1, repelem(p,r)]);
f=0;
g = zeros(p*r+1,1);

for i = 1:n
    iind = find(data(:,1) == i);
    mi = length(iind);
    ti = data(iind,2);
    Bi = eval_basis(ti, basis);
    Ri = zeros(mi, r);
    for j = 1:r
        C = cvec(indB(j):j*p);
        Ri(:,j) = Bi*C;
    end
    sig2 = exp(cvec(end));
    Sihat = Ri*Ri' + sig2*eye(mi);
    f = f + (trace((Sihat)\SiCell{i}) + log(det(Sihat)));

    if nargout > 1
        gi = zeros(p*r+1,1);
        ind_c = cumsum([0, repelem(p,r)]);
        indBr  = repelem(1:r, p);
        Dn = repelem(p,p*r);
        for j = 1:p*r
            Dcvec = zeros(Dn(j),1);
            Dcvec(j-(ind_c(indBr(j)))) = 1;
            ind_1 = 1:mi;

            RTDRDc = Ri(ind_1,indBr(j))*(Dcvec')*(Bi');
            RDRTDc = Bi*(Dcvec)*(Ri(ind_1,indBr(j))');
            DRRTDc = RTDRDc + RDRTDc;
            
            term1 = -trace( (topdm(Sihat)\DRRTDc*(topdm(Sihat)\SiCell{i})) );
            term2 = (trace(topdm(Sihat)\DRRTDc));

            gi(j) = term1+term2;
        end
        gi(end) = (-trace((topdm(Sihat)\(sig2*eye(mi))*(topdm(Sihat)\SiCell{i}))) ...
            + trace(topdm(Sihat)\(sig2*eye(mi))));
        g = g+gi;
    end

end

fullR = mat2cell(repmat(P, r,1),repelem(p,r), p);
fullR = blkdiag(fullR{:});
f = f + lambda.*cvec(1:end-1)'*fullR*cvec(1:end-1);
if nargout > 1
    g(1:end-1)  = g(1:end-1) + lambda.*(2.*fullR)*cvec(1:end-1);
end
end


function R = get_R(cvec, basis, M, r,p)
R = zeros(M, r);
B = eval_basis(linspace(0,1,M), basis);
indB = cumsum([1, repelem(p,r)]);
for i = 1:r
    C = cvec(indB(i):i*p);
    R(:,i) = B*C;
end
end

%Aux functions
function vech = vech(mat)
    n = size(mat, 2);
    vech = mat(tril(ones(n),0)==1);
end





function [trH, trH2, DtrH] = traceH(data, basis, R, l, r)
BB = [];
n = length(unique(data(:,1)));
m2 = 0;
for i = 1:n
    iind = find(data(:,1) == i);
    mi = length(iind);
    ti = data(iind,2);
    Bi = eval_basis(ti, basis);
    BB = [BB;Bi];
    m2 = m2 + mi;
end
H = BB*((BB'*BB + l.*R)\(BB'));
trH = trace(H)*r;
trH2 = trace(H^2)*r;
DtrH = trH-trH2;
end


function [Shat] = get_Shat(cvec, data,basis, r, p)
Shat = [];
indB = cumsum([1, repelem(p,r)]);
n = length(unique(data(:,1)));
for i = 1:n
    iind = find(data(:,1) == i);
    mi = length(iind);
    ti = data(iind,2);
    Bi = eval_basis(ti, basis);
    Ri = zeros(mi, r);
    for j = 1:r
        C = cvec(indB(j):j*p);
        Ri(:,j) = Bi*C;
    end
%     sig2 = exp(cvec(end));
    Sihat = Ri*Ri';% + sig2*eye(mi);
    Shat = [Shat;vech(Sihat)];
end
end





