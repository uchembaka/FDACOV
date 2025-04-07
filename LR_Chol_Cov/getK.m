function K = getK(S, err, DF)
if nargin < 3
    DF = 0;
end

if err
    col1 =  S(:,1)./sqrt(S(1,1)); %First column of L
    col1 = col1(2:end); %Exclude first point (it includes sig2)
    col1 = ((col1 - min(col1)) / (max(col1) - min(col1))) * (1 - 0) + 0; %scale 
    M = length(col1);
    eval_pts = linspace(0,1, M);
    wk_pts = ~isnan(col1);

    % basis = create_bspline_basis([0,1],max(15, round(0.5*M)));
    basis = create_bspline_basis([0,1], round(0.75*M));
    % lambdaVec = 10.^(-6:1);
    lambdaVec = exp(-12:0);
    nl = length(lambdaVec);
    gcv_vec = zeros(nl,1);
    for i = 1:nl
        tmp_fdPar = fdPar(basis, 2, lambdaVec(i));
        [~, ~, gcv] = smooth_basis(eval_pts(wk_pts), col1(wk_pts), tmp_fdPar);
        gcv_vec(i) = gcv;
    end
    lam = lambdaVec(gcv_vec == min(gcv_vec));
    fdParobj = fdPar(basis, 2, lam);
    [fdobj, df] = smooth_basis(eval_pts(wk_pts), col1(wk_pts), fdParobj);

    if DF
        K = ceil(df);
    else
        smth_col1 = eval_basis(eval_pts, basis)*getcoef(fdobj);
        
        knots = 5:0.5*M;
        diff_vec = zeros(length(knots), 1);
        for i = 1:length(knots)
            basis = create_bspline_basis([0,1],knots(i));
            fdParobj = fdPar(basis, 2, 0);
            fdobj = smooth_basis(eval_pts, col1, fdParobj);
            col1_est = eval_basis(eval_pts, basis)*getcoef(fdobj);
            diff_rmse = rmse(smth_col1, col1_est, 'all');
            diff_vec(i) = diff_rmse;
        end
        K = knots(diff_vec == min(diff_vec));
        K = K(1);
    end
else
    col1 =  S(:,1)./sqrt(S(1,1));
    col1 = ((col1 - min(col1)) / (max(col1) - min(col1))) * (1 - 0) + 0;
    M = length(col1);
    eval_pts = linspace(0,1, M);
    wk_pts = ~isnan(col1);
    basis = create_bspline_basis([0,1], M);
    lam = 0;
    fdParobj = fdPar(basis, 2, lam);
    fdobj = smooth_basis(eval_pts(wk_pts), col1(wk_pts), fdParobj);

    smth_col1 = eval_basis(eval_pts, basis)*getcoef(fdobj);
    
    knots = 5:0.5*M;
    diff_vec = zeros(length(knots), 1);
    prev_mrsse = 0; omega = 0.0001;
    for i = 1:length(knots)
        basis = create_bspline_basis([0,1],knots(i));
        fdParobj = fdPar(basis, 2, 0);
        fdobj = smooth_basis(eval_pts, col1, fdParobj);
        col1_est = eval_basis(eval_pts, basis)*getcoef(fdobj);
        diff_vec(i) = rmse(smth_col1, col1_est, 'all');
        curr_mrsse = diff_vec(i);
        if abs(prev_mrsse - curr_mrsse) <= omega || i == length(knots)
            K = knots(i);
            break
        else
            prev_mrsse = diff_vec(i);
        end 
    end
end

end