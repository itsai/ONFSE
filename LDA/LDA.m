function [M, lambda] = lda(X, labels)
%LDA Perform the LDA algorithm
%
%   [mappedX, mapping] = lda(X, labels, no_dims)
%
% The function runs LDA on a set of datapoints X. The variable
% no_dims sets the number of dimensions of the feature points in the 
% embedded feature space (no_dims >= 1, default = 2). The maximum number 
% for no_dims is the number of classes in your data minus 1. 
% The function returns the coordinates of the low-dimensional data in 
% mappedX. Furthermore, it returns information on the mapping in mapping.
%
%

% This file is part of the Matlab Toolbox for Dimensionality Reduction v0.7b.
% The toolbox can be obtained from http://ticc.uvt.nl/~lvdrmaaten
% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you 
% maintain the name of the original author.
%
% (C) Laurens van der Maaten
% Tilburg University, 2008

        
        % Make sure data is zero mean
    mapping.mean = mean(X, 1);
        X = bsxfun(@minus, X, mapping.mean);
    % X  = bsxfun(@rdivide, X, sqrt(var(X)));
        
        % Make sure labels are nice
        [classes, bar, labels] = unique(labels);
    nc = length(classes);
        
        % Intialize Sw
        Sw = zeros(size(X, 2), size(X, 2));
    
    % Compute total covariance matrix
    St = cov(X);

        % Sum over classes
        for i=1:nc
        
        % Get all instances with class i
        cur_X = X(labels == i,:);

                % Update within-class scatter
                C = cov(cur_X);
                p = size(cur_X, 1) / (length(labels) - 1);
                Sw = Sw + (p * C);
    end
    
    % Compute between class scatter
    Sb       = St - Sw;
    Sb(isnan(Sb)) = 0; Sw(isnan(Sw)) = 0;
        Sb(isinf(Sb)) = 0; Sw(isinf(Sw)) = 0;
    
    % Make sure not to embed in too high dimension
    
        % Perform eigendecomposition of inv(Sw)*Sb
    [M, lambda] = eig(Sb, Sw);