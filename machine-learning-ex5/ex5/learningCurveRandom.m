function [error_train, error_val] = ...
    learningCurveRandom(X, y, Xval, yval, lambda)

loops_nr = 50;

% Number of training examples
m = size(X, 1);
mval = size(Xval,1);

% You need to return these values correctly
error_train = zeros(m, 1);
error_val   = zeros(m, 1);

for i = 1:m
  error_train_loop = zeros(loops_nr, 1);
  error_val_loop = zeros(loops_nr, 1);
  for j = 1:loops_nr
    random_rows_train = randperm(m);
    random_rows_val = randperm(mval);

    X_train_rand = X(random_rows_train(1:i), :);
    X_val_rand = Xval(random_rows_val(1:i), :);
    y_train_rand = y(random_rows_train(1:i), :);
    y_val_rand = yval(random_rows_val(1:i), :);

    [theta] = trainLinearReg([ones(i, 1) X_train_rand], y_train_rand, lambda);
    [J_train, grad] = linearRegCostFunction([ones(i, 1) X_train_rand], y_train_rand, theta, 0);
    error_train_loop(j) = J_train;
    [J_val, grad] = linearRegCostFunction([ones(i, 1) X_val_rand], y_val_rand, theta, 0);
    error_val_loop(j) = J_val;
  end
    error_train(i) = sum(error_train_loop) ./ loops_nr;
    error_val(i) = sum(error_val_loop) ./ loops_nr;
end;

end
