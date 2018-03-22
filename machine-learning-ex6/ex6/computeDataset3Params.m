function [C sigma] = computeDataset3Params(X, y, Xval, yval)

  sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';
  c_vec = [0.01 0.03 0.1 0.3 1 3 10 30]';

  error = zeros(size(sigma_vec,1) * size(c_vec, 1), 3);
  error_index = 1;

  for i = 1:length(sigma_vec)
    for j = 1:length(c_vec)

      sigma_current = sigma_vec(i);
      c_current = c_vec(j);

      model= svmTrain(X, y, c_current, @(x1, x2) gaussianKernel(x1, x2, sigma_current));

      predictions = svmPredict(model, Xval);

      error(error_index, 1) = sigma_current;
      error(error_index, 2) = c_current;
      error(error_index,3) = mean(double(predictions ~= yval));

      error_index = error_index + 1;
    end
  end
  disp(error);
  [e i] = min(error(:,3));

%  disp(error(i,3));
%  disp(error(i,2));
%  disp(error(i,1));

  C = error(i,2);
  sigma = error(i,1);
