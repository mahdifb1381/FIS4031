%% Step 1: Loading and Preprocessing the UCI Air Quality Dataset
% Load the UCI Air Quality dataset
data = readtable('AirQualityUCI.csv', 'VariableNamingRule', 'preserve');

% Explore the data
disp(head(data));
disp('Data Dimensions:'); disp(size(data));

% Identify numeric and non-numeric columns
numericColumns = vartype('numeric');
nonNumericColumns = ~varfun(@isnumeric, data, 'OutputFormat', 'uniform');

% Replace -200 (missing value indicator) with NaN in numeric columns only
data{:, numericColumns}(data{:, numericColumns} == -200) = NaN;

% Remove rows with missing data across all columns
data = rmmissing(data);

% Extract numeric columns and normalize them (Min-Max Scaling)
numericData = data{:, numericColumns};
normData = normalize(numericData, 'range');

% Rebuild the table: combine non-numeric columns with normalized numeric data
nonNumericData = data(:, nonNumericColumns); % Select non-numeric columns
normNumericData = array2table(normData, 'VariableNames', data.Properties.VariableNames(numericColumns));
processedData = [nonNumericData, normNumericData];

disp('Data preprocessing complete. Processed data preview:');
disp(head(processedData));

%% Step 2: Splitting the Dataset
% Identify numeric columns in processedData
numericColumns = varfun(@isnumeric, processedData, 'OutputFormat', 'uniform');

% Extract numeric data from processedData
numericData = processedData{:, numericColumns}; % All numeric columns

% Assume the last numeric column is the target variable (Y)
X = numericData(:, 1:end-1); % All numeric columns except the last
Y = numericData(:, end);     % Last numeric column is the target variable

disp('Features (X) and target variable (Y) extracted successfully.');
disp(['Size of X: ', num2str(size(X))]);
disp(['Size of Y: ', num2str(size(Y))]);

% Step 2: Splitting the Dataset
% Set random seed for reproducibility
rng(13);

% Split the dataset into training, validation, and testing sets
numSamples = size(X, 1); % Total number of samples
indices = randperm(numSamples); % Shuffle the indices

% Define split indices
trainIdx = indices(1:round(0.6 * numSamples));          % 60% training
valIdx = indices(round(0.6 * numSamples) + 1:round(0.8 * numSamples)); % 20% validation
testIdx = indices(round(0.8 * numSamples) + 1:end);     % 20% testing

% Create training, validation, and testing sets
X_train = X(trainIdx, :); Y_train = Y(trainIdx);
X_val = X(valIdx, :);     Y_val = Y(valIdx);
X_test = X(testIdx, :);   Y_test = Y(testIdx);

% Display the sizes of the splits
disp('Dataset splitting complete.');
disp(['Training set size: ', num2str(size(X_train, 1))]);
disp(['Validation set size: ', num2str(size(X_val, 1))]);
disp(['Testing set size: ', num2str(size(X_test, 1))]);

%% Step 3: Implementing the RBF Neural Network

% Step 3: RBF Network Implementation
% Create RBF network with predefined spread and number of neurons
spread = 1; % Hyperparameter for the RBF kernel
goal = 0.01; % Mean squared error goal
maxNeurons = 50;

% Train RBF network using the newrb function
rbf_net = newrb(X_train', Y_train', goal, spread, maxNeurons);

% Predict on test data
Y_pred_rbf = sim(rbf_net, X_test')';

% Calculate performance metrics for RBF
mse_rbf = mean((Y_test - Y_pred_rbf).^2); % Mean Squared Error
rmse_rbf = sqrt(mse_rbf); % Root Mean Squared Error
r2_rbf = 1 - sum((Y_test - Y_pred_rbf).^2) / sum((Y_test - mean(Y_test)).^2); % R-squared

disp(['RBF Network - MSE: ', num2str(mse_rbf), ', RMSE: ', num2str(rmse_rbf), ', R^2: ', num2str(r2_rbf)]);

%% Step 4: Implementing the ANFIS Model

% Step 4: ANFIS Implementation using genfis2
% Combine training data for ANFIS
trainData = [X_train Y_train];

% Generate FIS using genfis2
radiuses = 0.5; % Radius for subtractive clustering
fis = genfis2(X_train, Y_train, radiuses);

% Train FIS using ANFIS
numEpochs = 100; % Number of training epochs
anfis_model = anfis(trainData, fis, numEpochs);

% Predict on test data
Y_pred_anfis = evalfis(X_test, anfis_model);


% Calculate performance metrics for ANFIS
mse_anfis = mean((Y_test - Y_pred_anfis).^2); % Mean Squared Error
rmse_anfis = sqrt(mse_anfis); % Root Mean Squared Error
r2_anfis = 1 - sum((Y_test - Y_pred_anfis).^2) / sum((Y_test - mean(Y_test)).^2); % R-squared

disp(['ANFIS - MSE: ', num2str(mse_anfis), ', RMSE: ', num2str(rmse_anfis), ', R^2: ', num2str(r2_anfis)]);

%% Step 5: Comparing Model Performance

% Displaying performance comparison
disp('Model Performance Comparison:');
disp(['RBF Network - MSE: ', num2str(mse_rbf), ', RMSE: ', num2str(rmse_rbf), ', R^2: ', num2str(r2_rbf)]);
disp(['ANFIS - MSE: ', num2str(mse_anfis), ', RMSE: ', num2str(rmse_anfis), ', R^2: ', num2str(r2_anfis)]);

% Plot predicted vs actual for both models
figure;
subplot(1, 2, 1);
scatter(Y_test, Y_pred_rbf);
xlabel('Actual');
ylabel('Predicted');
title('RBF Network: Actual vs Predicted');

subplot(1, 2, 2);
scatter(Y_test, Y_pred_anfis);
xlabel('Actual');
ylabel('Predicted');
title('ANFIS: Actual vs Predicted');

% Optionally, display error plots
figure;
subplot(1, 2, 1);
plot(Y_test - Y_pred_rbf);
title('RBF Network Prediction Error');
xlabel('Sample Index');
ylabel('Error');

subplot(1, 2, 2);
plot(Y_test - Y_pred_anfis);
title('ANFIS Prediction Error');
xlabel('Sample Index');
ylabel('Error');
