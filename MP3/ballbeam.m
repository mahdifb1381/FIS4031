% بارگذاری داده‌های توپ و تیر
bb = load('ballbeam.dat');

U = bb(:,1);  % Input - beam angle
Y = bb(:,2);  % Output - ball position

% Data Preprocessing
% 1. Normalize data
U_norm = (U - min(U)) / (max(U) - min(U));
Y_norm = (Y - min(Y)) / (max(Y) - min(Y));

% 2. Remove outliers using zscore
z_scores = zscore(U_norm);
outliers = abs(z_scores) > 3;
U_clean = U_norm(~outliers);
Y_clean = Y_norm(~outliers);

% 3. Split data for training and checking
data_length = length(U_clean);
train_idx = 1:2:data_length;
check_idx = 2:2:data_length;

trn_data = [U_clean(train_idx) Y_clean(train_idx)];
chk_data = [U_clean(check_idx) Y_clean(check_idx)];

% ANFIS Parameters
numMFs = 5;         % Increased number of MFs
mfType = 'gbellmf'; % Changed to generalized bell MF
epoch_n = 200;      % Increased epochs

% Generate FIS with improved settings
opt = genfisOptions('GridPartition');
opt.NumMembershipFunctions = numMFs;
opt.InputMembershipFunctionType = mfType;
in_fis = genfis(trn_data(:,1), trn_data(:,2), opt);

% Training options
anfis_opt = anfisOptions;
anfis_opt.InitialFIS = in_fis;
anfis_opt.EpochNumber = epoch_n;
anfis_opt.ValidationData = chk_data;
anfis_opt.OptimizationMethod = 1;  % Hybrid optimization
anfis_opt.DisplayANFISInformation = 1;
anfis_opt.DisplayErrorValues = 1;
anfis_opt.DisplayStepSize = 1;
anfis_opt.DisplayFinalResults = 1;

% Train ANFIS with options
[fis, trainError, stepSize, chkFIS, chkError] = anfis(trn_data, anfis_opt);

% Denormalize and evaluate
U_test = (U - min(U)) / (max(U) - min(U));
y_pred_norm = evalfis(U_test, fis);
y_pred = y_pred_norm * (max(Y) - min(Y)) + min(Y);

% Plot results
t = 0:0.1:(length(U)-1)*0.1;

figure('Name', 'Training Results');
subplot(2,1,1);
plot(trainError, 'b-', 'LineWidth', 1.5);
hold on;
plot(chkError, 'r--', 'LineWidth', 1.5);
title('ANFIS Training and Checking Error');
xlabel('Epoch');
ylabel('Error');
legend('Training Error', 'Checking Error');
grid on;

subplot(2,1,2);
plot(t, Y, 'b', t, y_pred, 'r--');
title('System Output vs ANFIS Prediction');
xlabel('Time (s)');
ylabel('Ball Position');
legend('Actual', 'ANFIS Prediction');
grid on;

% Calculate performance metrics
MSE = mean((Y - y_pred).^2);
RMSE = sqrt(MSE);
R2 = 1 - sum((Y - y_pred).^2) / sum((Y - mean(Y)).^2);

fprintf('\nPerformance Metrics:\n');
fprintf('MSE: %.4f\n', MSE);
fprintf('RMSE: %.4f\n', RMSE);
fprintf('R2: %.4f\n', R2);

% Plot MFs
figure('Name', 'Membership Functions');
subplot(2,1,1);
plotmf(fis, 'input', 1);
title('Input Membership Functions');

% Surface view
figure('Name', 'Rule Surface');
gensurf(fis);
title('ANFIS Rule Surface');