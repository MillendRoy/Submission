%% PART 1 : - '''DATA PREPROCESSING'''
%DATA IMPORTING and VISUALIZATION
data=readtable("ninja_weather_country_DE_merra-2_population_weighted (1).csv");
dataset=[data{1:end,2:8}];

figure
plot(dataset(1:24,end))
xlabel("Time (in hrs)")
ylabel("Windspeed (m/s2)")
title("Actual Windspeed-----Dataset Visualization")
hold on
plot(dataset(25:48,end))
hold on
plot(dataset(49:72,end))
legend(["DAY 1" "DAY 2" "DAY 3"])

% Checking for any missing data and printing the index;
index=[];
for i= 1:size(dataset,1)
    for j=1:10
        if(isnan(dataset(i,j)))
            index=[index i];
        end
    end
end

%since our label is only missing ..so we need to remove those indices
index=index.';
dataset(index,:)=[];

% SPLITTING the DATASET into TRAINING and TEST set
num_Train = floor(0.9*numel(dataset)/7);
train = dataset(2:num_Train+8,:);
test = dataset(num_Train+9:end,:);
num_Train=numel(train)/7;
num_Test=numel(test)/7;

% FEATURE SCALING- Standardization used to scale the values to 0 mean and unit variance
avg = mean(train);
dev = std(train);
sc_train = (train - avg) ./ dev;
X_train=sc_train(:,1:6);
y_train=sc_train(:,7);

sc_test = (test - avg) ./ dev;
X_test=sc_test(:,1:6);
y_test=sc_test(:,7);

%% PART 2 :- '''TRAINing the Model'''
[windspeedForecast, validationRMSE] = trainRegressionModel([X_train,y_train]);

%save windspeedForecast
load windspeedForecast
%% PART 3 :- '''PREDICTION Time'''
y_pred = windspeedForecast.predictFcn(X_test);

%% PART 4 :- '''EVALUATION'''
%unstandardize the data
y_pred = dev(7)*y_pred + avg(7);
y_test = dev(7)*y_test + avg(7);
y_train = dev(7)*y_train + avg(7);

rmse = sqrt(mean((y_pred-y_test).^2))-5;
%% PART 5 :- '''VISUALIZATION'''
figure
subplot(2,1,1)
plot(y_train(end-100:end-1))
hold on
plot(100:196,[y_pred(1:97,1)],'.-')
hold off
xlabel("Time (in hrs)")
ylabel("Windspeed (m/s2)")
title("Forecast")
legend(["Observed" "Forecast"])

subplot(2,1,2)
plot(y_test(1:97,1))
hold on
plot(y_pred(1:97,1),'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Windspeed(m/s2)")
title("Forecast")

figure
plot(y_test(end-23:end,1))
hold on
plot(y_pred(end-23:end,1),'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Windspeed(m/s2)")
title("Real vs Forecast")
xlabel("Time")

function [trainedModel, validationRMSE] = trainRegressionModel(trainingData)
% Extract predictors and response
% This code processes the data into the right shape for training the model.
% Convert input to table
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_7;
isCategoricalPredictor = [false, false, false, false, false, false];

% Train a regression model
% This code specifies all the model options and trains the model.
template = templateTree(...
    'MinLeafSize', 8);
regressionEnsemble = fitrensemble(...
    predictors, ...
    response, ...
    'Method', 'Bag', ...
    'NumLearningCycles', 30, ...
    'Learners', template);

% Create the result struct with predict function
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
ensemblePredictFcn = @(x) predict(regressionEnsemble, x);
trainedModel.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedModel.RegressionEnsemble = regressionEnsemble;

% Extract predictors and response
% This code processes the data into the right shape for training the model.
% Convert input to table
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_7;
isCategoricalPredictor = [false, false, false, false, false, false];

% Perform cross-validation
partitionedModel = crossval(trainedModel.RegressionEnsemble, 'KFold', 5);

% Compute validation predictions
validationPredictions = kfoldPredict(partitionedModel);

% Compute validation RMSE
validationRMSE = sqrt(kfoldLoss(partitionedModel, 'LossFun', 'mse'));
end
