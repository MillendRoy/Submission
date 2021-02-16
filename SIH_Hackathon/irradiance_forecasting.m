%% PART 1 : - '''DATA PREPROCESSING'''
%DATA IMPORTING and VISUALIZATION
data=readtable("ninja_weather_country_DE_merra-2_population_weighted (1).csv");
dataset=[data{1:end,2:10}];
for i=1:size(dataset,1)
    dataset(i,10)=dataset(i,8)+dataset(i,9);
end

figure
plot(dataset(1:24,end))
xlabel("Time (in hrs)")
ylabel("Irradiance (in W/m2)")
title("Actual Solar Irradiance-----Dataset Visualization")
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
% dataset(index,:)=[];

% SPLITTING the DATASET into TRAINING and TEST set
num_Train = floor(0.9*numel(dataset)/10);
train = dataset(2:num_Train+8,:);
test = dataset(num_Train+9:end,:);
num_Train=numel(train)/10;
num_Test=numel(test)/10;

% FEATURE SCALING- Standardization used to scale the values to 0 mean and unit variance
avg = mean(train);
dev = std(train);
sc_train = (train - avg) ./ dev;
X_train=sc_train(:,1:6);
y_train=sc_train(:,10);

sc_test = (test - avg) ./ dev;
X_test=sc_test(:,1:6);
y_test=sc_test(:,10);

%% PART 2 :- '''TRAINing the Model'''
[irradianceForecast, validationRMSE] = trainRegressionModel([X_train,y_train]);
%save irradianceForecast
load irradianceForecast
%% PART 3 :- '''PREDICTION Time'''
y_pred = irradianceForecast.predictFcn(X_test);

%% PART 4 :- '''EVALUATION'''
%unstandardize the data
y_pred = dev(10)*y_pred + avg(10);
y_test = dev(10)*y_test + avg(10);
y_train = dev(10)*y_train + avg(10);

rmse = sqrt(mean((y_pred-y_test).^2));
mae=mean(abs(y_test-y_pred));
%% PART 5 :- '''VISUALIZATION'''
figure
subplot(2,1,1)
plot(y_train(end-100:end-1))
hold on
plot(100:196,[y_pred(1:97,1)],'.-')
hold off
xlabel("Time (in hrs)")
ylabel("Solar Irradiance (W/m2)")
title("Forecast")
legend(["Observed" "Forecast"])

subplot(2,1,2)
plot(y_test(1:97,1))
hold on
plot(y_pred(1:97,1),'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Solar Irradiance (W/m2)")
title("Forecast")

figure
plot(y_test(end-23:end,1))
hold on
plot(y_pred(end-23:end,1),'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Solar Irradiance (W/m2)")
title("Real vs Forecast")
xlabel("Time")
%extra_solar=y_test-y_pred;
%save extra_solar;

function [trainedModel, validationRMSE] = trainRegressionModel(trainingData)
% Extract predictors and response
% This code processes the data into the right shape for training the model.
% Convert input to table
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_10;
isCategoricalPredictor = [false, false, false, false, false, false];

% Train a regression model
% This code specifies all the model options and trains the model.
template = templateTree(...
    'MinLeafSize', 4, ...
    'NumVariablesToSample', 5);
regressionEnsemble = fitrensemble(...
    predictors, ...
    response, ...
    'Method', 'Bag', ...
    'NumLearningCycles', 65, ...
    'Learners', template);

% Create the result struct with predict function
predictorExtractionFcn = @(x) array2table(x, 'VariableNames', predictorNames);
ensemblePredictFcn = @(x) predict(regressionEnsemble, x);
trainedModel.predictFcn = @(x) ensemblePredictFcn(predictorExtractionFcn(x));

% Add additional fields to the result struct
trainedModel.RegressionEnsemble = regressionEnsemble;
trainedModel.About = 'This struct is a trained model exported from Regression Learner R2019b.';
trainedModel.HowToPredict = sprintf('To make predictions on a new predictor column matrix, X, use: \n  yfit = c.predictFcn(X) \nreplacing ''c'' with the name of the variable that is this struct, e.g. ''trainedModel''. \n \nX must contain exactly 6 columns because this model was trained using 6 predictors. \nX must contain only predictor columns in exactly the same order and format as your training \ndata. Do not include the response column or any columns you did not import into the app. \n \nFor more information, see <a href="matlab:helpview(fullfile(docroot, ''stats'', ''stats.map''), ''appregression_exportmodeltoworkspace'')">How to predict using an exported model</a>.');

% Extract predictors and response
% This code processes the data into the right shape for training the model.
% Convert input to table
inputTable = array2table(trainingData, 'VariableNames', {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6', 'column_7', 'column_8', 'column_9', 'column_10'});

predictorNames = {'column_1', 'column_2', 'column_3', 'column_4', 'column_5', 'column_6'};
predictors = inputTable(:, predictorNames);
response = inputTable.column_10;
isCategoricalPredictor = [false, false, false, false, false, false];

% Perform cross-validation
partitionedModel = crossval(trainedModel.RegressionEnsemble, 'KFold', 5);

% Compute validation predictions
validationPredictions = kfoldPredict(partitionedModel);

% Compute validation RMSE
validationRMSE = sqrt(kfoldLoss(partitionedModel, 'LossFun', 'mse'));
end
