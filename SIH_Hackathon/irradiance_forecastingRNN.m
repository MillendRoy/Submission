%% PART 1 : - '''DATA PREPROCESSING'''
%DATA IMPORTING and VISUALIZATION
data=readtable("ninja_weather_country_DE_merra-2_population_weighted (1).csv");
dataset=[data{1:end,2:10}];
for i=1:size(dataset,1)
    dataset(i,10)=dataset(i,8)+dataset(i,9);
end
irradiance=dataset(:,10);
figure
plot(irradiance(1:24,end))
xlabel("Time (in hrs)")
ylabel("Irradiance (in W/m2)")
title("Actual Solar Irradiance-----Dataset Visualization")
hold on
plot(irradiance(25:48,end))
hold on
plot(irradiance(49:72,end))
legend(["DAY 1" "DAY 2" "DAY 3"])

% TAKING CARE OF MISSING DATA
% Since there are many missing data, we need to impute
% But normal statistical imputation (mean/median) may lead to wrong results
% Hence spline interpolation used for the purpose.
[m,n]=size(irradiance);
count=0;%helps to find no.of nan values
index=[];% stores the indices where we need to interpolate
time=[];
price=[];
for i=1:m
    if(isnan(irradiance(i,1)))
        count=count+1;
        index=[index i];
    else
        time=[time i];
        price=[price irradiance(i,1)];
    end
end
price_pol=spline(time,price,index);

%replace back to the nan values
for i=1:m
    if(ismember(i,index))
        irradiance(i,1)=price_pol(1,find(index==i));
    end
end

% SPLITTING the DATASET into TRAINING and TEST set
num_Train = floor(0.9*numel(irradiance));
train = irradiance(2:num_Train+8,:);
test = irradiance(num_Train+9:end,:);
num_Train=numel(train);
num_Test=numel(test);

% FEATURE SCALING- Standardization used to scale the values to 0 mean and unit variance
avg = mean(train);
dev = std(train);
sc_train = (train - avg) / dev;

% Defining the features and labels
X_train=[];
k=1;
l=1;
for i=1:24:num_Train
    for j=i:i+23
        X_train(k,l)=sc_train(j);
        l=l+1;
    end
    k=k+1;
    l=1;
end
X_train=X_train.';
y_train=X_train(:,2:end);
X_train=X_train(:,1:end-1);
idx = randperm(size(X_train,2),780);
X_validation = X_train(:,idx);
X_train(:,idx) = [];
y_validation = y_train(:,idx);
y_train(:,idx) = [];


%% PART 2 :- '''ARCHITECTURE CREATION for RNN (LSTM Networks)'''
input_size= 24;
output_size = 24;
hidden_units_size = 200;

layers = [...
    sequenceInputLayer(input_size)
    lstmLayer(hidden_units_size,'OutputMode','sequence')
    dropoutLayer(0.2)
    fullyConnectedLayer(output_size)
    regressionLayer]

%% PART 3 :- '''TRAINing the Model'''
%specifying the training options
options = trainingOptions('adam',... 
    'MaxEpochs',50, ...
    'MiniBatchSize',113, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',226, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'ValidationData',{X_validation,y_validation},...
    'ValidationFrequency',10,...
    'ValidationPatience',5,...
    'Plots','training-progress');

% training the LSTM network
%nw = trainNetwork(X_train,y_train,layers,options);
%load the pre-trained network
load irradiance_net
nw = irradiance_net;

%analyze the network
analyzeNetwork(nw)

%% PART 4 :- '''PREDICTION Time'''
% Applying the same feature scaling
sc_test = (test - avg) / dev;

X_test=[];
k=1;
l=1;
for i=1:24:num_Test
    for j=i:i+23
        X_test(k,l)=sc_test(j);
        l=l+1;
    end
    k=k+1;
    l=1;
end
X_test=X_test.';
y_test=X_test(:,2:end);
X_test=X_test(:,1:end-1);

%predict the first price using the last train data
nw = predictAndUpdateState(nw,X_train);
% [nw,y_pred] = predictAndUpdateState(nw,y_train(:,end));

%predict the next set of data
% for i = 2:size(X_test,2)
%     [nw,y_pred(:,i)] = predictAndUpdateState(nw,y_pred(:,i-1),'ExecutionEnvironment','cpu');
% end
[nw,y_pred] = predictAndUpdateState(nw,X_test);
%unstandardize the data
y_pred = dev*y_pred + avg;
y_test = dev*y_test + avg;

%% PART 5 :- '''EVALUATION'''
y_test=y_test.';
y_pred=y_pred.';
y_Pred=[];
for i=1:437
    for j=1:24
        y_Pred=[y_Pred;y_pred(i,j)];
    end
end

y_Test=[];
for i=1:437
    for j=1:24
        y_Test=[y_Test;y_test(i,j)];
    end
end
rmse = sqrt(mean((y_Pred-y_Test).^2));
mae=mean(abs(y_Test-y_Pred));
%% PART 6 :- '''VISUALIZATION'''
figure
subplot(2,1,1)
plot(train(end-100:end-1))
hold on
plot(100:196,[irradiance(num_Train);y_Pred(1:96,1)],'.-')
hold off
xlabel("Time (in hrs)")
ylabel("Solar Irradiance (W/m2)")
title("Forecast")
legend(["Observed" "Forecast"])

subplot(2,1,2)
plot(y_Test(1:96,1))
hold on
plot(y_Pred(1:96,1),'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Solar Irradiance (W/m2)")
title("Forecast")

figure
plot(y_Test(end-23:end,1))
hold on
plot(y_Pred(end-23:end,1),'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Solar Irradiance (W/m2)")
title("Real vs Forecast")
xlabel("Time")

irradiance_net = nw;
%save irradiance_net
temp=[data{end-10487:end-24,3}];
da='2016-12-01';
date='2016-12-30';
load date;
numdays = datenum(date) - datenum(da)+1
text_real=sprintf('irradiance_real_%s.xls',date);
text_forecast=sprintf('irradiance_forecast_%s.xls',date);
xlswrite(text_real,[y_Test(1:end-24,:)])
xlswrite('irradiance_forecast.xls',[data{end-263:end,3} y_Pred(end-263:end,:)])       
        