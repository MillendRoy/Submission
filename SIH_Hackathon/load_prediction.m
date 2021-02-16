%% PART 1 : - '''DATA PREPROCESSING'''
%DATA IMPORTING and VISUALIZATION
data=readtable("Germany_raw_data.xlsx");
dataset=[data{87650:end,[2 5 9 13]}]; %price temp wind_speed load


figure
plot(dataset(1:24,end))
xlabel("Time (in hrs)")
ylabel("Actual Load Demand (in *10 W)")
title("Actual load Consumption-----Dataset Visualization")
hold on
plot(dataset(25:48,end))
hold on
plot(dataset(49:72,end))
legend(["DAY 1" "DAY 2" "DAY 3"])


% TAKING CARE OF MISSING DATA
% Since there are many missing data, we need to impute
% But normal statistical imputation (mean/median) may lead to wrong results
% Hence spline interpolation used for the purpose.
index=[];
time=[];
price=[];
for i= 1:17543
        if(isnan(dataset(i,1)))
            index=[index i];
        else
            time=[time i];
            price=[price dataset(i,1)];
        end
end
price_pol=spline(time,price,index);

%replace back to the nan values
for i=1:17543
    if(ismember(i,index))
        dataset(i,1)=price_pol(1,find(index==i));
    end
end

% SPLITTING the DATASET into TRAINING and TEST set
num_Train = floor(0.9*numel(dataset)/4);
train = dataset(1:num_Train+3,:);
test = dataset(num_Train+4:end,:);
num_Train=numel(train)/4;
num_Test=numel(test)/4;

% FEATURE SCALING- Standardization used to scale the values to 0 mean and unit variance
avg = mean(train);
dev = std(train);
sc_train = (train - avg) ./ dev;

% Defining the features and labels
X_train=[];
k=1;
l=1;
for i=1:24:num_Train
    for j=i:i+23
        X_train(k,l)=sc_train(j,1);
        l=l+1;
    end
    k=k+1;
    l=1;
end
k=1;
l=25;
for i=1:24:num_Train
    for j=i:i+23
        X_train(k,l)=sc_train(j,2);
        l=l+1;
    end
    k=k+1;
    l=25;
end
k=1;
l=49;
for i=1:24:num_Train
    for j=i:i+23
        X_train(k,l)=sc_train(j,3);
        l=l+1;
    end
    k=k+1;
    l=49;
end
k=1;
l=73;
for i=1:24:num_Train
    for j=i:i+23
        X_train(k,l)=sc_train(j,4);
        l=l+1;
    end
    k=k+1;
    l=73;
end

X_train=X_train.';
y_train=X_train(73:end,2:end);
X_train=X_train(:,1:end-1);

idx = randperm(size(X_train,2),66);
X_validation = X_train(:,idx);
X_train(:,idx) = [];
y_validation = y_train(:,idx);
y_train(:,idx) = [];


%% PART 2 :- '''ARCHITECTURE CREATION for RNN (LSTM Networks)'''
input_size= 96;
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
    'MaxEpochs',10, ...
    'MiniBatchSize',113, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',226, ...
    'LearnRateDropFactor',0.2, ...
    'Verbose',0, ...
    'ValidationData',{X_validation,y_validation},...
    'ValidationFrequency',2,...
    'ValidationPatience',3,...
    'Plots','training-progress');

% training the LSTM network
%nw = trainNetwork(X_train,y_train,layers,options);
%load the pre-trained network
load load_net
nw = load_net;

%analyze the network
analyzeNetwork(nw)

%% PART 4 :- '''PREDICTION Time'''
% Applying the same feature scaling
sc_test = (test - avg) ./ dev;

X_test=[];
k=1;
l=1;
for i=1:24:num_Test
    for j=i:i+23
        X_test(k,l)=sc_test(j,1);
        l=l+1;
    end
    k=k+1;
    l=1;
end
k=1;
l=25;
for i=1:24:num_Test
    for j=i:i+23
        X_test(k,l)=sc_test(j,2);
        l=l+1;
    end
    k=k+1;
    l=25;
end
k=1;
l=49;
for i=1:24:num_Test
    for j=i:i+23
        X_test(k,l)=sc_test(j,3);
        l=l+1;
    end
    k=k+1;
    l=49;
end
k=1;
l=73;
for i=1:24:num_Test
    for j=i:i+23
        X_test(k,l)=sc_test(j,4);
        l=l+1;
    end
    k=k+1;
    l=73;
end

X_test=X_test.';
y_test=X_test(73:end,2:end);
X_test=X_test(:,1:end-1);

%predict the first price using the last train data
nw = predictAndUpdateState(nw,X_train);
[nw,y_pred] = predictAndUpdateState(nw,X_test);

%load_net = nw;
%save load_net

y_train=y_train.';
y_test=y_test.';
y_pred=y_pred.';
%unstandardize the data
y_pred = dev(4).*y_pred + avg(4);
y_test = dev(4)*y_test + avg(4);
y_train = dev(4)*y_train + avg(4);

%% PART 5 :- '''EVALUATION'''
y_Pred=[];
for i=1:72
    for j=1:24
        y_Pred=[y_Pred;y_pred(i,j)];
    end
end

y_Test=[];
for i=1:72
    for j=1:24
        y_Test=[y_Test;y_test(i,j)];
    end
end
rmse = sqrt(mean((y_Pred-y_Test).^2));

%% PART 6 :- '''VISUALIZATION'''
figure
subplot(2,1,1)
plot(train(end-100:end,4))
hold on
plot(101:196,[train(end,4); y_Pred(1:95,1)],'.-')
hold off
xlabel("Time (in hrs)")
ylabel("Load Demand (in *10 W)")
title("Forecast")
legend(["Observed" "Forecast"])

subplot(2,1,2)
plot(y_Test(1:96,1))
hold on
plot(y_Pred(1:96,1),'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Load Demand (in *10 W)")
title("Actual vs Forecast")

accuracy=100-(rmse*100/70000);
extra_load=y_Test-y_Pred;
save extra_load;
y_test_load=y_Test;
y_pred_load=y_Pred;
save y_test_load;
save y_pred_load;

da='2016-12-01';
load date;
numdays = datenum(date) - datenum(da)+1;
text_real=sprintf('load_real24_%s.xls',date);
text_forecast=sprintf('load_forecast24_%s.xls',date);
xlswrite(text_real,[y_Test(end-743+(numdays*24):end-743+24+(numdays*24),1)].')
xlswrite(text_forecast,[y_Test(end-744-23+(numdays*24):end-744+(numdays*24),1); y_Pred(end-743+(numdays*24):end-743+24+(numdays*24),1)].')  