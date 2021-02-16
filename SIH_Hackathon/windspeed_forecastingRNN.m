%% PART 1 : - '''DATA PREPROCESSING'''
%DATA IMPORTING and VISUALIZATION
data=readtable("ninja_weather_country_DE_merra-2_population_weighted (1).csv");
windspeed=[data{43827:end,[2 3 4 5 6 7 8]}];

figure
plot(windspeed(1:24,end))
xlabel("Time (in hrs)")
ylabel("Windspeed (m/s)")
title("Actual Windspeed-----Dataset Visualization")
hold on
plot(windspeed(25:48,end))
hold on
plot(windspeed(49:72,end))
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
num_Train = floor(0.9*numel(windspeed)/7);
train = windspeed(1:num_Train-6,:);
test = windspeed(num_Train-6:end,:);
num_Train=numel(train)/7;
num_Test=numel(test)/7;

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
k=1;
l=97;
for i=1:24:num_Train
    for j=i:i+23
        X_train(k,l)=sc_train(j,5);
        l=l+1;
    end
    k=k+1;
    l=97;
end
k=1;
l=121;
for i=1:24:num_Train
    for j=i:i+23
        X_train(k,l)=sc_train(j,6);
        l=l+1;
    end
    k=k+1;
    l=121;
end
k=1;
l=145;
for i=1:24:num_Train
    for j=i:i+23
        X_train(k,l)=sc_train(j,7);
        l=l+1;
    end
    k=k+1;
    l=145;
end
X_train=X_train.';
y_train=X_train(145:end,2:end);
X_train=X_train(:,1:end-1);

idx = randperm(size(X_train,2),300);
X_validation = X_train(:,idx);
X_train(:,idx) = [];
y_validation = y_train(:,idx);
y_train(:,idx) = [];


%% PART 2 :- '''ARCHITECTURE CREATION for RNN (LSTM Networks)'''
input_size= 168;
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
load windspeed_net
nw = windspeed_net;

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
k=1;
l=97;
for i=1:24:num_Test
    for j=i:i+23
        X_test(k,l)=sc_test(j,5);
        l=l+1;
    end
    k=k+1;
    l=97;
end
k=1;
l=121;
for i=1:24:num_Test
    for j=i:i+23
        X_test(k,l)=sc_test(j,6);
        l=l+1;
    end
    k=k+1;
    l=121;
end
k=1;
l=145;
for i=1:24:num_Test
    for j=i:i+23
        X_test(k,l)=sc_test(j,7);
        l=l+1;
    end
    k=k+1;
    l=145;
end
X_test=X_test.';
y_test=X_test(145:end,2:end);
X_test=X_test(:,1:end-1);

%predict the first price using the last train data
nw = predictAndUpdateState(nw,X_train);
% [nw,y_pred] = predictAndUpdateState(nw,y_train(:,end));

%predict the next set of data
% for i = 2:size(X_test,2)
%     [nw,y_pred(:,i)] = predictAndUpdateState(nw,y_pred(:,i-1),'ExecutionEnvironment','cpu');
% end
[nw,y_pred] = predictAndUpdateState(nw,X_test);

y_train=y_train.';
y_test=y_test.';
y_pred=y_pred.';
%unstandardize the data
y_pred = dev(7).*y_pred + avg(7);
y_test = dev(7)*y_test + avg(7);
y_train = dev(7)*y_train + avg(7);

%% PART 5 :- '''EVALUATION'''
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

%% PART 6 :- '''VISUALIZATION'''
figure
subplot(2,1,1)
plot(train(end-100:end-1))
hold on
plot(100:196,[windspeed(num_Train);y_Pred(1:96,1)],'.-')
hold off
xlabel("Time (in hrs)")
ylabel("Windspeed (m/s)")
title("Forecast")
legend(["Observed" "Forecast"])

subplot(2,1,2)
plot(y_Test(1:96,1))
hold on
plot(y_Pred(1:96,1),'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Windspeed(m/s)")
title("Forecast")

figure
plot(y_Test(end-263:end,1))
hold on
plot(y_Pred(end-263:end,1),'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Windspeed(m/s)")
title("Real vs Forecast")
xlabel("Time")
mae=mean(abs(y_Test-y_Pred));
%accuracy=100-mean((abs(y_Test-y_Pred).*100)./y_Test);

slice=[];
in1=[];
for i=1:(size(y_Pred,1)-24)
    if (y_Pred(i,1)>=2.9)&&(y_Pred(i,1)<=3.8)
        slice=[slice;y_Pred(i,1)];
        in1=[in1;i];
    end
end
in1=in1+94705;
in=[105170:105193]';
xlswrite('indices_slice.xls',[in1;in])
xlswrite('windspeed_slice.xls',[slice;y_Pred(end-23:end,:)])
windspeed_net = nw;
save windspeed_net
da='2016-12-01';
date='2016-12-30';
load date;
numdays = datenum(date) - datenum(da)+1;
text_real=sprintf('windspeed_real_%s.xls',date);
text_forecast=sprintf('windspeed_forecast_%s.xls',date);
xlswrite(text_real,[y_Test(1:end-24,:)])
xlswrite('windspeed_forecast.xls',[y_Pred(end-263:end,:)])        