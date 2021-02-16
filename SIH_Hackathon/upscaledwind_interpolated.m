load wind_power
%wind_power=[Wind13{:,:}];
% wind_power=readtable("Wind_13.1_power.xls");
% wind_power=readtable("Wind_13.2_power.xls");
%wind power is in MW therefore has to be changed to W
for i=1:size(wind_power,1)
   wind_power(i,1)=wind_power(i,1).*1e6;
end
plot(wind_power)
data=readtable("K:\Smart Grid\Germany_raw_data.xlsx");
data=[data{:,2:end}];
wind_orig=[];
for i=1:105193
    if ismember(i,indicesslice)
        wind_orig=[wind_orig;data(i,11)];
    end
end
wind_orig=wind_orig.*10;
figure
plot(wind_power)
hold on 
plot(wind_orig)
hold off
legend(["forecast" "original"])
wind_train=[wind_power(2:241,1) wind_orig(1:240,1)];
wind_test=[wind_power(242:end,1) wind_orig(241:end)];
wind_xtest=wind_test(:,1);
wind_ytest=wind_test(:,2);
%rmse = sqrt(mean(((wind_train(:,1)-wind_train(:,2)).^2)));
%wind_xtest=wind_xtest+rmse;
%wind_train(:,1)=wind_train(:,1)-rmse;
figure
plot(wind_xtest)
hold on 
plot(wind_ytest)
hold off
legend(["forecast" "original"])
%save wind_scale;
load wind_scale;
wind_ypred = wind_scale.predictFcn(wind_xtest);

% Pwind_real=wind_orig(end-24:end,1);
% save Pwind_real
% Pwind=[wind_orig(end-48:end-25,1)' wind_power(end-24:end,1)'];
% save Pwind

figure
plot(wind_ytest)
hold on 
plot(wind_ypred)
hold off

index=[1:60:1441]';
missing=[];
for i=1:1441
    if (~ismember(i,index))
        missing=[missing ;i];
    end
end
wind_pol=spline(index,wind_ypred,missing);        
wind=[];
a=1;b=1;
for i=1:1441
    if (ismember(i,index))
        wind=[wind wind_ypred(a,1)];
        a=a+1;
    else
        wind=[wind wind_pol(b,1)];
        b=b+1;
    end
end
Pwind_f=wind';
figure
plot(1:1441,Pwind_f)
hold on 
plot(index,wind_ypred)
hold off
xlswrite('wind_forecast.xls',Pwind_f(:,:))

