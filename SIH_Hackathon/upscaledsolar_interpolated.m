%load saved solar_power
load solar_power
for i=1:size(solar_power,1)
    if(solar_power(i,1)<=2.9)
        solar_power(i,1)=0;
    end
end

%solar power is in KW therefore has to be changed to W
for i=1:size(solar_power,1)
   solar_power(i,1)=solar_power(i,1)*1e3;
end
plot(solar_power)
data=readtable("K:\Smart Grid\Germany_raw_data.xlsx");
solar_orig=[data{end-264:end,8}];
solar_orig=solar_orig.*10;
solar_train=[solar_power(1:240,1) solar_orig(1:240,1)];
solar_test=[solar_power(241:end,1) solar_orig(241:end)];
solar_xtest=solar_test(:,1);
solar_ytest=solar_test(:,2);

%save solar_scale
load solar_scale
solar_ypred = solar_scale.predictFcn(solar_xtest);

Ppv_real=solar_ytest;
%save Ppv_real
Ppv=[solar_orig(end-48:end-25,1)' solar_ypred'];
%save Ppv

rmse = sqrt(mean((solar_ypred-solar_ytest).^2));
mae=mean(abs(solar_ytest-solar_ypred));

figure
plot(solar_ytest)
hold on 
plot(solar_ypred)
hold off

index=[1:60:1441]';
missing=[];
for i=1:1441
    if (~ismember(i,index))
        missing=[missing ;i];
    end
end
solar_pol=spline(index,solar_ypred,missing);        
solar=[];
a=1;b=1;
for i=1:1441
    if (ismember(i,index))
        solar=[solar solar_ypred(a,1)];
        a=a+1;
    else
        solar=[solar solar_pol(b,1)];
        b=b+1;
    end
end
Ppv_f=solar';
for i=1:size(Ppv_f,1)
    if(Ppv_f(i,1)<0)
        Ppv_f(i,1)=0;
    end
end
figure
plot(1:1441,Ppv_f)
hold on 
plot(index,solar_ypred)
hold off
%xlswrite('solar_forecast.xls',Ppv_f(:,:))

