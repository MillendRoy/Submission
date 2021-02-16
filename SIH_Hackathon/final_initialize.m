date='2016-12-30';
load Ppv_real;
clear avg Ceq Clink Cs data dataset dev Fc Fnom hidden_units_size i idx Increment_MPPT input_size irradiance irradiance_net j k Ki_Ireg Ki_VDCreg Kp_Ireg Kp_VDCreg l layers LimitL_lreg 
clear LimitL_VDCreg Limits_MPPT LimitU_Ireg LimitU_VDCreg Lm_xfo num_Test num_Train nw options output_size Pbase_sec Pc Pnom Pnom_xfo Qc RL RRLchoke Rlff Rm_xfo rmse Ron Rs sc_test sc_train solar_orig LimitL_Ireg RLchoke RLff  VaIa_Grid
clear solar_power solar_scale solar_test solar_train solar_xtest solar_ypred solar_ytest temp test TotalLeakage tout train Ts_Control Ts_Power Vab_VSC1 Vala_Grid Vf Vfd Vnom_dc Vnom_prim Vnom_sec W1_xfo W2_xfo X_test X_train X_validation y_pred y_Pred y_test y_Test y_train y_validation
load Ppv;
clear avg Ceq Clink Cs data dataset dev Fc Fnom hidden_units_size i idx Increment_MPPT input_size irradiance irradiance_net j k Ki_Ireg Ki_VDCreg Kp_Ireg Kp_VDCreg l layers LimitL_lreg 
clear LimitL_VDCreg Limits_MPPT LimitU_Ireg LimitU_VDCreg Lm_xfo num_Test num_Train nw options output_size Pbase_sec Pc Pnom Pnom_xfo Qc RL RRLchoke Rlff Rm_xfo rmse Ron Rs sc_test sc_train solar_orig LimitL_Ireg RLchoke RLff  VaIa_Grid
clear solar_power solar_scale solar_test solar_train solar_xtest solar_ypred solar_ytest temp test TotalLeakage tout train Ts_Control Ts_Power Vab_VSC1 Vala_Grid Vf Vfd Vnom_dc Vnom_prim Vnom_sec W1_xfo W2_xfo X_test X_train X_validation y_pred y_Pred y_test y_Test y_train y_validation
load y_test_wind;
load y_pred_wind;
load y_test_price;
load y_pred_price;
load y_test_load;
load y_pred_load;
clear avg count data dataset dev hidden_units_size i idx index input_size j k l layers load_net m n num_Test num_Train
clear nw options output_size price price_data price_net price_pol rmse sc_test sc_train test time train trainedModel X_test X_train X_validation y_pred y_Pred
clear y_test y_Test y_train y_validation

w=numel(y_test_wind);
l=numel(y_test_load);
p=numel(y_test_price);
y_test_wind=y_test_wind(end-l+1:end);
y_pred_wind=y_pred_wind(end-l+1:end);
y_test_price=y_test_price(end-l+1:end);
y_pred_price=y_pred_price(end-l+1:end);
[Pwind,cost_data,load_data,load_real,cost_real,Pwind_real]=accdate(date,y_test_wind,y_pred_wind,y_test_price,y_pred_price,y_test_load, y_pred_load);
time=[60:60:86460]';
load_data=load_data'*10;
cost_data=cost_data';
Pwind=Pwind'*10; 
index=[1:60:2881]';
missing=[];
for i=1:2881
    if (~ismember(i,index))
        missing=[missing i];
    end
end
missing=missing';
cost_pol=spline(index,cost_data,missing);        
cost=[];
a=1;b=1;
for i=1:2881
    if (ismember(i,index))
        cost=[cost cost_data(a,1)];
        a=a+1;
    else
        cost=[cost cost_pol(b,1)];
        b=b+1;
    end
end
costData=cost';
% figure
% plot(1:2881,costData)
% hold on 
% plot(index,cost_data)
% hold off

load_pol=spline(index,load_data,missing);        
load=[];
a=1;b=1;
for i=1:2881
    if (ismember(i,index))
        load=[load load_data(a,1)];
        a=a+1;
    else
        load=[load load_pol(b,1)];
        b=b+1;
    end
end
loadData=load';
% figure
% plot(1:2881,loadData)
% hold on 
% plot(index,load_data)
% hold off
avg=mean(loadData);
loadData=loadData-avg;

Ppv_pol=spline(index,Ppv,missing);        
Ppv_f=[];
a=1;b=1;
Ppv=Ppv';
for i=1:2881
    if (ismember(i,index))
        Ppv_f=[Ppv_f Ppv(a,1)];
        a=a+1;
    else
        Ppv_f=[Ppv_f Ppv_pol(b,1)];
        b=b+1;
    end
end
Ppv_f=Ppv_f';
% figure
% plot(1:2881,Ppv_f)
% hold on 
% plot(index,Ppv)
% hold off

Pwind_pol=spline(index,Pwind,missing);        
Pwind_f=[];
a=1;b=1;
for i=1:2881
    if (ismember(i,index))
        Pwind_f=[Pwind_f Pwind(a,1)];
        a=a+1;
    else
        Pwind_f=[Pwind_f Pwind_pol(b,1)];
        b=b+1;
    end
end
Pwind_f=Pwind_f';
% figure
% plot(1:2881,Pwind_f)
% hold on 
% plot(index,Pwind)
% hold off

% Microgrid Settings
loadBase = avg;   % Base Load of Microgrid [W]

BattCap = 2500;     % Energy Storage Rated Capacity [kWh]
batteryMinMax.Pmin = -400e3;    % Max Discharge Rate [W]
batteryMinMax.Pmax = 400e3;     % Max Charge Rate [W]


% Online optimization parameters
FinalWeight = 1;    % Final weight on energy storage
timeOptimize = 5;    % Time step for optimization [min]
timePred = 24;        % Predict ahead horizon [hours]

% Battery SOC Energy constraints (keep between 20%-80% SOC)
battEnergy = 3.6e6*BattCap;
batteryMinMax.Emax = 0.8*battEnergy;
batteryMinMax.Emin = 0.2*battEnergy;


% Setup Optimization time vector
optTime = timeOptimize*60; %300secs
stepAdjust = (timeOptimize*60)/(time(2)-time(1)); %5mins
N = numel(time(1:stepAdjust:end))-1; %288
tvec = (1:N+1)'*optTime; %288x1----300,600,...86700

% Horizon for "sliding" optimization
M = find(tvec > timePred*3600,1,'first');%289

PpvVec = Ppv_f(2:stepAdjust:end);
loadDataOpt = loadData(2:stepAdjust:end) + loadBase;
C = costData(2:stepAdjust:end);
PwindVec = Pwind_f(2:stepAdjust:end);

CostMat = zeros(N,M);
PpvMat = zeros(N,M);
PloadMat = zeros(N,M);
PwindMat = zeros(N,M);
% Construct forecast vectors for optimization (N x M) matrix
for i = 1:N
    CostMat(i,:) = C(i:i+M-1);
    PpvMat(i,:) = PpvVec(i:i+M-1);
    PloadMat(i,:) = loadDataOpt(i:i+M-1);
    PwindMat(i,:) = PwindVec(i:i+M-1);
end

CostForecast.time = tvec(1:N);
CostForecast.signals.values = CostMat;
CostForecast.signals.dimensions = M;

PpvForecast.time = tvec(1:N);
PpvForecast.signals.values = PpvMat;
PpvForecast.signals.dimensions = M;

PwindForecast.time = tvec(1:N);
PwindForecast.signals.values = PwindMat;
PwindForecast.signals.dimensions = M;

PloadForecast.time = tvec(1:N);
PloadForecast.signals.values = PloadMat;
PloadForecast.signals.dimensions = M;
costData=double(costData);
loadData=double(loadData);
clear a b i missing ;
load_real=load_real'*10;
cost_real=cost_real';
Ppv_real=Ppv_real';
Pwind_real=Pwind_real'*10;
index_real=[1:60:1441]';
missing=[];
for i=1:1441
    if (~ismember(i,index_real))
        missing=[missing i];
    end
end
missing=missing';
cost_realpol=spline(index_real,cost_real,missing);        
costreal=[];
a=1;b=1;
for i=1:1441
    if (ismember(i,index_real))
        costreal=[costreal cost_real(a,1)];
        a=a+1;
    else
        costreal=[costreal cost_realpol(b,1)];
        b=b+1;
    end
end
costrealData=costreal';
% figure
% plot(1:1441,costrealData)
% hold on 
% plot(index_real,cost_real)
% hold off

load_realpol=spline(index_real,load_real,missing);        
loadreal=[];
a=1;b=1;
for i=1:1441
    if (ismember(i,index_real))
        loadreal=[loadreal load_real(a,1)];
        a=a+1;
    else
        loadreal=[loadreal load_realpol(b,1)];
        b=b+1;
    end
end
loadrealData=loadreal';
% figure
% plot(1:1441,loadrealData)
% hold on 
% plot(index_real,load_real)
% hold off
loadrealData=loadrealData-avg;

Ppv_realpol=spline(index_real,Ppv_real,missing);        
Ppv_realf=[];
a=1;b=1;
Ppv_real=Ppv_real';
for i=1:1441
    if (ismember(i,index_real))
        Ppv_realf=[Ppv_realf Ppv_real(a,1)];
        a=a+1;
    else
        Ppv_realf=[Ppv_realf Ppv_realpol(b,1)];
        b=b+1;
    end
end
Ppv_realf=Ppv_realf';
% figure
% plot(1:1441,Ppv_realf)
% hold on 
% plot(index_real,Ppv_real)
% hold off

Pwind_realpol=spline(index_real,Pwind_real,missing);        
Pwind_realf=[];
a=1;b=1;

for i=1:1441
    if (ismember(i,index_real))
        Pwind_realf=[Pwind_realf Pwind_real(a,1)];
        a=a+1;
    else
        Pwind_realf=[Pwind_realf Pwind_realpol(b,1)];
        b=b+1;
    end
end
Pwind_realf=Pwind_realf';
% figure
% plot(1:1441,Pwind_realf)
% hold on 
% plot(index_real,Pwind_real)
% hold off
loadrealData=double(loadrealData);

clear a b i missing; 
function [Pwind,cost_data,load_data,load_real,cost_real,Pwind_real]=accdate(date,y_test_wind,y_pred_wind,y_test_price,y_pred_price,y_test_load, y_pred_load)
da='2016-12-01';
numdays = datenum(date) - datenum(da)+1;
Pwind=[y_test_wind(end-744-23+(numdays*24)-1:end-744+(numdays*24)-1,1); y_pred_wind(end-743+(numdays*24)-1:end-743+24+(numdays*24)-1,1)]';
cost_data=[y_test_price(end-744-23+(numdays*24)-1:end-744+(numdays*24)-1,1); y_pred_price(end-743+(numdays*24)-1:end-743+24+(numdays*24)-1,1)]';
load_data=[y_test_load(end-744-23+(numdays*24)-1:end-744+(numdays*24)-1,1); y_pred_load(end-743+(numdays*24)-1:end-743+24+(numdays*24)-1,1)]';
load_real=[y_test_load(end-743+(numdays*24)-1:end-743+24+(numdays*24)-1,1)]';
Pwind_real=[y_test_wind(end-743+(numdays*24)-1:end-743+24+(numdays*24)-1,1)]';
cost_real=[y_test_price(end-743+(numdays*24)-1:end-743+24+(numdays*24)-1,1)]';
end