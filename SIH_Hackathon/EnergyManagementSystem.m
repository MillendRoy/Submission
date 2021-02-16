function [optVec,SOH] = fcn(t, SOC,Ebatt,dt,Cost,Ppv,Pwind, Pload)
%#codegen

M = 289;
optVec = 0;
Pbatt = zeros(M,1);
battp=Ebatt;
EnergyWeight = 1;

%SOH constraints
SOH=100;
if (t/(31536000)< 15)
    SOH=SOH-0.03*SOH*t/(31536000);
else
    SOH=-1;
end

if(SOH==-1)
    message="Battery Maintenace Required";
    disp(message);
end

%SOH dependency on Ebatt
battp=battp*SOH/100;


% Energy constraints
batteryMinMax.Emax = 0.8*Ebatt;
batteryMinMax.Emin = 0.2*Ebatt;

% Power constraints
batteryMinMax.Pmin = -400e3;
batteryMinMax.Pmax = 400e3;

% Declare function (contains linprog) as extrinsic
coder.extrinsic('battSolarOptimize_fullMatrices');

% Battery parameters
Einit = SOC/100*Ebatt;

    
[~,Pbatt,~] = battSolarOptimize_fullMatrices(M,dt,Ppv,Pwind,Pload,Einit,Cost,...
    EnergyWeight,batteryMinMax);

optVec = Pbatt(1);
% optVec = mean(Pbatt(1:5));E