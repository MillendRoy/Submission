simout=sim('Forecast_time');

ESS=simout.ESS;
Eess=simout.Eess;
Egrid=simout.Egrid;
Eload=simout.Eload;
Epv=simout.Epv;
Ewind=simout.Ewind;
Pgrid=simout.Pgrid;
SOC=simout.SOC;
dates=datetime(date,'InputFormat','yyyy-MM-dd');
save dates;
