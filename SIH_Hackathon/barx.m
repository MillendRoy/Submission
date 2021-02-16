D = duration(0,1:5:1440,0);
demand=[];
for i= 1442:2881
    if (mod(i-1,5)==0)
        demand=[demand ;loadData(i)];
    end
end
hawa=[];
for i= 1442:2881
    if (mod(i-1,5)==0)
        hawa=[hawa ;Pwind_f(i)];
    end
end
surya=[];
for i= 1442:2881
    if (mod(i-1,5)==0)
        surya=[surya ;Ppv_f(i)];
    end
end
gridy=[];
for i= 2:1441
    if (mod(i-1,5)==0)
        gridy=[gridy ;Pgrid(i)];
    end
end
charger=[];
for i= 2:1441
    if (mod(i-1,5)==0)
        charger=[charger ;ESS(i)];
    end
end
figure
bar(D(157:180),[surya(157:180)/1000 hawa(157:180)/1000 gridy(157:180)/1000 charger(157:180)/1000],'stacked')
hold('on')
plot(D(157:180),demand(157:180)/1000+loadBase/1000)
hold('off')
xlabel("Time (in hrs)")
ylabel("Power (in W)")
legend(["Solar Power" "Wind Power" "Grid Power" "ESS Power" "Load"])