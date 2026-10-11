"""Observed-label scores and honest matched-support portfolio accumulation."""
import numpy as np
import pandas as pd


class Scores:
    def __init__(self):self.n=0;self.abs=0.;self.sq=0.;self.error=0.;self.actual=0.
    def update(self,actual,predicted):
        a,p=np.asarray(actual,dtype=float),np.asarray(predicted,dtype=float)
        good=np.isfinite(a)&np.isfinite(p);e=p[good]-a[good]
        self.n+=int(good.sum());self.abs+=float(np.abs(e).sum());self.sq+=float((e*e).sum())
        self.error+=float(e.sum());self.actual+=float(np.abs(a[good]).sum())
    def result(self):
        return dict(n=self.n,MAE=self.abs/self.n if self.n else None,
                    RMSE=(self.sq/self.n)**.5 if self.n else None,
                    WAPE=100*self.abs/self.actual if self.actual>0 else None,
                    Bias=self.error/self.n if self.n else None,ActualEnergy=self.actual)


class ValidationScores:
    def __init__(self):
        self.house=Scores();self.port=Scores();self.daily_abs=[]
    def update_day(self,frame,predicted):
        actual=frame.Target.to_numpy(dtype=float);prediction=np.asarray(predicted,dtype=float)
        good=np.isfinite(actual)&np.isfinite(prediction)
        self.house.update(actual,prediction)
        q=pd.DataFrame({"Timestamp":frame.TargetTimestamp,"a":np.where(good,actual,0.),
                        "p":np.where(good,prediction,0.),"n":good.astype(int)})
        sums=q.groupby("Timestamp")[["a","p","n"]].sum();sums=sums[sums.n>0]
        if sums.empty:
            return  # Forecast exists, but no matched actual support: accuracy undefined.
        self.port.update(sums.a,sums.p)
        self.daily_abs.append(float(abs(sums.p.sum()-sums.a.sum())))
    def result(self):
        return dict(Household=self.house.result(),MatchedPortfolio=self.port.result(),
                    DailyEnergyMAE=float(np.mean(self.daily_abs)) if self.daily_abs else None,
                    DailyEnergyMedianAE=float(np.median(self.daily_abs)) if self.daily_abs else None,
                    EvaluatedDays=len(self.daily_abs))


def selection_key(result):
    # Portfolio procurement volume is primary; household MAE is tie-break.
    return (result["MatchedPortfolio"]["WAPE"],result["Household"]["MAE"])


def subgroup_labels(frame):
    d={"PV":frame.Installation_HasPVSystem,"HPType":frame.Survey_HeatPump_Installation_Type,
       "EV":frame.Survey_Installation_HasElectricVehicle,"ForecastState":frame.ForecastState,
       "Weather_ID":frame.Weather_ID,"SunshineAvailable":np.where(frame.PredictedSunshineMissing,"Unavailable","Available"),
       "HeatPumpDHW":frame.Survey_DHW_Production_ByHeatPump,
       "ElectricDHW":frame.Survey_DHW_Production_ByElectricWaterHeater,
       "SolarDHW":frame.Survey_DHW_Production_BySolar}
    months=frame.TargetTimestamp.dt.month
    d["Season"]=months.map({12:"DJF",1:"DJF",2:"DJF",3:"MAM",4:"MAM",5:"MAM",6:"JJA",7:"JJA",8:"JJA",9:"SON",10:"SON",11:"SON"})
    d["LivingAreaBand"]=pd.cut(frame.Survey_Building_LivingArea,[0,150,200,300,np.inf],labels=["<=150","150-200","200-300",">300"]).astype(object).fillna("Unknown")
    d["PredictedTemperatureBand"]=pd.cut(frame.PredictedTemperature,[-np.inf,0,5,10,15,20,np.inf],labels=["<=0","0-5","5-10","10-15","15-20",">20"]).astype(object).fillna("Unknown")
    return d
