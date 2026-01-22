import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthBegin
import statsmodels.api as sm
from pandas_datareader.famafrench import FamaFrenchReader

os.chdir(os.path.dirname(os.path.abspath(__file__)))

ff3_daily = FamaFrenchReader("F-F_Research_Data_Factors_daily", start=1926).read()[0].reset_index()
ff3_daily["date"] = pd.to_datetime(ff3_daily["Date"])
ff3_daily = ff3_daily[["date", "Mkt-RF", "SMB", "HML"]].astype(float)

ff3_monthly = FamaFrenchReader("F-F_Research_Data_Factors", start=1926).read()[0].reset_index()
ff3_monthly["date"] = pd.to_datetime(ff3_monthly["Date"].astype(str) + "01")
ff3_monthly = ff3_monthly[["date", "Mkt-RF", "SMB", "HML"]].astype(float)

start_date = "1926-07-01"
end_date = "2015-12-31"

daily = ff3_daily.set_index("date").loc[start_date:end_date]
monthly = ff3_monthly.set_index("date").loc[start_date:end_date]

con_var = daily["Mkt-RF"].resample("M").apply(lambda x: ((x - x.mean())**2).sum())
con_var = con_var.iloc[:-1]
con_var.index = con_var.index.to_period("M").to_timestamp() + MonthBegin(1)

monthly_sd = daily["Mkt-RF"].resample("M").std().iloc[1:]
monthly_var = daily["Mkt-RF"].resample("M").var().iloc[1:]

ret_per_var = monthly["Mkt-RF"] / monthly_var

sorts = pd.qcut(con_var, 5, labels=False) + 1
monthly["sorts"] = sorts.values
monthly_sd = monthly_sd.to_frame("Mkt-RF")
monthly_sd["sorts"] = sorts.values
ret_per_var = ret_per_var.to_frame("Mkt-RF")
ret_per_var["sorts"] = sorts.values

monthly_ret_ann = monthly["Mkt-RF"] * 12
monthly_sd_ann = monthly_sd["Mkt-RF"] * 22

ret_sorts = monthly_ret_ann.groupby(monthly["sorts"]).mean()
sd_sorts = monthly_sd_ann.groupby(monthly_sd["sorts"]).mean()
rpv_sorts = ret_per_var["Mkt-RF"].groupby(ret_per_var["sorts"]).mean()

labels = ["Low Vol", "2", "3", "4", "High Vol"]
ret_sorts.index = labels
sd_sorts.index = labels
rpv_sorts.index = labels

plt.figure(figsize=(8,4))
plt.subplot(2,2,1)
ret_sorts.plot(kind="bar", color="darkblue", ylim=(0,12))
plt.title("Average Return")
plt.subplot(2,2,2)
sd_sorts.plot(kind="bar", color="darkblue", ylim=(0,40))
plt.title("Standard Deviation")
plt.subplot(2,2,3)
rpv_sorts.plot(kind="bar", color="darkblue", ylim=(0,8))
plt.title("E[R]/Var(R)")
plt.tight_layout()
plt.savefig("out/sorts_1.png", dpi=600)
plt.close()

c = monthly_ret_ann.std() / (monthly_ret_ann / con_var).std()
managed = c / con_var * monthly_ret_ann

X = sm.add_constant(monthly_ret_ann)
model = sm.OLS(managed, X).fit()

Xc = sm.add_constant(monthly[["Mkt-RF", "SMB", "HML"]] * 12)
model_c = sm.OLS(managed, Xc).fit()

table_a = pd.DataFrame({
    "MktRF": [model.params[1]],
    "SE MktRF": [model.bse[1]],
    "Alpha": [model.params[0]],
    "SE Alpha": [model.bse[0]],
    "N": [int(model.nobs)],
    "R2": [model.rsquared],
    "RMSE": [np.sqrt(np.mean(model.resid**2))]
}, index=["Mkt.sigma"])

print(table_a)

table_b = pd.DataFrame({
    "Alpha": [model_c.params[0]],
    "SE Alpha": [model_c.bse[0]]
})

print(table_b)

start_date = "1926-07-01"
end_date = "2023-07-31"

daily = ff3_daily.set_index("date").loc[start_date:end_date]
monthly = ff3_monthly.set_index("date").loc[start_date:end_date]

D = 91

def rolling_con_var(x):
    idx = daily.index.get_loc(x.index[0])
    window = daily.iloc[idx-D:idx]["Mkt-RF"]
    return D/22 * ((window - window.mean())**2).sum()

con_var = daily["Mkt-RF"].iloc[D:].resample("M").apply(rolling_con_var)
con_var.index = con_var.index.to_period("M").to_timestamp()

monthly = monthly.loc[con_var.index]

monthly_sd = daily["Mkt-RF"].iloc[D:].resample("M").std()
monthly_var = daily["Mkt-RF"].iloc[D:].resample("M").var()

ret_per_var = monthly["Mkt-RF"] / monthly_var

sorts = pd.qcut(con_var, 5, labels=False) + 1
monthly["sorts"] = sorts.values
monthly_sd = monthly_sd.to_frame("Mkt-RF")
monthly_sd["sorts"] = sorts.values
ret_per_var = ret_per_var.to_frame("Mkt-RF")
ret_per_var["sorts"] = sorts.values

monthly_ret_ann = monthly["Mkt-RF"] * 12
monthly_sd_ann = monthly_sd["Mkt-RF"] * 22

ret_sorts = monthly_ret_ann.groupby(monthly["sorts"]).mean()
sd_sorts = monthly_sd_ann.groupby(monthly_sd["sorts"]).mean()
rpv_sorts = ret_per_var["Mkt-RF"].groupby(ret_per_var["sorts"]).mean()

ret_sorts.index = labels
sd_sorts.index = labels
rpv_sorts.index = labels

plt.figure(figsize=(8,4))
plt.subplot(2,2,1)
ret_sorts.plot(kind="bar", color="darkblue", ylim=(0,12))
plt.title("Average Return")
plt.subplot(2,2,2)
sd_sorts.plot(kind="bar", color="darkblue", ylim=(0,40))
plt.title("Standard Deviation")
plt.subplot(2,2,3)
rpv_sorts.plot(kind="bar", color="darkblue", ylim=(0,8))
plt.title("E[R]/Var(R)")
plt.tight_layout()
plt.savefig("out/sorts_2.png", dpi=600)
plt.close()

c = monthly_ret_ann.std() / (monthly_ret_ann / con_var).std()
managed = c / con_var * monthly_ret_ann

X = sm.add_constant(monthly_ret_ann)
model = sm.OLS(managed, X).fit()

Xc = sm.add_constant(monthly[["Mkt-RF", "SMB", "HML"]] * 12)
model_c = sm.OLS(managed, Xc).fit()

table_a = pd.DataFrame({
    "MktRF": [model.params[1]],
    "SE MktRF": [model.bse[1]],
    "Alpha": [model.params[0]],
    "SE Alpha": [model.bse[0]],
    "N": [int(model.nobs)],
    "R2": [model.rsquared],
    "RMSE": [np.sqrt(np.mean(model.resid**2))]
}, index=["Mkt.sigma"])

print(table_a)

table_b = pd.DataFrame({
    "Alpha": [model_c.params[0]],
    "SE Alpha": [model_c.bse[0]]
})

print(table_b)
