
### safety_efficiency_GLM.py

### This script enables statistical exploration of how efficiency decisions 
### impact safety outcomes in air traffic management.


### Main steps:
### 1. Database Connection & Query Execution
####   - Uses credentials from a local Excel file to connect to Oracle.
####  - Executes a SQL query to pull traffic flow and safety-related data.
####
### 2. Data Preprocessing
####   - Converts date fields to datetime format.
####   - Handles missing values by filling with zeros.
####   - Creates ratio-based variables (e.g., LOCAL_ARIA_COUNT_RATIO).
####   - Filters out rows with invalid denominators.

### 3. Regression Modeling (runRegression)
####   - Supports different response variable types:
####     - count: Poisson vs. Negative Binomial (chooses based on overdispersion).
####     - rate_pos: Gamma or Tweedie models for positive rate outcomes.
####     - rate_0_1: Binomial or Tweedie for bounded rate variables.
####   - Automatically creates lagged predictor variables (time-series context).
####   - Returns fitted model, data, and diagnostic details.

### 4. Effect Analysis & Summary
####   - Iterates over a set of safety response variables (e.g., NUM_TCAS, MA_COUNT).
####   - Extracts coefficients, p-values, and interprets significance/effect direction.
####   - Produces a summary pivot table of variable effects across models.

### 5. Model Diagnostics & Validation
####   - Reports log-likelihood, deviance, Pearson chi-square, and zero-inflation.
####   - Generates residual plots to assess model fit.
####   - Calculates Variance Inflation Factors (VIF) for multicollinearity checks.


### Author:
####     Ehsan Esmaeilzadeh (ehsan@mitre.org)

### Date:
####     June 2025







import pandas as pd
pd.set_option("display.max_columns", None)
import numpy as np

import cx_Oracle

import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.tools.tools import add_constant
from scipy.stats import shapiro
import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV,PoissonRegressor
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats






## conect to Oracle
login_info = pd.read_excel("~/.....")
login_info.set_index("Schema", inplace=True)
username = "kimia"
password = login_info.loc[username]["Pass"]
service_name = login_info.loc[username]["Database"]
host = "...."
port = "1521"
# connection = cx_Oracle.connect('{}/{}@{}'.format(usr,pas,db))
dsn = cx_Oracle.makedsn(host, port, service_name=service_name)
conn = cx_Oracle.connect(user=username, password=password, dsn=dsn)









## pull the data from a pre-processed table
## you need to have access to this table in EHSAN schema
query = f"""
SELECT T.TMI_TYPE,T.AIRPORT,T.FIRSTSENT,T.LASTSENT,T.FIRSTSTART,T.LASTEND,T.MAXEND,
T.EARLY_CNX_HRS,T.FIRSTREASON,T.LASTREASON,T.THREADID,T.SLICE_START_UTC,T.EFFARR,
T.DLATI,T.TBFM_DELAY,T.AAR,ARRDEMAND,NVL(T.ADL_ARRDEMAND,T.ARRDEMAND) AS ADL_ARRDEMAND,
T.EXCESS_DEMAND,
CASE WHEN NVL(T.ADL_ARRDEMAND,T.ARRDEMAND) - T.AAR < 0 THEN 0 ELSE NVL(T.ADL_ARRDEMAND,T.ARRDEMAND) - T.AAR END AS ADL_EXCESS_DEMAND,
S.PCV_PROBABILITIES,S.ARIA_COUNT,S.SIGNIFICANT_ARIA_COUNT,S.LOCAL_ARIA_COUNT,
S.LOCAL_SIGNIFICANT_ARIA_COUNT,S.MA_COUNT,S.HOLD_FLT_CNT,S.HOLD_TOTAL_DURATION,
S.VECTOR_FLT_CNT,S.VECTOR_TOTAL_DURATION,NVL(S.HOLD_FLT_CNT,0)+NVL(S.HOLD_FLT_CNT,0) AS TOTAL_HOLD_VECTOR_FLT_CNT,
S.TOTAL_ADDITIONAL_TIME AS TOTAL_HOLD_VECTOR_DURATION,S.NUM_TCAS,
S.TOTAL_MIX_USE_CASES,S.MIX_USE_DISTSEP_VIOLATIONS,S.APPROACH_COUNT,S.NUM_DEPARTURES,
T.DATA_DATE

FROM SAFETY_EFFICIENCY_TMI_EARLY_CNX_QH T 
LEFT JOIN SAFETY_EFFICIENCY_TMI_EARLY_CNX_QH_SAFETY S ON
T.SLICE_START_UTC = TO_DATE(S.T_15_MIN_BIN,'MM/DD/YY HH24:MI') AND
T.THREADID = S.SCENARIO
WHERE
SLICE_START_UTC BETWEEN LEAST(LASTEND - 30/1440,TRUNC(LASTEND,'HH24') - 30/1440)  AND GREATEST (LASTEND + 4/24,TRUNC(LASTEND,'HH24') + 4/24) AND
DATA_DATE <= TO_DATE('08/31/2024','MM/DD/YYYY')"""

df = pd.read_sql(query, con=conn)

conn.close()

df['SLICE_START_UTC'] = pd.to_datetime(df['SLICE_START_UTC'])
df['FIRSTSTART'] = pd.to_datetime(df['FIRSTSTART'])
df['LASTEND'] = pd.to_datetime(df['LASTEND'])
df['DATA_DATE'] = pd.to_datetime(df['DATA_DATE'])

df['EFFARR'] = df['EFFARR'].fillna(0)
df['EXCESS_DEMAND'] = df['EXCESS_DEMAND'].fillna(0)
df['TOTAL_HOLD_VECTOR_DURATION'] = df['TOTAL_HOLD_VECTOR_DURATION'].fillna(0)
df['TBFM_DELAY'] = df['TBFM_DELAY'].fillna(0)
df['DLATI'] = df['DLATI'].fillna(0)

df['ARIA_COUNT'] = df['ARIA_COUNT'].fillna(0)
df['LOCAL_ARIA_COUNT'] = df['LOCAL_ARIA_COUNT'].fillna(0)
df['PCV_PROBABILITIES'] = df['PCV_PROBABILITIES'].fillna(0)
df['NUM_TCAS'] = df['NUM_TCAS'].fillna(0)
df['MA_COUNT'] = df['MA_COUNT'].fillna(0)
df['TOTAL_MIX_USE_CASES'] = df['TOTAL_MIX_USE_CASES'].fillna(0)

df['LOCAL_ARIA_COUNT_RATIO'] = df['LOCAL_ARIA_COUNT']/df['EFFARR']
df['NUM_TCAS_RATIO'] = df['NUM_TCAS']/df['EFFARR']
df['MA_COUNT_RATIO'] = df['MA_COUNT']/df['EFFARR']
df['TOTAL_MIX_USE_CASES_RATIO'] = df['TOTAL_MIX_USE_CASES']/df['EFFARR']

df = df[(df['EFFARR'] != 0) & (df['AAR'] != 0)].copy()








## run the regression
## decide on the best model based for each safety metric
def runRegression(apt_input,var_input,var_type):

    data = df2.copy()
    data[var_input] = df[var_input] 
    data = data[data['AIRPORT'] == apt_input]

    Independent_vars = [col for col in data.columns if col not in [var_input, 'THREADID', 'SLICE_START_UTC', 'AIRPORT']]

    data = data.sort_values(by=['THREADID', 'SLICE_START_UTC']).reset_index(drop=True)

    num_lags = 4
    for metric in Independent_vars:
        for lag in range(1, num_lags + 1):  # 4 lags
            data[f'{metric}_Lag{lag}'] = data.groupby('THREADID')[metric].shift(lag)


    data.dropna(inplace=True)

    independent_vars = [col for col in data.columns if col not in [var_input, 'THREADID', 'SLICE_START_UTC', 'AIRPORT']]

    X = sm.add_constant(data[independent_vars]) 
    y = data[var_input]


    if var_type == 'rate_pos':

        mean_response = data[var_input].mean()
        var_response = data[var_input].var()
        variance_to_mean_ratio = var_response / mean_response

        skewness = stats.skew(data[var_input])

        if variance_to_mean_ratio > 1.5 or skewness > 1:
            formula = f"{var_input} ~ " + " + ".join(independent_vars)
            # Define a range of var_power values to test
            var_power_values = np.arange(1.0, 2.1, 0.1)  # Tweedie supports var_power between 1 and 2
            results = []
            # Iterate through different var_power values and fit models
            for var_p in var_power_values:
                try:
                    model = sm.GLM(y, X,family=sm.families.Tweedie(var_power=var_p, link=sm.families.links.log())).fit()
                    results.append((var_p, model.aic, model.llf))  
                except Exception as e:
                    print(" ")
            results_df = pd.DataFrame(results, columns=["var_power", "AIC", "Log-Likelihood"])
            # Select the best var_power (lowest AIC)
            best_var_power = results_df.loc[results_df["AIC"].idxmin(), "var_power"]
            # Refit the model using the best var_power
            model = sm.GLM(y, X,family=sm.families.Tweedie(var_power=best_var_power, link=sm.families.links.log())).fit()            
            model_family = type(model.model.family).__name__ 
            model_type = type(model.model).__name__ 
        else:
            model = sm.GLM(y, X,family=sm.families.Gamma(link=sm.families.links.log())).fit()
            model_family = type(model.model.family).__name__
            model_type = type(model.model).__name__ 


    elif var_type == 'count':

        mean_response = data[var_input].mean()
        var_response = data[var_input].var()
        vmr = var_response / mean_response

        poisson_model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()

        # Likelihood Ratio Test
        lrt_stat = 2 * (nb_model.llf - poisson_model.llf)
        p_value = stats.chi2.sf(lrt_stat, df=1)

        # Overdispersion Factor
        poisson_deviance = ((data[var_input] - poisson_model.fittedvalues) ** 2 / poisson_model.fittedvalues).sum()
        overdispersion_factor = poisson_deviance / poisson_model.df_resid


        if vmr > 1.5 or p_value < 0.05 or overdispersion_factor > 1.5:
            model = sm.GLM(y, X, family=sm.families.NegativeBinomial()).fit()
            model_type = type(model.model).__name__ 
            model_family = type(model.model.family).__name__ 

        else:    
            model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
            model_type = type(model.model).__name__ 
            model_family = type(model.model.family).__name__ 



    elif var_type == 'rate_0_1':        

        data = data.replace([np.inf, -np.inf], np.nan)
        data = data.dropna(subset=[var_input] + independent_vars)

        zero_ratio = (data[var_input] == 0).mean()

        X = sm.add_constant(data[independent_vars]) 
        y = data[var_input]

        try:
            if zero_ratio > 0.6:
                model = sm.GLM(y, X, 
                    data=data,
                    family=sm.families.Tweedie(var_power=1.5, link=sm.families.links.log())
                ).fit()
                model_type = "Tweedie GLM"
            else:
                model = sm.GLM(y, X, 
                    data=data,
                    family=sm.families.Binomial(link=sm.families.links.logit())
                ).fit(scale='X2')  # Adds robust variance estimate
                model_type = "Quasi-Binomial GLM"

        except Exception as e:
            print(f" Model fitting failed: {str(e)}")

        model_family = type(model.model.family).__name__         


    coef = model.params
    p_values = model.pvalues
    exp_coef = np.exp(coef) 

    def interpret_variable(coef, exp_coef, p_value):
        if p_value < 0.05: 
            if exp_coef > 1:
                return f"Increases response by {round((exp_coef - 1) * 100, 1)}%"
            else:
                return f"Decreases response by {round((1 - exp_coef) * 100, 1)}%"
        else:
            return "Not significant"

    interpretation_df = pd.DataFrame({
        "Variable": coef.index,
        "Coefficient": coef.values,
        "IRR (Exp Coef)": exp_coef.values,
        "p-value": p_values.values, 
        "Interpretation": [interpret_variable(coef[i], exp_coef[i], p_values[i]) for i in coef.index]
    })

    return model, data,X, y









## run the results for a given airport

apt = 'SFO'

df2 = df[['SLICE_START_UTC','AIRPORT','THREADID','EFFARR','EXCESS_DEMAND','DLATI','TBFM_DELAY','TOTAL_HOLD_VECTOR_DURATION']]

response_var_list = ['LOCAL_ARIA_COUNT','PCV_PROBABILITIES','NUM_TCAS','MA_COUNT','TOTAL_MIX_USE_CASES']


summary_results = []

for response_var in response_var_list:
    if response_var.endswith('_RATIO'):
        var_type = 'rate_0_1'
    elif response_var == 'PCV_PROBABILITIES':
        var_type = 'rate_pos'
    else:
        var_type = 'count'

    model, data, X, y = runRegression(apt, response_var,var_type)

    coef = model.params
    p_values = model.pvalues
    exp_coef = np.exp(coef)

    def categorize_effect(p_value, exp_coef):
        if p_value < 0.05:
            effect_size = (exp_coef - 1) * 100
            return f'{effect_size:.1f}%'
        return '✖'

    for var in coef.index:
        if var != 'const':  # Exclude the constant term
            effect_category = categorize_effect(p_values[var], exp_coef[var])
            summary_results.append({
                'Variable': var,
                'Response Variable': response_var,
                'Effect Category': effect_category
            })

summary_df = pd.DataFrame(summary_results)
summary_pivot = summary_df.pivot(index='Response Variable', columns='Variable', values='Effect Category').fillna('✖')
summary_pivot.head()










## Model Validation

# Extract key statistics from the Negative Binomial model
log_likelihood = model.llf  # Log-Likelihood
deviance = model.deviance   # Deviance
pearson_chi2 = model.pearson_chi2  # Pearson Chi-Square
df_residual = model.df_resid  # Residual Degrees of Freedom
zero_percentage = (data[response_var] == 0).sum() / len(data) * 100


# Compute test ratios
deviance_ratio = deviance / df_residual
pearson_ratio = pearson_chi2 / df_residual

# Print Summary
summary_report = f"""
================= Model Fit Summary =================
Log-Likelihood: {log_likelihood:.2f}

Deviance Test:
- Deviance: {deviance:.2f}
- Residual Degrees of Freedom: {df_residual}
- Deviance / df: {deviance_ratio:.2f}

Pearson Chi-Square Test:
- Pearson Chi-Square: {pearson_chi2:.2f}
- Pearson Chi-Square / df: {pearson_ratio:.2f}

Percentage of Zero Values in model: {zero_percentage:.2f}%

=====================================================
"""

print(summary_report)



plt.figure(figsize=(8,6))
plt.scatter(model.fittedvalues, model.resid_deviance, alpha=0.5)
plt.axhline(0, linestyle='dashed', color='red')
plt.xlabel("Fitted Values")
plt.ylabel("Deviance Residuals")
plt.title("Residual Plot for Poisson Model")
plt.show()


vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif_data.head(20)
