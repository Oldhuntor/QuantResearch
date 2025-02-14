# Exercise 1: Simulate the distributions of the DF coefficient test and DF t-test

# Case 1: no deterministic terms

# Function WP simulates a RW of a given length and returns the sample moments
# for the nominator ( int_0^1 W(s) dW(s) ) and the denominator (int_0^1 W^2(s) ds)
# of the DF test statistics

WP = function(nobs){
  e = rnorm(nobs) # vector of epsilons of length T
  y = cumsum(e)  # vector of Y_1, ..., Y_T
  y_1 = y[1:(nobs-1)] # Vector Y_{t-1}
  intW2 = nobs^(-2)*sum(y_1^2) # denominator of the DF distribution
  intWdW = nobs^(-1)*sum(y_1*e[2:nobs]) #nominator of the DF distribution
  ans = list(intW2=intW2, intWdW=intWdW)
  ans
}

# Initialize the parameters of the simulations

set.seed(999)

T = 120  #number of time points
nn = 1000 #number of replications

DF_coef = rep(0, nn)
DF_tstat = rep(0, nn)

for(i in 1:nn){
  moments = WP(T) # store the cross-products for each replication
  DF_coef[i] = moments$intWdW / moments$intW2
  DF_tstat[i] = moments$intWdW / sqrt(moments$intW2)
}

hist(DF_coef)
hist(DF_tstat)

# empirical critical values: compare with the asymptotic ones from Table 2.1 of Choi's (2015) book
quantile(DF_coef, probs=c(0.01,0.05,0.1))
quantile(DF_tstat, probs=c(0.01,0.05,0.1))

# Exercise 2: experiment with different distributions for the error terms e in WP,
# as well as with different T (e.g. 100, 200, 500, 1000, 2500) and see whether/how the distribution changes

# Exercise 3: Implement the DF-test for the other two cases: case 2 (intercept) and case 3 (intercept and trend)
# Observe how the distribution of the test statistics shifts to the left with increasing number of det. terms


# A function to simulate a RW, RW with drift, a trend-stationary process and a stationary AR(1)

RWsim = function(T, Y0, slope, trend, psi){
  e = rnorm (T)
  RW = cumsum(e) + Y0
  RW_drift = slope*trend + RW
  trend_stationary = slope*trend + e
  ar1 = arima.sim(model = list(ar = psi), n=T)
  ans = list(RW=RW, RW_drift = RW_drift, trend_stationary = trend_stationary, ar1  = ar1)
}


set.seed(999)
T = 240 # corresponds to 20 years of monthly data
Y0 = rnorm(1) #initial value for the RW
trend = 1:T
univar = runif(2) # trend slope and ar1 parameter
slope = univar[1]/2 # slope ~ U(0, 0.5)
psi = univar[2]*0.3 + 0.7 # AR parameter psi~U(0.7,1)

RWs  = RWsim (T, Y0, slope, trend, psi)

minv = min(RWs$RW_drift, RWs$RW, RWs$trend_stationary, RWs$ar1)
maxv = max(RWs$RW_drift, RWs$RW, RWs$trend_stationary, RWs$ar1)

plot.ts(RWs$RW_drift, ylim = c(minv, maxv), ylab = NA,
        main = "Plots of RW with drift (black), RW (magenta), \n trend+WN(blue) and stationary AR(1) (green)")
lines(ts(RWs$RW), col = "magenta")
lines(ts(RWs$trend_stationary), col = "blue")
lines(ts(RWs$ar1), col = "darkgreen")


# Test the simulated series for a unit root with ADF test
# It's implemented in many packages, including urca by Bernhard Pfaff and bootUR by Smeekes and Wilms

library(bootUR)
library(urca)

# Illustration: testing the simulated series for a unit root using different unit root tests

# ADF test, urca package
# This implementation also computes the F-test for the joint null hypotheses in Cases 2 and 3
RWdrift.dftr = ur.df(y=RWs$RW_drift, type='trend', selectlags = "AIC")
summary(RWdrift.dftr)
plot(RWdrift.dftr) #residual diagnostics; nothing of concern here

###############################################
# Augmented Dickey-Fuller Test Unit Root Test #
###############################################


#Test regression trend


#Call:
#  lm(formula = z.diff ~ z.lag.1 + 1 + tt + z.diff.lag) #one lagged difference included, however insignificant

#Residuals:
#  Min       1Q   Median       3Q      Max
#-2.38497 -0.70596  0.03921  0.66432  2.47088

#Coefficients:
#  Estimate Std. Error t value Pr(>|t|)
#(Intercept) -0.4612155  0.1720419  -2.681  0.00787 **
#  z.lag.1     -0.0534104  0.0191034  -2.796  0.00561 **
#  tt           0.0004153  0.0008895   0.467  0.64100       #trend term is insignificant, since the original RW without drift seems quite trending
#z.diff.lag  -0.0259040  0.0647825  -0.400  0.68962         #first lagged difference insignificant
#---
#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

#Residual standard error: 0.9365 on 234 degrees of freedom
#Multiple R-squared:  0.03669,	Adjusted R-squared:  0.02434
#F-statistic: 2.971 on 3 and 234 DF,  p-value: 0.03256

#The latter test statistic tests the joint H0: beta = 0, phi=1;
#However, don't trust the p-value! Rather use Table B.7 in the Appendix of Hamilton (1994)
#Rule of thumb: for Case 2 (constant only), value of the F-stat above 4.5 indicates significance at the 5% level,
#               for Case 3 (constant and trend), value of the F-stat above 5.3 indicates significance at the 5% level
#Here we're in Case 3, and fail to reject the joint null of a RW with drift vs. trend-stationarity

#Value of test-statistic is: -2.7959 2.9615 4.2088

# Above are the values of the following test statistics:
# tau3 = -2.7959: this is the value of the ADF t-statistic for testing H0: rho = 0 in Case 3.
# since it is greater than the 10% critical value of -3.13, we can't reject the null of a unit root.
# In case of rejection, there's no need to look at the further two test statistics.
# If the unit root is not rejected, we recall that H0 was actually a composite null hypothesis: H0: rho=0, beta=0
# The next reported statistic is phi3 = 2.9615. It refers to an F-test of the composite null  H0: rho=0, beta=0.
# Comparing it with the given critical values (tabulated by Dickey and Fuller (1981)), we fail to reject it.
# In this case we either have a unit root, or a nonzero trend term, or both.
# If beta=0 cannot be rejected by the regular t-test (as in our case, trend term is insignificant), we should re-run the ADF case in Case 2 (intercept only, no trend)
# Phi2 is an F-test for the composite null H0: rho=0, beta=0, mu=0.


#Critical values for test statistics:
#  1pct  5pct 10pct
#tau3 -3.99 -3.43 -3.13
#phi2  6.22  4.75  4.07
#phi3  8.43  6.49  5.47

####################################################

# Re-run the ADF test without a trend term
RWdrift.dfc = ur.df(y=RWs$RW_drift, type='drift', selectlags = "AIC")
summary(RWdrift.dfc)
plot(RWdrift.dfc)

################################################
# Augmented Dickey-Fuller Test Unit Root Test #
###############################################

#Test regression drift


#Call:
#  lm(formula = z.diff ~ z.lag.1 + 1 + z.diff.lag)

#Residuals:
#  Min       1Q   Median       3Q      Max
#-2.39349 -0.70087  0.04331  0.67409  2.43187

#Coefficients:
#  Estimate Std. Error t value Pr(>|t|)
#(Intercept) -0.41782    0.14454  -2.891   0.0042 **
#  z.lag.1     -0.05438    0.01896  -2.868   0.0045 **
#  z.diff.lag  -0.02439    0.06459  -0.378   0.7060
#---
#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

#Residual standard error: 0.935 on 235 degrees of freedom
#Multiple R-squared:  0.0358,	Adjusted R-squared:  0.02759
#F-statistic: 4.362 on 2 and 235 DF,  p-value: 0.0138


#Value of test-statistic is: -2.8683 4.3478

# The DF t-stat for Case 2  = -2.8683; we  can reject the null of a unit root only at the 10% level, but not at 5%
# The Phi1 statistic is an F-type test for H0: mu=rho=0 (again with non-standard distribution, tabulated by Dickey and Fuller (1981))
# We also fail to reject this one only at the 10% level, hinting at a possible problem with low power
# We note the one lagged difference included in the ADF regression is insignificant; remove it.

#Critical values for test statistics:
#  1pct  5pct 10pct
#tau2 -3.46 -2.88 -2.57
#phi1  6.52  4.63  3.81
#


RWdrift.dfc2 = ur.df(y=RWs$RW_drift, type='drift', lag = 0)
summary(RWdrift.dfc2)

# Conclusion remains qualitatively the same, though: our simulated RW with drift is classified as a pure RW at the 10% significance level.


# Now let's test the same series again by the ADF implementation in the bootUR package

# "Textbook" ADF based on asymptotic critical values
RWdrift.dfBUR1 =  adf(RWs$RW_drift, deterministics = "trend", min_lag = 0, criterion = "AIC", two_step = FALSE)
print(RWdrift.dfBUR1)

#One-step ADF test (with trend) on a single time series

#data: RWs$RW_drift
#null hypothesis: Series has a unit root
#alternative hypothesis: Series is stationary

#estimate largest root statistic p-value
#RWs$RW_drift                0.9479    -2.796  0.2004

# largest root is close to unity, numerical value of the ADF t-statistic is exactly the same as in the URCA package, also p-value is available

# Use the modified AIC by Ng and Perron (2001) instead of the AIC

RWdrift.dfBUR2 =  adf(RWs$RW_drift, deterministics = "trend", min_lag = 0, criterion = "MAIC", two_step = FALSE)
print(RWdrift.dfBUR2)

#estimate largest root statistic p-value
#RWs$RW_drift                0.9579    -2.078  0.5548

RWdrift.dfBUR2$details
# number of selected lags by MAIC is 8, hence the lower value of the test statistic (more lags -> less power)
# However, there's no guidance whether the trend specification was appropriate;
# this can be dealt with by the union of rejections approach, see below

# Bootstrap correction to improve power in small samples
RWdrift.dfBUR3 = boot_adf(data = RWs$RW_drift, bootstrap = "SB", deterministics = "trend", detrend = "OLS")
print(RWdrift.dfBUR3)
#retrieve the default specifications
RWdrift.dfBUR3$specifications
RWdrift.dfBUR3$details

# Bootstrap correction to improve power in small samples
RWdrift.dfBUR4 = boot_adf(data = RWs$RW_drift, bootstrap = "SB", deterministics = "trend", detrend = "QD")
print(RWdrift.dfBUR3)
#retrieve the default specifications
RWdrift.dfBUR4$specifications
RWdrift.dfBUR4$details

#SB bootstrap OLS test (with intercept and trend) on a single time series

#data: RWs$RW_drift
#null hypothesis: Series has a unit root
#alternative hypothesis: Series is stationary

#estimate largest root statistic p-value
#RWs$RW_drift                0.9479     -2.79  0.1851
#No lagged differences selected this time, p-value is a bit lower than in the first ADF test -> improved power by bootstrapping


#####################
# Union of rejections approach by Harvey et al. (2009, 2012): no choice of deterministic terms or detrending method required
# A very simple idea by Harvey (2009): perform two tests, one with intercept and one with trend
# thereby adjusting the significance level to control the size
# Harvey (2012): perform 4 tests at once:
# the above two versions of the ADF, eaich one with OLS detrending and with GLS-detrending ("quasi-differencing", QD / GLS-ADF)
# Bootstrapped versions implemented in bootUR, both for determining the size correction and the p-values
# 6 bootstrap methods available: the sieve bootstrap (SB), moving block bootstrap (MBB), sieve wild bootstrap
# (SWB), dependent wild bootstrap (DWB), block wild bootsrap (BWB) and autoregressive
# wild bootstrap (AWB)
# Any of these is good to correct for size distortions arising from neglected serial correlation
# Use Wild bootstrap for heteroskedastic data
# For panel unit root tests in panel data with cross-sectional dependence: don't use Sieve (SB/SWB), the rest from this package are OK
# Lag-length is re-selected at each replication

RWdrift.dfBootUnion = boot_union(data = RWs$RW_drift, bootstrap = "SWB", union_quantile = 0.05) #sign. level for the multiple testing procedure
print(RWdrift.dfBootUnion)
RWdrift.dfBootUnion$details

#data: RWs$RW_drift
#null hypothesis: Series has a unit root
#alternative hypothesis: Series is stationary

#estimate largest root statistic p-value
#RWs$RW_drift                    NA        -1  0.1426




# Exercise 5: Investigate the performance (empirical size and power) of the ADF test with misspecified deterministic terms
# For simplicity, use the "textbook" implementation with asymptotic critical values
# Case A: true process has intercept and no trend, but trend is included in the ADF regression
# Case B: true process has intercept and trend, but no trend is included in the ADF regression
# Use T = {70, 120, 240, 500} and nn = 1000


# Exercise 6: Compare the performance of the ADF test using asymptotic and bootstrapped critical values for different values of T
# We'd expect the bootstrapped version to perform better for small T

# For implementations of the Phillips-Perron tests
# and further unit root / stationarity tests see Chapter 5  of Pfaff (2008)
