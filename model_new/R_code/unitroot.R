# Note: all results below refer to the first simulated series with the given seed



# Exercise 1: Simulate a tri-variate cointegrated system
# by using Phillips' triangular representation
# Assume a single cointegrating vector beta = (1, -0.5, -0.5)

library(MASS)
set.seed(999)

T = 120 #10 years of monthly data, not too long
# in order to illustrate the low power of some of the tests
k = 3 # 3-variate system

#contemporaneous covariance matrix of the innovations
sigma = matrix(c(1, 0.2, 0.3, 0.2, 1, 0.2, 0.3, 0.2, 1), ncol = 3)

# start with iid (over time) innovations e
e = mvrnorm(T, mu = rep (0, k), Sigma = sigma)

#partial sum process of the innovations for the stochastic trends in the second and third variables Y_2 and Y_3
y_3 = cumsum(e[, 3])
y_2 = cumsum(e[, 2])

# Disequilibrium error/EC term Z_t: an AR(1) process
# with AR parameter phi = 0.5 serves as innovations to the first variable
z = arima.sim(model = list(ar = 0.5), n = T, innov = e[,1])
y_1 = 0.5*y_2 + 0.5*y_3 + z

# Write down the equations for this triangular system

# plot the simulated processes
minv = min(y_1, y_2, y_3, z)
maxv = max(y_1, y_2, y_3, z)

plot.ts(y_1, ylim = c(minv, maxv), ylab = NA,
        main = "Simulated CI system, k=3, r=1, beta = (1, -0.5, -0.5) with CI error Z")
lines(ts(y_2), lty = 3)
lines(ts(y_3), lty = 4)
lines(ts(z), col = "blue", lty = 1)
legend("bottomleft", legend = c("Y_1t", "Y_2t", "Y_3t", "Z_t"), lty = c(1,2,3,1), col=c("black", "black", "black", "blue"))

# We observe that the plot of Y_1t,
# defined by beta=(1,-0.5, -0.5) as the simple average of Y_2t and Y_3t,
# is approximately in the middle


# Exercise 1A: Simulate a 3-variate system with two cointegrating vectors:
# beta_1 = (1, -0.5 , 1) and beta_2 = (0, 1, -1).
# You give it a go!


# Exercise 2A, 3A, 4A:
# You may wish to investigate how the results of the cointegration tests and estimates from the (V)ECM models below
# change when you assume an AR parameter close to unity (e.g. 0.9 or 0.95) for the EC term Z_t




#########################################

# Exercise 2: Test for cointegration by Engle and Granger residual-based test for cointegration using
# A) pre-specified CI vector beta
# B) estimated CI vector beta
# Use the simulated series with a single CI relationship

# For Case A) we may use any unit root test, e.g. the ADF test,
# with its usual limiting distribution on Z
# Recall that the limiting distribution changes only if the CI vector is estimated from the data

library(bootUR)
z.adf = boot_adf(data = z, bootstrap = "SB", deterministics = "intercept", detrend = "QD")
print(z.adf)
# data: z
#null hypothesis: Series has a unit root
#alternative hypothesis: Series is stationary

#estimate largest root  statistic   p-value
#z         0.5662       -2.737      0.04852
# Hence we reject the null of a unit root in the CI residual Z at the 5% level

# Do you notice any problem with the test settings?




# As an exercise, test for cointegration using other pre-specified vectors,
# e.g. beta1 = (1, 0.5, 1) and beta2 =  (2, -1, -1). What do you observe in each case and why?




# Case B) Step 1: Estimate beta by OLS
lr.reg = lm(y_1 ~ y_2 + y_3)
summary(lr.reg) #Don't let the high significance fool you.
# Unless the reisduals are I(0), this regression is spurious!

# The estimated cointegrating vector is (-1, 0.46325, 0.63912)
# or, multiplied by -1, (1, -0.46325, -0.63912):
# close to our true beta = (1, -0.5, -0.5)

# The residuals of this regression are our estimated EC term (beta'*Y_t = Z_t)
zhat = residuals(lr.reg)
plot.ts(zhat, ylab = NA,
        main = "Estimated CI error Z")
# The residuals do look stationary, but we have to formally test them for a unit root

# Step 2: Test the estimated CI residual for a unit root
# We have several options here.

# Option 1: Use the ADF or the PP tests Z_alpha/Z_tau,
# but compare the value of the returned test statistic
# with the critical values in Phillips and Ouliaris (PO) (1990)
# or with those in Engle and Yoo (1987, Table 2)
# The former critical values are based on T=500,
# while the latter on T=200 (hence the slight differences after the 2nd decimal)


###############
# What about deterministic trends?
# Hansen, 1992, JoE: "It is suggested that trends be excluded in the levels regression for maximal efficiency"
# That is, we'd rather not include a determistic trend term in the cointegrating regression.
# Note, however, that for correct inference in practice we should also take the trending properties of the
# original series in levels into account:
# "The deterministic trends in the data affect the limiting distributions of the test statistics
# whether or not we detrend the data (Hansen, 1992, JoE) -> check the lecture notes!
# Hassler (2001): "Tests based on n integrated regressors with drifts but without detrending
# (i.e. not including a trend term in the CI regression)
# require the critical values for n-1 detrended regressors".

# Hansen's findings refer specifically to the ADF, Z_alpha and Z_tau tests of PO (1990)
# There is no mention of the newly proposed P_u and P_z tests of PO (1990),
# but this critique could apply to them as well

# Here, by simply looking at the plots of the three simulated series,
# we'll assume that all three are trending with a downward slope.

# That is, we are in Case 2 of Hansen's remarks,
# i.e. both the regressand Y_1 and the regressors Y_2 and Y_3
# can be thought of being I(1) with drift.


# Option 1A: Let's try the ADF test first
# No deterministic terms, since we're dealing with residuals,
# which are by construction zero-mean

zhat.adf1 = adf(zhat, deterministics = "none", min_lag = 0, criterion = "MAIC", two_step = FALSE)
print(zhat.adf1)

# ADF Zt (t-ratio) -statistic: -3.336
# Look up the critical values in Table IIc (Z_t_hat and ADF statistics (demeaned and detrended)), pp.190 of PO (1990)
# dimension parameter is k-2 = 1 (n = 1 explanatory variables )

# 10% critical value is -3.5184  and the 5% crit. value is -3.8000; 15% crit. value is -3.3283.

# -> We can only reject the null of a unit root in the CI regression residuals at the 15% level!

# Even if we assumed no drift in the variables, and were thus in Case 1 of Hansen's remark,
# looking up in Table IIb (demeaned = constant only), with dimension k-1=2 (i.e. n=2 explanatory variables)
# we'd have critical values - 3.4494 and - 3.767 at the 10% and 5% level, respectively
# -> same conclusions at the 10%/15% level.



# Option 1B: let's now try the PP Z_t (Z_tau) unit root test
# (but again, we won't "trust" the reported critical values,
# since we're dealing with estimated CI residuals and NOT with raw data in levels

library(urca)
zhat.ppZt = ur.pp(zhat, type="Z-tau", model="constant", lags="short")

# Z-tau can be viewed as "equivalent" statistic to the ADF test statistic:
# same limiting distribution, same divergence rate of sqrt(T) under the alternative of stationarity.

# Here the lags option "short" or "long" controls the number of lags to be included
# in the computation of the long-run variance.
# "short": lag order = int(4*(T/100)^{1/4})
# "long":  lag order = int(12*(T/100)^{1/4})

# There are only the options "constant" or "trend" for the deterministic terms,
# so no option to exclude the constant from the regression equation

summary(zhat.ppZt)
#Z-tau  is: -7.6195

# Compare with the same critical values as for the ADF test in Table IIc
# (Z_t_hat and ADF statistics (demeaned and detrended))
# Again, do not forget to reduce the dimension parameter
# to equal the number of explanatory variables minus one: 2-1=1
# 10% critical value is -3.5184 and the 5% one is -3.8000
# Now we can reject the null hypothesis of a unit root!
# The much lower negative value of the Z-tau statistic may be interpreted as indication
# of the higher power of PP's Z_t test in comparison with the ADF test
# (and of the fact that we've included a constant in the regression for the Z_t because we had no other option).


# Option 1C: let's try the more powerful Z_alpha statistic by Phillips and Perron (1988);
# it diverges at rate T under the alternative of stationarity, much like the DF-coefficient test,
# which doesn't have a corresponding version in the ADF test
# This is why it is also the one recommended by Phillips and Perron

zhat.ppZalpha = ur.pp(zhat, type="Z-alpha", model="constant", lags="short")
# again, only options "constant" or "trend" are available
summary(zhat.ppZalpha)
# Value of test-statistic, type: Z-alpha  is: -75.6265
# Now this is a very negative value!
# Again, remember that we're dealing with estimated CI residulas
# Again assume we're dealing with drifting RWs -> Case 2 from the lecture
# Thus compare Z-alpha with the critical values from Phillips and Oularis (1990),
# Table Ic (hat_Z_alpha statistic (demeaned and detrended)) for n=1=k-2 explanatory variable
# 10% critical value is -23.191 and the 5% crit. value is -27.086.
# We reject the null of a unit root in the CI residuals with far more certainty
# that with the ADF test!



# Option 2: Use the Phillips and Ouliaris (1990) tests as they are implemented in the urca and tsseries packages

# Note that PO (1990) consider altogether five different test statistics:
# ADF by Said and Dickey (1984), and Z_alpha and Z_tau(Z_t) by PP on the one hand,
# for which they show that their limiting distribution differs
# when they are applied to CI residuals Z_t as compared to raw data in levels.


# On the other hand, they propose two new tests:
# P_u, a variance ratio test statistic, and P_z - a multivariate trace statistic.
# The latter is not to be confused with Johansen's LR_trace statistic,
# although they both (P_z and LR_trace) share the
# feature that they are invariant to the ordering of the variables in the system, which is advantageous.

# The remaining four residual-based no-cointegration tests (ADF, Z_alpha, Z_t and P_u)
# do depend on which variable we normalize with respect to (i.e. we place it on the LHS of the CI regression).

# Both "new" PO (1990) tests - the P_u and P_z tests - are T-consistent,
# i.e. asymptotically as powerful as the Z_alpha,
# and thus more powerful than the sqrt(T)-consistent t-ratio ADF/Z_tau tests.

# For expressions of the Z_alpha and Z_tau statistics of Phillips and Perron (1988)
# you may refer to eq. 5.3a and 5.3b
# (5.5a and 5.5b, respectively, in the regression with detrending),
# pp. 95 of Pfaff (2008) (Analysis of integrated and cointegrated series with R)
# The PO P_u and P_z test statistics can be found in eq. 7.2 and 7.6 in Section 7.2 of Pfaff (2008)



# Option 2A: The Z_alpha test is implemented in the tseries package:
library(tseries)
zhat.ppZalpha2 = po.test(cbind(y_1, y_2, y_3), demean = TRUE, lshort = TRUE)
# demean=TRUE includes a constant term in the CI regression
# No detrended option is considered in this implementation
# Therefore, this function is NOT suitable for testing for cointegration e.g. in a
# system where Y_1 is trending, while the RHS regressors are not


# Phillips-Ouliaris demeaned = -82.604, Truncation lag parameter = 1, p-value = 0.01

# Different value than the PP Z-alpha test statistic computed with the PP Z_alpha test from the urca package...
# Differences might be due to different kernel method and/or bandwidth selection
# p-values are interpolated (unclear how) from Tables Ia and Ib of Phillips and Oullaris



# The two "new" PO tests for no cointegration:
# the P_u and the P_z tests, are implemented in the urca package


# Option 2B: Phillips and Ouliaris (1990) P_u test

zhat.PO_Pu = ca.po(cbind(y_1, y_2, y_3), demean = "const", lag="short", type  ="Pu")
summary(zhat.PO_Pu)

# ordering of the variables matters: the CI vector to be estimated is normalized w.r.t. the first variable
# other demean options are "trend" and "none"
# The value of the test statistic is: 90.4446
# No critical value is reported, so we have to again look up the critical values ourselves in the corresponding table.

# This time this is Table IIIc of PO(1990);
# Note that contrary to the previous three tests, here the rejection region for the null is in the RIGHT tail
# The 10% critical value with dimension parameter k-1=2 is 46.1061, the 5% critical value is 53.8300
# (if we were to extrapolate Hansen's results about the trend effects
# onto the limiting distribution also to this statistic,
# we'd look up at table IIIc with dimension parameter equal to 2-1=1;
# 10% crit. value is 41.2488 and 5% crit. value is 48.8439)
# Again, very strong evidence against the null of a unit root in the CI residual

# Option 2C: Phillips and Ouliaris (1990) P_z test

zhat.PO_Pz = ca.po(cbind(y_1, y_2, y_3), demean = "const", lag="short", type  ="Pz")
summary(zhat.PO_Pz)

# The value of the test statistic is: 120.5411
# Critical values of Pz are:
#  10pct    5pct     1pct
# 80.2034  89.7619  109.4525
# These are the critical values from Table IVb of PO (1990)
# Compare also with the corresponding crit. values from Table IVc: 71.96 81.38 and 102.02
# Hence again we reject H0: no cointegration.

# Therefore, we can safely conclude that there is cointegration between Y_1, Y_2 and Y_3.

############
# A small side exercise: The P_u and the P_z tests were proposed as tests for no cointegration;
# can you use them as unit root tests instead?
# I.e. apply them to first differences of I(1) regressors rather than residuals from a cointegrating regression.
# Try it out in a small simulation study. What do you observe?
############

# In order to be sure that there is only one CI relation, we need to check also the sub-systems
# Test for cointegration in the sub-systems by e.g. the P_z test only

# Y_1 and Y_2
lr.reg12 = lm(y_1 ~ y_2)
z12 = resid(lr.reg12)
ts.plot(z12)
ur.pp(z12, type="Z-alpha", model="constant", lags="short")
# The value of the test statistic is: -40.7287
# Based on the crit. values in Table IVb of PO for n=1 we cannot reject the null of no cointegration
# This is the result we expected, since we know the DGP and we know that only Y_1,Y_2 and Y_3 are cointegrated


# Y_1 and Y_3
lr.reg13 = lm(y_1 ~ y_3)
z13 = resid(lr.reg13)
ts.plot(z13)
ur.pp(z13, type="Z-alpha", model="constant", lags="short")
# The value of the test statistic is: -20.7934
# Based on the crit. values in Table IVb of PO (1990) for n=1 we cannot reject the null of no cointegration
# Again, this result is as expected


# Y_2 and Y_3
lr.reg23 = lm(y_2 ~ y_3)
z23 = resid(lr.reg23)
ts.plot(z23)
ur.pp(z23, type="Z-alpha", model="constant", lags="short")
# The value of the test statistic is: -4.8409
# Again, as expected, no cointegration is found

# You may test for cointegration in these bivariate subsystems also by the other residual-based tests discussed above as an exercise





#########################################

# Exercise 3: Fit an Error-Correction model

# We'll again use the data on the 3-variate system generated in Exercise 1
# with a single cointegrating relation beta = (1, -0.5, -0.5)
# We already know the three variables are cointegrated with a single CI relation,
# while neither two are individually cointegrated
# Suppose that the CI vector is unknown, but normalized w.r.t. to the first variable
# We can estimate the CI relation by OLS from the long-run relation in levels
# (we've done this in lr.reg) - it's T-consistent
# Hence we can plug it in in an ECM model for Y_1 as if it were known
# What we'll in fact use is not the estimate for the CI vector beta,
# but rather the residual zhat from lr.reg - it's lagged value is the EC term

dy1 = diff(y_1)
dy2 = diff(y_2)
dy3 = diff(y_3)

Tz = length(zhat)
z_l1 = zhat[-c((Tz-2), Tz)] # EC term; we just drop the last observation of z here
length(z_l1)

diff.dat = data.frame(embed(cbind(dy1, dy2, dy3), 2)) #puts together the first differences of Y with the first lagged differences
colnames(diff.dat)  =c('dy1', 'dy2', 'dy3', 'dy1.1', 'dy2.1', 'dy3.1')

ecm.reg = lm(dy1 ~ z_l1  + dy1.1 + dy2.1 + dy3.1, data = diff.dat)

# Exercise: Write down the equation we're estimating here for dy1

summary(ecm.reg)

# EC term and first own lagged difference of y_1 are highly significant
# There is Granger causality running from Y_2 and Y_3 to Y_1
# The speed of adjustment coefficient is -0.77:
# it's negative (as expected, given the normalization of the CI vector w.r.t Y_1)
# That is, 77% of the disequlibrium error will be corrected within the next observation period, or
# that it takes approx. 1/0.77 = 1.3 periods to return to equilibrium
# -> rather quick mean-reversion of the EC term


# As always, check the residuals:
uhat = residuals(ecm.reg)
ts.plot(uhat)
acf(uhat)
Box.test(uhat, lag = 10, type =  "Ljung-Box", fitdf = 4)
# Nice. No serial correlation, hence no need for the inclusion of more lagged differences


# What about validity of inference?
# Which coefficients of the ECM can we really interpret as statistically significant?

# We know that standard inference applies to the regressors which can represented as zero-mean I(0) series
# These are the short-run dynamics and the EC term
# Thus we can indeed trust their significance levels
# The estimator for the intercept is consistent,
# but its asymptotic distribution is not normal, hence interpret with caution


# How about estimating an ECM model for delta Y_2:
# We know that Y_1 and Y_3 are exogenous for it, but will this show up in the estimates?

ecm.regY2 = lm(dy2 ~ z_l1  + dy1.1 + dy2.1 + dy3.1, data = diff.dat)
summary(ecm.regY2)

# Nothing is significant:
# there's no error-correction mechanism driving Y_2 (as it should be, since we simulated it as a RW)


# There is an ecm package which seems to be designed to fit error-correction models
# However, I couldn't get it to include further lags of the differenced variables beyound the first lag
# This can be a serious shortcoming in applied work
# I'd rather recommend using the dynlm package
# (chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://cran.r-project.org/web/packages/dynlm/dynlm.pdf)






#########################################

# Exercise 4A: Estimate a VECM model

# Again, we're going to use our simulated 3-variate system Y = (Y_1, Y_2, Y_3)
Ydat = cbind(y_1, y_2, y_3)


# Step 1: Test for the cointegrating rank using Johansen's LR_trace and maximum eigenvalue test statistics

# Lag order selection: in levels

library(vars) #contributed again by Bernhard Pfaff -> we trust the validity of the implementation

VARselect(Ydat, lag.max = 6, type = "both") #type = "const", "trend", "both" or "none"
# all model selection criteria point to a VAR(1) in levels, which we know is the true DGP
# that is, no short-run dynamics in a VECM

# Step 2: ML Estimation/Computation of the LR trace test statistic for the cointegrating rank
# LR_trace test statistic: computed as a by-product from the ML estimation procedure
# Therefore, MLE:


H1 = ca.jo(Ydat, type = 'trace', K=2, spec = 'transitory', ecdet = 'const')

# type = 'trace' or 'eigen' computes either the LR trace test statistic or the LR max eigenvalue test statistic
# K specifies the lag order of the VAR in levels
# However, the minimum value of the lag parameter in the ca.jo function is K=2:
# that is, a VAR(2) in levels and hence a VAR(1) in first differences
# So we'll have to use K=2, even if it exceeds the true lag order of our process Y_t
# spec = 'long-run' or 'transitory': with 'transitory' the EC term is Pi*Y_{t-1}
# (the Gamma_j matrices measure the transitory impacts)
# while with 'long-run' the EC term is Pi*Y_{t-p}
# Both specifications of the VECM are valid, consult eq. 4.8a and 4.9a in Pfaff (2008)
# In the lectures we've worked with the 'transitory' form, hence this is what we are using here too.
# ecdet: Character, ‘none’ for no intercept in EC term, ‘const’ for constant term in EC term
# and ‘trend’ for trend variable in EC term
# Here we'll use the constant in the EC term specification,
# as we saw that our series are all trending in the same direction
# with no real discrepancy between the slopes which would justify including a trend in the EC term
# In general, one should be careful about including a trend in the EC term: difficult for interpretation

summary(H1)

######################
# Johansen-Procedure #
######################

#Test type: trace statistic , without linear trend and constant in cointegration

#Eigenvalues (lambda):
#  [1]  3.122476e-01  3.855883e-02  2.888187e-02 -2.575967e-18

# Even from the orders of magnitude of the estimated eigenvalues (0.31, 0.038, 0.028, 0.000)
# you can see that the first one looks significantly higher that zero,
# while the remaining ones are much closer to zero


#Values of test statistic and critical values of test: start from the lowest line, H0: r=0:

#           test 10pct  5pct  1pct
#r <= 2 |  3.46  7.52  9.24 12.97
#r <= 1 |  8.10 17.85 19.96 24.60   # the value of the test statistic 8.10 is lower than the 10% crit. value => we fail to reject H0: r<=1 in favour of H1: r>1. Thus we conclude that the CI rank is 1. We need not consider the upper row(s).
#r = 0  | 52.27 32.00 34.91 41.07   # the value of the test statistic is 52.27, much higher than the 1% crit. value -> reject H0: CI rank r=0. Move to the upper line.

# Eigenvectors, normalised to first column:
#  (These are the cointegration relations)
# Depending on the selected cointegrating rank r, we take only the first r columns of this matrix as our matrix beta
# Since we concluded that r=1, the first of these vectors is our CI vector

#             y_1.l1     y_2.l1     y_3.l1  constant
#y_1.l1    1.00000000   1.000000   1.000000  1.000000
#y_2.l1   -0.45276831   4.129669  -4.995685 -2.579256
#y_3.l1   -0.64545050  18.410059   5.729567 -3.738320
#constant  0.07175981 131.918421 -13.689515 -8.179262

# Indeed, in beta_hat = (1, -0.4527, -0.6454) we recognize our true beta (1. -0.5, -0.5)
# These ML estimates are also close to the OLS estimates from the levels CI regression, but not the same

#Weights W:
#  (This is the loading matrix)

#       y_1.l1       y_2.l1        y_3.l1      constant
#y_1.d -0.77035297 -0.003858694  0.0001180072 -3.126895e-17
#y_2.d -0.09305994 -0.001920556  0.0057624835 -1.587925e-18
#y_3.d  0.01648642 -0.003022230 -0.0038297285 -2.538686e-18

# Now this is the candidate alpha matrix, and recall, it also has dimensions k x r.
# Since r=1 in our case, we again select only the first column.
# Do you recognize -0.77 as the speed of adjustment coefficient in the equation for Delta Y_1?



# Testing for cointegration by the maximum eigenvalue test statistic:
H1_maxeig = ca.jo(Ydat, type = 'eigen', K=2, spec = 'transitory', ecdet = 'const')
summary(H1_maxeig)

# We again interpret the table with the Values of test statistic and critical values of test in the same way as above,
# starting from the bottom row H0: r=0.
# Again, the null of no cointegration is strongly rejected, while H0: r<=1 is not, thus pointing to r=1.


# Cointegrating vector and loading matrix are estimated simultaneously in the ca.jo command

# A small computational exercise: Compute the 3x3 matrix Pi from the output above
# Compute and plot estimates of the stochastic trends


