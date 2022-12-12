# Bayesian-Linear-Regression

The purpose of this task is to predict the temperatures in Jerusalem during the second half of a day
using linear regression and a couple of different basis functions. The temperatures to predict are for
November 16th, 2020, supplied in the file nov162020.npy.

The supplied file jerus_daytemps.npy1
contains data for the mean daily temperature of November in Givat Ram,
Jerusalem, at 8 different hours T = [02:00, 05:00, 08:00, 11:00, 14:00, 17:00, 20:00, 23:00]. The hours in T will be
the xs for the model, while the ys will be the temperatures. In this exercise we will fit the temperature at a time t
using linear regression:

![Linear_Reg_formula](/images/linear_reg_formula.png)

where:

![Linear_Reg](/images/linear_reg2.png)

## The Process
1. We fit a regular linear regression model to the first half of November 16, and predict the temperatures 
for the second half of the day.
2. We learned a prior by using the historical data provided.
3. We fit a bayesian linear regression model to the first half of the temperatures of November 16 to get the posterior
p(θ|D) and predict the temperatures in the second half of the day.

## The basis Functions

### Polynomial Basis Functions
These basis functions form a polynomial (as we discussed in class) and are given by:

![Poly_reg1](/images/polynomial_reg_1.png)


Because the numbers x^d can be quite large, using the polynomial basis functions as is may cause numerical problems.
To mitigate this, we can use the following (equivalent) basis functions:

![Poly_reg2](/images/polynomial_reg_2.png)

In this task I fit polynomials of degrees 3 and 7.

### Gaussian Basis Functions
The gaussian basis functions are defined as:

![Gaussian1](/images/gaussian_1.png)

where:

![Gaussian2](/images/gaussian_2.png)

for some beta and k values chosen ahead of time. The beta chosen for this task is 3, and the centers are:

![Centers](/images/centers.png)

### Cubic Regression Splines
Regression splines are, simply put, piece wise polynomial functions. A piece wise polynomial function is a function that is
made up of segments, each of which are defined by a different polynomial. In this sense, cubic regression splines are
functions made up of segments of independent 3rd order polynomials (hence cubic), with the added feature that we constrain the
2nd derivative to be continuous. Cubic regression splines are defined by the following set of basis
functions:

![Cubic_Reg1](/images/cubic_reg.png)

where:

![Cubic_Reg2](/images/cubic_reg2.png)

and the Xi's are chosen ahead of time. Those Xi's are called Knots. The knots chosen in this task are:

![Knots](/images/knots.png)

## Plots

![Plot1](/images/myplot1.png)

![Plot2](/images/myplot2.png)

![Plot3](/images/myplot3.png)

![Plot4](/images/myplot4.png)

![Plot5](/images/myplot5.png)

![Plot6](/images/myplot6.png)

![Plot7](/images/myplot7.png)

![Plot8](/images/myplot8.png)

![Plot9](/images/myplot9.png)

![Plot10](/images/myplot10.png)

![Plot11](/images/myplot11.png)

![Plot12](/images/myplot12.png)

![Plot13](/images/myplot13.png)

![Plot14](/images/myplot14.png)

![Plot15](/images/myplot15.png)

![Plot16](/images/myplot16.png)

![Plot17](/images/myplot17.png)

![Plot18](/images/myplot18.png)