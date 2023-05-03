Download Link: https://assignmentchef.com/product/solved-cs1675-homework-3-laplace-approximation-for-the-gaussian-likelihood
<br>
This homework assignment has two primary goals. First, you will program the Laplace Approximation for the Gaussian likelihood with an unknown mean and unknown noise. Second, you will be introduced to resampling. You will use 5-fold cross-validation to identify that a more complex model is overfitting to the training set, and is thus fooled by noise.

Completing this assignment requires filling in missing pieces of information from existing code chunks, programming complete code chunks from scratch, typing discussions about results, and working with LaTeX style math formulas. A template .Rmd file is available to use as a starting point for this homework assignment. The template is available on CourseWeb.

<strong>IMPORTANT:</strong> Please pay attention to the eval flag within the code chunk options. Code chunks with eval=FALSE will <strong>not</strong> be evaluated (executed) when you Knit the document. You <strong>must</strong> change the eval flag to be eval=TRUE. This was done so that you can Knit (and thus render) the document as you work on the assignment, without worrying about errors crashing the code in questions you have not started. Code chunks which require you to enter all of the required code do not set the eval flag. Thus, those specific code chunks use the default option of eval=TRUE.

<h2>Load packages</h2>

This assignment uses the dplyr and ggplot2 packages, which are loaded in the code chunk below. The resampling questions will make use of the modelr package, which we will load later in this report. The assignment also uses the tibble package to create tibbles, and the readr package to load CSV files. All of the listed packages are part of the tidyverse and so if you downloaded and installed the tidyverse already, you will have all of these packages. This assignment will use the MASS package to generate random samples from a MVN distribution. The MASS package should be installed with base R, and is listed with the System Library set of packages.

library(dplyr)library(ggplot2)

<h2>Problem 1</h2>

You are tasked by a manufacturing company to study the variability of a component that they produce. This particular component has an important figure of merit that is measured on every single manufactured piece. The engineering teams feel that the measurement is quite noisy. Your goal will be to learn the unknown mean and unknown noise of this figure of merit based on the noisy observations.

The data set that you will work with is loaded in the code chunk below and assigned to the df_01 object. As the glimpse() function shows, df_01 consists of two variables, obs_id and x. There are 405 rows or observations. obs_id is the “observation index” and x is the measured value.

df_01 &lt;- readr::read_csv(“https://raw.githubusercontent.com/jyurko/CS_1675_Spring_2020/master/hw_data/hw03/df_01.csv”,                          col_names = TRUE)## Parsed with column specification:## cols(##   obs_id = col_double(),##   x = col_double()## )df_01 %&gt;% glimpse()## Observations: 405## Variables: 2## $ obs_id &lt;dbl&gt; 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, …## $ x      &lt;dbl&gt; 7.3299610, 4.0119121, -0.5633241, -8.2168261, -1.307001…

<h3>1a)</h3>

Let’s start out by visualizing and summarizing the available measurements.

<h4>PROBLEM</h4>

<strong>Create a “run chart” of the measurments by piping the </strong><strong>df_01</strong><strong> data set into </strong><strong>ggplot()</strong><strong>. Map the </strong><strong>x</strong><strong> aesthetic to </strong><strong>obs_id</strong><strong> and the </strong><strong>y</strong><strong> aesthetic to </strong><strong>x</strong><strong>. Use </strong><strong>geom_line()</strong><strong> and </strong><strong>geom_point()</strong><strong> geometric objects to display the measured value with respect to the observation index.</strong>

<strong>Visually, do any values appear to stand out from the “bulk” of the measurements?</strong>

<h4>SOLUTION</h4>

Your solution here.

<h3>1b)</h3>

<h4>PROBLEM</h4>

<strong>Create a histogram with </strong><strong>geom_histogram()</strong><strong>, by mapping the </strong><strong>x</strong><strong> aesthetic to the variable </strong><strong>x</strong><strong>. Set the number of bins to be 25.</strong>

<strong>Based on the histogram and the run chart, where are most of the measurement values concentrated? What shape does the distribution look like?</strong>

<h4>SOLUTION</h4>

Your solution here.

<h3>1c)</h3>

<h4>PROBLEM</h4>

<strong>Use the </strong><strong>summary()</strong><strong> function to summarize the </strong><strong>df_01</strong><strong> data set. Then calculate the 5th and 95th quantile estimates of the variable </strong><strong>x</strong><strong>. Lastly, calculate the standard deviation on the </strong><strong>x</strong><strong> variable. Are the summary statistics in line with your conclusions based on the run chart and histogram?</strong>

<em>HINT</em>: The quantiles can be estimated via the quantile() function. Type ?quantile into the R Console for help with the arguments to the quantile function.

<h4>SOLUTION</h4>

### apply the summary function  ### calculate the quantile estimates  ### calculate the empirical standard deviation of x

?

<h3>1d)</h3>

The summary statistics in Problem 1c) were estimated by using all of the 405 observations. The manufacturing company produces several other components which could also be measured. It will take a considerable effort to measure the same number of components as are available in the current data set. Therefore, it is important to understand the behavior of the sample average with respect to the number of observations.

In dplyr we can calculate “rolling” sample averages with the cummean() function. This function “accumulates” all of the observations up to a current index and calculates the sample average. We can call the cummean() function directly within a mutate() call, which allows considerable flexibility, since we do not have use for-loops to create this operation. Let’s visualize how the “rolling sample average” changes over time.

<h4>PROBLEM</h4>

<strong>Pipe the </strong><strong>df_01</strong><strong> data set into a </strong><strong>mutate()</strong><strong> call. Create a new column (variable) in the data set, </strong><strong>rolling_avg</strong><strong>, which is equal to the rolling average of </strong><strong>x</strong><strong>. Pipe the result into </strong><strong>ggplot()</strong><strong> and map the </strong><strong>x</strong><strong> aesthetic to </strong><strong>obs_id</strong><strong> and the </strong><strong>y</strong><strong> aesthetic to </strong><strong>rolling_avg</strong><strong>. Display the rolling average with respect to the observation index with the </strong><strong>geom_line()</strong><strong> geometric object.</strong>

<strong>Which portion of the rolling average appears the most variable? Does the rolling average steady-out at all?</strong>

<h4>SOLUTION</h4>

Your solution here.

<h3>1e)</h3>

In this problem, you will focus on two specific sample sizes. You will calculate the sample average after 5 observations and 45 observations.

<h4>PROBLEM</h4>

<strong>What does the sample average equal to after 5 observations? What does the sample average equal to after 45 observations?</strong>

<h4>SOLUTION</h4>

Your solution here.

<h2>Problem 2</h2>

With a simple Exploratory Data Analysis (EDA) completed, it’s time to begin modeling. You are interested in learning the unknown mean of the figure of merit and the unknown noise in the measurement process. Because the manufacturing company is interested in understanding the behavior of the estimates “over time”, it will be critical to assess uncertainty in the unknowns at low sample sizes. Therefore, you use a Bayesian approach to estimating the unknown mean, (mu), and unknown noise, (sigma).

The figure of merit, x, is a continuous variable, and after discussions with the engineering teams, it seems that a Guassian likelihood is a reasonable assumption to make. You will assume that each observation is conditionally independent of the others given (mu) and (sigma). You will also assume that the two unknowns are a-priori independent. The joint posterior distribution on (mu) and (sigma) conditioned on the observations (mathbf{x}) is therefore proportional to:

[ pleft(mu,sigma mid mathbf{x} right) propto prod_{n=1}^{N} left( mathrm{normal}left( x_n mid mu,sigma right) right) times pleft(muright) times pleft( sigma right) ]

<h3>2a)</h3>

You will be applying the Laplace Approximation to learn the unknown (mu) and (sigma). In lecture, and in homework 02, it was discussed that it is a good idea to first transform (sigma) to a new variable (varphi), before executing the Laplace Approximation.

<h4>PROBLEM</h4>

<strong>Why is it a good idea to transform </strong><strong>(sigma)</strong><strong> before performing the Laplace Approximation?</strong>

<h4>SOLUTION</h4>

?

<h3>2b)</h3>

Consider a general transformation, or link function, applied to the unknown (sigma):

[ varphi = gleft (sigma right) ]

The inverse link function which allows “back-transforming” from (varphi) to (sigma) is:

[ sigma = g^{-1} left( varphi right) ]

<h4>PROBLEM</h4>

<strong>Write out the joint-posterior between the unknown </strong><strong>(mu)</strong><strong> and unknown transformed parameter </strong><strong>(varphi)</strong><strong> by accounting for the probability change-of-variables formula. You must keep the prior on </strong><strong>(mu)</strong><strong> and </strong><strong>(sigma)</strong><strong> general for now, and make use of the general inverse link function notation. Denote the likelihood as a Gaussian likelihood, as shown in the Problem 2 description.</strong>

<h4>SOLUTION</h4>

?

<h3>2c)</h3>

After discussions with the engineering teams at the manufacturing company, it was decided that a normal prior on (mu) and an Exponential prior on (sigma) are appropriate. The normal prior is specified in terms of a prior mean, (mu_0), and prior standard deviation, (tau_0). The Exponential prior is specified in terms of a rate parameter, (lambda). The prior distributions are written out for you below.

[ pleft(muright) = mathrm{normal}left( mu mid mu_0,tau_0 right) ]

[ pleft(sigmaright) = mathrm{Exp}left( sigma mid lambda right) ]

Because you will be applying the Laplace approximation, you decide to use a log-transformation as the link function:

[ varphi = log left( sigma right) ] The engineering team feels fairly confident that the measurement process is rather noisy. They feel the noise is obscuring a mean figure of merit value that is actually negative. For that reason, it is decided to set the hyperparameters to be (mu_0 = -1/2), (tau_0 = 1), and (lambda = 1/3). You are first interested in how their prior beliefs change due to the first 5 observations.

You will define a function to calculate the un-normalized log-posterior for the unknown (mu) and (varphi). You will follow the format discussed in lecture, and so the hyperparameters and measurements will be supplied within a list. The list of the required information is defined for you in the code chunk below. The hyperparameter values are specified, as well as the first 5 observations are stored in the variable xobs. The list is assigned to the object info_inform_N05 to denote the prior on the unknown mean is informative and the first 5 observations are used.

info_inform_N05 &lt;- list(  xobs = df_01$x[1:5],  mu_0 = -0.5,  tau_0 = 1,  sigma_rate = 1/3)

<h4>PROBLEM</h4>

<strong>You must complete the code chunk below in order to define the </strong><strong>my_logpost()</strong><strong> function. This function is in terms of the unknown mean </strong><strong>(mu)</strong><strong> and unknown transformed parameter </strong><strong>(varphi)</strong><strong>. Therefore you MUST account for the change-of-variables transformation. The first argument to </strong><strong>my_logpost()</strong><strong> is a vector, </strong><strong>unknowns</strong><strong>, which contains all of the unknown parameters to the model. The second argument </strong><strong>my_info</strong><strong>, is the list of required information. The unknown mean and unknown transformed parameter are extracted from </strong><strong>unknowns</strong><strong>, and are assigned to the </strong><strong>lik_mu</strong><strong> and </strong><strong>lik_varphi</strong><strong> variables, respectively. You must complete the rest of the code chunk. Comments provide hints for what calculations you must perform.</strong>

<strong>You ARE allowed to use built-in </strong><strong>R</strong><strong> functions to evaluate densities.</strong>

<em>HINT</em>: the info_inform_N05 list of information provides to you the names of the hyperparameter and observation variables to use in the my_logpost() function. This way all students will use the same nomenclature for evaluating the log-posterior.

<em>HINT</em>: The second code chunk below provides three test cases for you. If you correctly coded the my_logpost() function, the first test case which sets unknowns = c(0, 0) will yield a value of -76.75337. The second test case which sets unknowns = c(-1, -1) yields a value of -545.4938. The third test case which sets unknowns = c(1, 2) yields a value of -19.49936. If you do not get three values very close to those printed here, there is a typo in your function.

<h4>SOLUTION</h4>

my_logpost &lt;- function(unknowns, my_info){  # extrack the unknowns  lik_mu &lt;- unknowns[1]  lik_varphi &lt;- unknowns[2]    # backtransform to sigma  lik_sigma &lt;-     # calculate the log-likelihood  log_lik &lt;-     # calculate the log-prior on each parameter  log_prior_mu &lt;-     log_prior_sigma &lt;-     # calculate the log-derivative adjustment due to the   # change-of-variables transformation  log_deriv_adjust &lt;-     # sum all together  }

Check test cases.

my_logpost(c(0, 0), info_inform_N05) my_logpost(c(-1, -1), info_inform_N05) my_logpost(c(1, 2), info_inform_N05)

<h3>2d)</h3>

Because this problem consists of just two unknowns, we can graphically explore the true log-posterior surface. We did this in lecture by studying in both the original (left(mu, sigmaright)) space, as well as the transformed and unbounded (left(mu, varphiright)) space. However, in this homework assignment, you will continue to focus on the transformed parameters (mu) and (varphi).

In order to visualize the log-posterior surface, you must define a grid of parameter values that will be applied to the my_logpost() function. A simple way to create a full-factorial grid of combinations is with the expand.grid() function. The basic syntax of the expand.grid() function is shown in the code chunk below for two variables, x1 and x2. The x1 variable is a vector of just two values c(1, 2), and the variable x2 is a vector of 3 values 1:3. As shown in the code chunk output printed to the screen the expand.grid() function produces all 6 combinations of these two variables. The variables are stored as columns, while their combinations correspond to a row within the generated object. The expand.grid() function takes care of the “book keeping” for us, to allow varying x2 for all values of x1.

expand.grid(x1 = c(1, 2),            x2 = 1:3,            KEEP.OUT.ATTRS = FALSE,            stringsAsFactors = FALSE) %&gt;%   as.data.frame() %&gt;% tbl_df()## # A tibble: 6 x 2##      x1    x2##   &lt;dbl&gt; &lt;int&gt;## 1     1     1## 2     2     1## 3     1     2## 4     2     2## 5     1     3## 6     2     3

You will now make use of the expand.grid() function to create a grid of candidate combinations between (mu) and (varphi).

<h4>PROBLEM</h4>

<strong>Complete the code chunk below. The variables </strong><strong>mu_lwr</strong><strong> and </strong><strong>mu_upr</strong><strong> correspond to the lower and upper bounds in the grid for the unknown </strong><strong>(mu)</strong><strong> parameter. Specifiy the lower bound to be the 0.01 quantile (1st percentile) based on the prior on </strong><strong>(mu)</strong><strong>. Specify the upper bound to the 0.99 quantile (99th percentile) based on the prior on </strong><strong>(mu)</strong><strong>. The variables </strong><strong>varphi_lwr</strong><strong> and </strong><strong>varphi_upr</strong><strong> are the lower and upper bounds in the grid for the unknown </strong><strong>(varphi)</strong><strong> parameter. Specify the lower bound to be the log of the 0.01 quantile (1st percentile) based on the prior on </strong><strong>(sigma)</strong><strong>. Specificy the upper bound to be the log of the 0.99 quantile(99th percentile) based on the prior on </strong><strong>(sigma)</strong><strong>.</strong><strong>Create the grid of candidate values, </strong><strong>param_grid</strong><strong>, by setting the input vectors to the </strong><strong>expand.grid()</strong><strong> function to be 201 evenly spaced points between the defined lower bound and upper bounds on the paraemters.</strong>

<em>HINT</em>: remember that probability density functions in R each have their own specific quantile functions…

<h4>SOLUTION</h4>

### bounds on mumu_lwr &lt;- mu_upr &lt;-  ### bounds on varphivarphi_lwr &lt;- varphi_upr &lt;-  ### create the grid param_grid &lt;- expand.grid(mu = ,                          varphi = ,                          KEEP.OUT.ATTRS = FALSE,                          stringsAsFactors = FALSE) %&gt;%   as.data.frame() %&gt;% tbl_df()

<h3>2e)</h3>

The my_logpost() function accepts a vector as the first input argument, unknowns. Thus, you cannot simply pass in the separate columns of the param_grid data set. To overcome this, you will define a “wrapper” function, which manages the call to the log-posterior function. The wrapper, eval_logpost() is started for you in the code chunk below. The first argument to eval_logpost() is a value for the unknown mean, the second argument is a value to the unknown (varphi) parameter, the third argument is the desired log-posterior function, and the fourth argument is the list of required information.

This problem tests that you understand how to call a function, and how the input arguments to the log-posterior function are structured. You will need to understand that structure in order to perform the Laplace Approximation later on.

<h4>PROBLEM</h4>

<strong>Complete the code chunk below such that the user supplied </strong><strong>my_func</strong><strong> function has the </strong><strong>mu_val</strong><strong> and </strong><strong>varphi_val</strong><strong> variables combined into a vector with the correct order.</strong>

<strong>Once you complete the </strong><strong>eval_logpost()</strong><strong> function, the second code chunk below uses the </strong><strong>purrr</strong><strong> package to calculate the log-posterior over all combinations in the grid. The result is stored in a vector </strong><strong>log_post_inform_N05</strong><strong>.</strong>

<h4>SOLUTION</h4>

eval_logpost &lt;- function(mu_val, varphi_val, my_func, my_info){  my_func( , my_info)}log_post_inform_N05 &lt;- purrr::map2_dbl(param_grid$mu,                                       param_grid$varphi,                                       eval_logpost,                                       my_func = my_logpost,                                       my_info = info_inform_N05)

<h3>2f)</h3>

The code chunk below defines the viz_logpost_surface() function for you. It generates the log-posterior surface contour plots in the style presented in lecture. You are required to interpret the log-posterior surface and describe the most probable values and uncertainty in the parameters.

viz_logpost_surface &lt;- function(log_post_result, grid_values){  gg &lt;- grid_values %&gt;%     mutate(log_dens = log_post_result) %&gt;%     mutate(log_dens_2 = log_dens – max(log_dens)) %&gt;%     ggplot(mapping = aes(x = mu, y = varphi)) +    geom_raster(mapping = aes(fill = log_dens_2)) +    stat_contour(mapping = aes(z = log_dens_2),                 breaks = log(c(0.01/100, 0.01, 0.1, 0.5, 0.9)),                 color = “black”) +    scale_fill_viridis_c(guide = FALSE, option = “viridis”,                         limits = log(c(0.01/100, 1.0))) +    labs(x = expression(mu), y = expression(varphi)) +    theme_bw()    print(gg)}

<h4>PROBLEM</h4>

<strong>Call the </strong><strong>viz_logpost_surface()</strong><strong> function by setting the </strong><strong>log_post_result</strong><strong> argument equal to the </strong><strong>log_post_inform_N05</strong><strong> vector, and the </strong><strong>grid_values</strong><strong> argument equal to the </strong><strong>param_grid</strong><strong> object.</strong>

<strong>Which values of the transformed unbounded </strong><strong>(varphi)</strong><strong> parameter are completely ruled out, even after 5 observations? Which values for the unknown mean, </strong><strong>(mu)</strong><strong>, appear to be the most probable? Is the posterior uncertainty on </strong><strong>(mu)</strong><strong> small compared to the prior uncertainty?</strong>

<h4>SOLUTION</h4>

viz_logpost_surface( )

?

<h3>2g)</h3>

The log-posterior visualization in Problem 2f) is based just on the first 5 observations. You must now repeat the analysis with 45 and all 405 observations. The code chunk below defines the two lists of required information, assuming the informative prior on (mu). You will use these two lists instead of the info_inform_N05 to recreate the log-posterior surface contour visualization.

info_inform_N45 &lt;- list(  xobs = df_01$x[1:45],  mu_0 = -0.5,  tau_0 = 1,  sigma_rate = 1/3) info_inform_N405 &lt;- list(  xobs = df_01$x,  mu_0 = -0.5,  tau_0 = 1,  sigma_rate = 1/3)

<h4>PROBLEM</h4>

<strong>Complete the two code chunks below. The first code chunk requires you to complete the </strong><strong>purrr::map2_dbl()</strong><strong> call, which evaluates the log-posterior for all combinations of the parameters. By careful that you use the correct list of required information. The </strong><strong>log_post_inform_N45</strong><strong> object corresponds to the 45 sample case, while the </strong><strong>log_post_info_N405</strong><strong> object corresponds to using all of the samples.</strong>

<strong>The second and third code chunks require that you call the </strong><strong>viz_logpost_surface()</strong><strong> function for the posterior based on 45 samples, and the posterior based on all 405 samples, respectively.</strong>

<strong>How does the log-posterior surface change as the sample size increases? Does the most probable values of the two parameters change as the sample size increases? Does the uncertainty in the parameters decrease?</strong>

<h4>SOLUTION</h4>

### check the arguments from param_grid!!!log_post_inform_N45 &lt;- purrr::map2_dbl(param_grid$,                                       param_grid$,                                       eval_logpost,                                       my_func = my_logpost,                                       my_info = ) ### check the arguments from param_grid!!!log_post_inform_N405 &lt;- purrr::map2_dbl(param_grid$,                                        param_grid$,                                        eval_logpost,                                        my_func = my_logpost,                                        my_info = info_inform_N405)viz_logpost_surface( )viz_logpost_surface( )

?

<h2>Problem 3</h2>

It’s now time to work with the Laplace Approximation. The first step is to find the posterior mode, or the maximum a posteriori (MAP) estimate. An iterative optimization scheme is required to do so for most of the posteriors you will use in this course. As described in lecture, you will use the optim() function to perform the optimization.

<h3>3a)</h3>

The code chunk below defines two different initial guesses for the unknowns. You will try out both initial guesses and compare the optimization results. You will focus on the 5 observation case.

init_guess_01 &lt;- c(-1, -1)init_guess_02 &lt;- c(2, 2)

<h4>PROBLEM</h4>

<strong>Complete the two code chunks below. The first code chunk finds the posterior mode (the MAP) based on the first initial guess </strong><strong>init_guess_01</strong><strong> and the second code chunk uses the second initial guess </strong><strong>init_guess_02</strong><strong>. You must fill in the arguments to the </strong><strong>optim()</strong><strong> call in order to approximate the posterior based on 5 observations.</strong>

<strong>To receive full credit you must: specify the initial guesses correctly, specify the function to be optimized correctly, specify the gradient evaluation correctly, correctly pass in the list of required information for 5 samples, tell </strong><strong>optim()</strong><strong> to return the hessian matrix, and ensure that </strong><strong>optim()</strong><strong> is trying to maximize the log-posterior rather than attempting to minimize it.</strong>

<h4>SOLUTION</h4>

map_res_01 &lt;- optim(,                    ,                    gr = ,                    ,                    method = ,                    hessian = ,                    control = )map_res_02 &lt;- optim(,                    ,                    gr = ,                    ,                    method = ,                    hessian = ,                    control = )

<h3>3b)</h3>

You tried two different starting guesses…will you get the same optimized results?

<h4>PROBLEM</h4>

<strong>Compare the two optimization results from Problem 3a). Are the identified optimal parameter values the same? Are the Hessian matrices the same? Was anything different?</strong>

<strong>What about the log-posterior surface gave you a hint about how the two results would compare?</strong>

<h4>SOLUTION</h4>

Your solution here.

<h3>3c)</h3>

Finding the mode is the first step in the Laplace Approximation. The second step uses the negative inverse of the Hessian matrix as the approximate posterior covariance matrix. You will use a function, my_laplace(), to perform the complete Laplace Approximation. This way, this one function is all that’s needed in order to perform all steps of the Laplace Approximation.

<h4>PROBLEM</h4>

<strong>Complete the code chunk below. The </strong><strong>my_laplace()</strong><strong> function is adapted from the </strong><strong>laplace()</strong><strong> function from the </strong><strong>LearnBayes</strong><strong> package. Fill in the missing pieces to double check that you understand which portions of the optimization result correspond to the mode and which are used to approximate the posterior covariance matrix.</strong>

<h4>SOLUTION</h4>

my_laplace &lt;- function(start_guess, logpost_func, …){  # code adapted from the `LearnBayes“ function `laplace()`  fit &lt;- optim(,               ,               gr = NULL,               …,               method = “BFGS”,               hessian = TRUE,               control = list(fnscale = -1, maxit = 1001))    mode &lt;-   h &lt;-   p &lt;- length(mode)  # we will discuss what int means in a few weeks…  int &lt;- p/2 * log(2 * pi) + 0.5 * log(det(h)) + logpost_func(mode, …)  list(mode = ,       var_matrix = ,       log_evidence = int,       converge = ifelse(fit$convergence == 0,                         “YES”,                          “NO”),       iter_counts = fit$counts[1])}

<h3>3d)</h3>

You now have all of the pieces in place to perform the Laplace Approximation. Execute the Laplace Approximation for the 5 sample, 45 sample, and 405 sample cases.

<h4>PROBLEM</h4>

<strong>Call the </strong><strong>my_laplace()</strong><strong> function in order to perform the Laplace Approximation based on the 3 sets of observations you studied with the log-posterior surface visualizations. Check that each converged.</strong>

<h4>SOLUTION</h4>

laplace_res_inform_N05 &lt;- my_laplace( ) laplace_res_inform_N45 &lt;- my_laplace( ) laplace_res_inform_N405 &lt;- my_laplace( ) ### check if they eached converged

<h3>3e)</h3>

The MVN approximate posteriors that you solved for in Problem 3d) are in the (left( mu, varphi right)) space. In order to help the manufacturing company, you will need to back-transform from (varphi) to (sigma), while accounting for any potential posterior correlation with (mu). A simple way to do so is through random sampling. The MVN approximate posteriors are a known type of distribution, a Multi-Variate Normal (MVN). You are therefore able to call a random number generator which uses the specified mean vector and specified covariance matrix to generate random samples from a MVN. Back-transforming from (varphi) to (sigma) is accomplished by simply using the inverse link function.

The code chunk below defines the generate_post_samples() function for you. The user provides the Laplace Approximation result as the first argument, and the number of samples to make as the second argument. The MASS::mvrnorm() function is used to generate the posterior samples. Almost all pieces of the function are provided to you, except you <strong>must</strong> complete the back-transformation from (varphi) to (sigma) by using the correct inverse link function.

<h4>PROBLEM</h4>

<strong>Complete the code chunk below by using the correct inverse link function to back-transform from </strong><strong>(varphi)</strong><strong> to </strong><strong>(sigma)</strong><strong>.</strong>

<h4>SOLUTION</h4>

generate_post_samples &lt;- function(mvn_info, N){  MASS::mvrnorm(n = N,                 mu = mvn_info$mode,                 Sigma = mvn_info$var_matrix) %&gt;%     as.data.frame() %&gt;% tbl_df() %&gt;%     purrr::set_names(c(“mu”, “varphi”)) %&gt;%     mutate(sigma = ) ### backtransform to sigma!!!}

<h3>3f)</h3>

1e4 posterior samples based on the 3 different observation sample sizes are generated for you in the code chunk below.

set.seed(200121)post_samples_inform_N05 &lt;- generate_post_samples(laplace_res_inform_N05, N = 1e4)post_samples_inform_N45 &lt;- generate_post_samples(laplace_res_inform_N45, N = 1e4)post_samples_inform_N405 &lt;- generate_post_samples(laplace_res_inform_N405, N = 1e4)

You will summarize the posterior samples and visualize the posterior marginal histograms.

<h4>PROBLEM</h4>

<strong>Use the </strong><strong>summary()</strong><strong> function to summarize the posterior samples based on each of the three observation sample sizes. What are the posterior mean values on </strong><strong>(mu)</strong><strong> and </strong><strong>(sigma)</strong><strong> as the number of observations increase? Visualize the posterior histograms on </strong><strong>(mu)</strong><strong> and </strong><strong>(sigma)</strong><strong> based on each of the three observation sample sizes.</strong>

<h4>SOLUTION</h4>

Your solution here.

<h3>3g)</h3>

Another benefit of working with posterior samples rather than the posterior density is that it is relatively straight forward to answer potentially complex questions. At the start of this project, the engineering teams felt that mean value of the figure of merit was negative. You can now tell them the probability of that occuring. Additionally, the engineering teams felt that the measurement process was rather noisy. You can provide uncertainty intervals on the unknown (sigma). Depending on the amount of uncertainty about the noise, that might help them decide if company should invest in more precise measurement equipment.

<h4>PROBLEM</h4>

<strong>What is the posterior probability that </strong><strong>(mu)</strong><strong> is positive for each of the three observation sample sizes? What are the 0.05 and 0.95 quantiles on </strong><strong>(sigma)</strong><strong> for each of the three observation sample sizes?</strong>

<h4>SOLUTION</h4>

Your solution here.

<h2>Problem 4</h2>

In this problem you get first hand experience working with the lm() function in R. You will make use of that function to fit simple to complex models on low and high noise data sets. You will then use cross-validation to assess if the models are overfitting to a training set.

Two data sets are in for you in the code chunks below. Both consist of an input x and a continuous response y. The two data sets are synthetic. Both come from the same underlying true functional form. The true functional form is not given to you. You are given random observations generated around that true functional form. The df_02_low data set corresponds to a random observations with low noise, while the df_02_high data set corresponds to random observations with high noise.

df_02_low &lt;- readr::read_csv(“https://raw.githubusercontent.com/jyurko/CS_1675_Spring_2020/master/hw_data/hw03/df_02_low.csv”,                              col_names = TRUE)## Parsed with column specification:## cols(##   x = col_double(),##   y = col_double()## )df_02_high &lt;- readr::read_csv(“https://raw.githubusercontent.com/jyurko/CS_1675_Spring_2020/master/hw_data/hw03/df_02_high.csv”,                              col_names = TRUE)## Parsed with column specification:## cols(##   x = col_double(),##   y = col_double()## )

Show a glimpse of each data set below.

df_02_low %&gt;% glimpse()## Observations: 30## Variables: 2## $ x &lt;dbl&gt; 1.18689051, 0.74887469, 0.48150224, -0.63588243, -0.10975984…## $ y &lt;dbl&gt; -1.18839785, -0.22018125, 0.52781651, -1.66456080, 0.0454377…df_02_high %&gt;% glimpse()## Observations: 30## Variables: 2## $ x &lt;dbl&gt; 1.18689051, 0.74887469, 0.48150224, -0.63588243, -0.10975984…## $ y &lt;dbl&gt; 1.3047004, 6.4105645, -2.5582175, 5.6122242, 0.6981725, -1.3…

<h3>4a)</h3>

Create scatter plots in ggplot2 between the response y and the input x.

<h4>PROBLEM</h4>

<strong>Use </strong><strong>ggplot()</strong><strong> with </strong><strong>geom_point()</strong><strong> to visualize scatter plots between the response and the input for both the low and high noise data sets. Since you know that </strong><strong>df_02_low</strong><strong> corresponds to low noise, can you make a guess about the true functional form that generated the data?</strong>

<h4>SOLUTION</h4>

Your solution here.

<h3>4b)</h3>

The lm() function in R is quite flexible. You can use a formula interface to specify models of various functional relationships between the response and the inputs. To get practice working with the formula interface you will create three models for the low noise case and three models for the high noise case. You will specificy a linear relationship between the response and the input, a quadratic relationship, and an 8th order polynomial.

There are many ways the polynomials can be created. For this assignment though, you will use the I() function to specify the polynomial terms. This approach will seem quite tedious, but the point is to get practice working with the formula interface. We will worry about efficiency later in the semester. The formula interface for stating the response, y, has a cubic relationship with the input x is just:

y ~ x + I(x^2) + I(x^3)

The ~ operator reads as “is a function of”. Thus, the term to the left of ~ is viewed as a response, and the expression to the right of ~ is considered to be the features or predictors. With the formula specified the only other required argument to lm() is the data object. Thus, when using the formula interface, the syntax to creating a model with lm() is:

&lt;model object&gt; &lt;- lm(&lt;output&gt; ~ &lt;input expression&gt;, data = &lt;data object&gt;)

<h4>PROBLEM</h4>

<strong>Create a linear relationship, quadratic relationship, and 8th order polynomial model for the low and high noise cases. Use the formula interface to define the relationships.</strong>

<h4>SOLUTION</h4>

The low noise case models are given below.

mod_low_linear &lt;-  mod_low_quad &lt;-  mod_low_8th &lt;-

The high noise case models are given in the code chunk below.

mod_high_linear &lt;-  mod_high_quad &lt;-  mod_high_8th &lt;-

<h3>4c)</h3>

There are many different approaches to inspect the results of lm(). A straightforward way is to use the summary() function to display a print out of each model’s fit to the screen. Here, you will use summary() to read the R-squared performance metric associated with each model.

<h4>PROBLEM</h4>

<strong>Use the </strong><strong>summary()</strong><strong> function to print out the summary of each model’s fit. Which of the three relationships had the highest R-squared for the low noise case? Which of the three relationships had the highest R-squared for the high noise case?</strong>

<h4>SOLUTION</h4>

Your solution here.

<h3>4d)</h3>

As mentioned previously, there are many methods in R to extract the performance of a lm() model. The modelr package, which is within the tidyverse, includes some useful functions for calculating the Root Mean Squared Error (RMSE) and R-squared values. The syntax for calculating the RMSE with the modelr::rmse() function is:

modelr::rmse(&lt;model object&gt;, &lt;data set&gt;)

The data supplied to modelr::rmse() can be a new data set, as long as the response is included in the data set. modelr::rmse() manages the book keeping for making predictions, comparing those predictions to the observed responses, calculating the errors, and summarizing. We will be discussing all of these steps later in the semester. For now, it’s practical to know a function to quickly compute an important quantity such as RMSE.

We will use the modelr package for the remainder of Problem 4, so the code chunk below loads it into the current session.

library(modelr)

<h4>PROBLEM</h4>

<strong>Use the </strong><strong>modelr::rmse()</strong><strong> function to calculate the RMSE for each of the models on their corresponding training sets. Thus, calculate the RMSE of the low noise case models with respect to the low noise data set. Calculate the RMSE of the high noise case models with respect to the high noise data set.</strong>

<h4>SOLUTION</h4>

The low noise case RMSEs are calculated below.

modelr::rmse( )modelr::rmse( )modelr::rmse( )

The high noise case RMSEs are calculated below.

modelr::rmse( )modelr::rmse( )modelr::rmse( )

<h3>4e)</h3>

The previous performance metrics were calculated based on the training set alone. We know that only considering the training set performance metrics such as RMSE and R-squared will prefer overfit models, which simply chase the noise in the training data. To assess if a model is overfit, we can breakup a data set into multiple training and test splits via cross-validation. For the remainder of this problem, you will work with 5-fold cross-validation to get practice working with multiple data splits.

You will not have create the data splits on your own. In fact, the data splits are created for you in the code chunk below, for the low noise case. The modelr::crossv_kfold() function has two main input arguments, data and k. The data argument is the data set we wish to performance k-fold cross-validation on. The k argument is the number of folds to create. There is a third argument, id, which allows the user to name the fold “ID” labels, which by default are named “.id”. As previously stated, the code chunk below creates the 5-fold data splits for the low noise case for you.

set.seed(23413)cv_info_k05_low &lt;- modelr::crossv_kfold(df_02_low, k = 5)

The contents of the cv_info_k05_low object are printed for you below. As you can see, cv_info_k05_low contains 3 columns, train, test, and .id. Although the object appears to be a data frame or tibble, the contents are not like most data frames. The train and test columns are actually lists which contain complex data objects. These complex objects are pointer-like in that they store how to access the training and test data sets from the original data set. In this way, the resampled object can be more memory efficient than just storing the actual data splits themselves. The .id column is just an ID, and so each row in cv_info_k05_low is a particular fold.

cv_info_k05_low## # A tibble: 5 x 3##   train        test         .id  ##   &lt;named list&gt; &lt;named list&gt; &lt;chr&gt;## 1 &lt;resample&gt;   &lt;resample&gt;   1    ## 2 &lt;resample&gt;   &lt;resample&gt;   2    ## 3 &lt;resample&gt;   &lt;resample&gt;   3    ## 4 &lt;resample&gt;   &lt;resample&gt;   4    ## 5 &lt;resample&gt;   &lt;resample&gt;   5

There are several ways to access the data sets directly. You can convert the resampled objects into integers, which provide the row indices associated with the train or test splits for each fold. To access the indices you need to use the [[]] format since you are selecting an element in a list. For example, to access the row indices for all rows selected in the first fold’s training set:

as.integer(cv_info_k05_low$train[[1]])##  [1]  2  3  4  5  6  8  9 10 11 13 14 15 16 17 18 21 22 23 24 25 26 27 29## [24] 30

Likewise to access the row indices for the third fold’s test set:

as.integer(cv_info_k05_low$test[[3]])## [1]  4 11 17 21 26 30

By storing pointers, the resample objects are rather memory efficient. We can make use of functional programming techniques to quickly and efficiently train and test models across all folds. In this assignment, though, you will turn the resampled object into separate data sets. Although not memory efficient, doing so allows you to work with each fold directly.

<h4>PROBLEM</h4>

<strong>Convert each training and test split within each fold to a separate data set. To do so, use the </strong><strong>as.data.frame()</strong><strong> function instead of the </strong><strong>as.integer()</strong><strong> function. The object names in the code chunks below denote training or test and the particular fold to assign. You only have to work with the resampled object based on the low noise data set.</strong>

<h4>SOLUTION</h4>

The fold training sets should be specified in the code chunk below.

low_noise_train_fold_01 &lt;- low_noise_train_fold_02 &lt;- low_noise_train_fold_03 &lt;- low_noise_train_fold_04 &lt;- low_noise_train_fold_05 &lt;-

The fold test sets should be specified in the code chunk below.

low_noise_test_fold_01 &lt;-low_noise_test_fold_02 &lt;-low_noise_test_fold_03 &lt;-low_noise_test_fold_04 &lt;- low_noise_test_fold_05 &lt;-

<h3>4f)</h3>

With the training and test splits available, now it’s time to train the models on each training fold split. You can ignore the linear relationship model for the remainder of the assignment, and focus just on the quadratic relationship and 8th order polynomial.

<h4>PROBLEM</h4>

<strong>Fit or train the quadratic relationship and 8th order polynomial using each fold’s training split. Use the formula interface the define the relationship in each </strong><strong>lm()</strong><strong> call.</strong>

<h4>SOLUTION</h4>

The quadratic relationship fits should be specified below.

mod_low_quad_fold_01 &lt;-  mod_low_quad_fold_02 &lt;-  mod_low_quad_fold_03 &lt;-  mod_low_quad_fold_04 &lt;-  mod_low_quad_fold_05 &lt;-

The 8th order polynomial fits should be specified below.

mod_low_8th_fold_01 &lt;-  mod_low_8th_fold_02 &lt;-  mod_low_8th_fold_03 &lt;-  mod_low_8th_fold_04 &lt;-  mod_low_8th_fold_05 &lt;-

<h3>4g)</h3>

Let’s compare the RMSE of the quadratic relationship to the 8th order polynomial, within each training fold. In this way, we can check that even after splitting the data, comparing models based on their training data still favors more complex models.

<h4>PROBLEM</h4>

<strong>Calculate the RMSE for the quadratic relationship, using each fold’s model fit, relative to the training splits. Thus, you should calculate 5 quantities, and store the 5 quantities in a vector named </strong><strong>cv_low_quad_fold_train_rmse</strong><strong>. Perform the analogous operation for the 8th order polynomial fits, and store in the vector named </strong><strong>cv_low_8th_fold_train_rmse</strong><strong>.</strong>

<strong>Calculate the average RMSE across the 5-folds for both relationships. Which relationship has the lowest average RMSE?</strong>

<h4>SOLUTION</h4>

Your solution here.

<h3>4h)</h3>

Now, it’s time to compare the quadratic and 8th order polynomial performance in the 5 test splits. Repeat the steps you performed in Problem 4g), but this time with the test splits, instead of the training splits.

<h4>PROBLEM</h4>

<strong>Calculate the test split RMSEs for each fold for the quadratic and 8th order polynomial. Assign the results to </strong><strong>cv_low_quad_fold_test_rmse</strong><strong> and </strong><strong>cv_low_8th_fold_test_rmse</strong><strong>, for the quadratic and 8th order polynomial, respectively. Which relationship has the lowest average cross-validation RMSE on the test splits?</strong>

<h4>SOLUTION</h4>

Your response here.