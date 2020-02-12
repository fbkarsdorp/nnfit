data {
  int<lower=3> N;
  vector[N] y;
}

transformed data {
  real mean_y = mean(y);
  real sd_y = sd(y);
  real normal_sigma = sd_y * 100;
  real lambda = 1 / 30.0;
  real y_low = sd_y / 1000;
  real y_high = sd_y * 1000;
}

parameters {
  real mu;
  real <lower=0> nu;
  real<lower=0> sigma;
}

model {
  sigma ~ uniform(y_low, y_high);
  mu ~ normal(mean_y, normal_sigma);
  nu ~ exponential(1.0 / 30.0);
  y ~ student_t(nu, mu, sigma);
}
