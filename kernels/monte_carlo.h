/* (c) 2014 Maximilian Gerhard Wolter (2009956434) */

# include <random.h>

// results (no cv):
// arithmetic mean (running per path)
// -> payoff (per path)
//    -> value (per path)
//       -> value mean (running total)
//       -> value stddev (running total)

// results (cv):
// arithmetic mean (running per path)
// geometric mean (running per path)
// -> payoff (per path)
// -> cv payoff (per path)
//    -> value (per path)
//    -> cv value (per path)
//       -> value mean (running total)
//       -> cv value mean (running total)
//       -> (cv value * value) mean (running total)
// write back to host:
// * value
// * cv value
// * value mean (partial of total solution)
// * cv value mean (partial of total solution)
// * (cv value * value) mean (partial of total solution)
// calculate on host:
// * value mean (total solution)
// * cv value mean (total solution)
// * (cv value * value) mean (total solution)
// * cv value variance
// * covariance
// * theta
// * z values
// * z mean
// * z stddev

float monte_carlo_asian_asset_path(float start_price, float risk_free_rate, float volatility, float maturity, unsigned int steps, __private stdnormal_float_prng_state *prng_state)
{
	float delta_t = maturity/steps;

	float drift = exp((risk_free_rate - 0.5*volatility*volatility) * delta_t);

	float growth_factor = drift * exp(volatility * sqrt(delta_t) * stdnormal_float_random(prng_state));
}
