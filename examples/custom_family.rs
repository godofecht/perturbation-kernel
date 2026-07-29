//! Plugging your own model into the engine.
//!
//! The three built-in families are demonstrations. The extension point
//! is the trait triple, and this example implements all three for a
//! question people actually ask.
//!
//! # The situation
//!
//! You fit a straight line to a noisy series and report a positive
//! slope. Your `y` values carry measurement error you can quantify.
//! Does the slope stay positive once you account for it?
//!
//! That is a sharper question than a p-value on the slope. A p-value
//! asks whether the slope differs from zero under an assumed noise
//! model. This asks the direct thing: *perturb the data by the error
//! I actually measured, refit, and count how often the sign holds.*
//!
//! # What it needs
//!
//! * a [`Perturbation`] -- jitter every observation by `N(0, theta)`;
//! * a [`ForwardModel`] -- refit, returning the slope;
//! * an [`Invariance`] -- the fraction of refits whose slope is
//!   positive.
//!
//! Run with `cargo run --release --example custom_family`.

use perturbation_kernel::config::{Backend, Config, Intensity, Lipschitz, Reduction};
use perturbation_kernel::engine::Engine;
use perturbation_kernel::forward::ForwardModel;
use perturbation_kernel::invariance::Invariance;
use perturbation_kernel::perturbation::Perturbation;
use perturbation_kernel::reduce;
use perturbation_kernel::report::Report;
use perturbation_kernel::Rng;
use rand_distr::{Distribution, Normal, Uniform};
use serde_json::json;

/// The observed series. `x` is the index, so only `y` is state.
type Series = Vec<f64>;

// =====================================================================
// C2: the perturbation family
// =====================================================================

/// Add `N(0, theta)` to every observation, with `theta ~ U[0, max]`.
///
/// The null parameter is `0.0`: at zero intensity the series comes
/// back unchanged, which is the contract that makes "perturb, then
/// recover" mean anything.
struct MeasurementError {
    sigma_max: f64,
}

impl Perturbation<Series> for MeasurementError {
    type Theta = f64;

    fn null(&self) -> f64 {
        0.0
    }

    fn sample_theta(&self, rng: &mut Rng) -> f64 {
        Uniform::new_inclusive(0.0, self.sigma_max).sample(rng)
    }

    fn apply(&self, s: &Series, theta: &f64, rng: &mut Rng) -> Series {
        let noise = Normal::new(0.0, *theta).expect("sigma >= 0");
        s.iter().map(|y| y + noise.sample(rng)).collect()
    }
}

// =====================================================================
// C3: the forward model
// =====================================================================

/// Ordinary least squares slope against the index.
///
/// This must be a pure function of the state: any randomness in the
/// readout belongs in the perturbation, not here.
struct FitSlope;

impl ForwardModel<Series, f64> for FitSlope {
    fn eval(&self, s: &Series) -> f64 {
        let n = s.len() as f64;
        let mean_x = (n - 1.0) / 2.0;
        let mean_y = reduce::mean(s);
        // Sum over i of (x_i - xbar)(y_i - ybar), and the same for
        // (x_i - xbar)^2. The denominator is the closed form for
        // consecutive integers.
        let cov: f64 = s
            .iter()
            .enumerate()
            .map(|(i, y)| (i as f64 - mean_x) * (y - mean_y))
            .sum();
        let var_x = n * (n * n - 1.0) / 12.0;
        cov / var_x
    }

    /// The slope is a fixed linear functional of `y`, so its Lipschitz
    /// constant is the norm of that functional's weights.
    fn lipschitz(&self) -> Option<f64> {
        Some(1.0)
    }
}

// =====================================================================
// C4: the invariance functional
// =====================================================================

/// Fraction of the ensemble whose slope kept the sign we reported.
///
/// Permutation-invariant by construction (it is a mean), and reduced
/// through [`reduce::mean`] so the tree order is the one the schema
/// mandates.
struct SignSurvival {
    reported_positive: bool,
}

impl Invariance<f64> for SignSurvival {
    fn measure(&self, ensemble: &[f64]) -> Report {
        let held: Vec<f64> = ensemble
            .iter()
            .map(|slope| {
                let positive = *slope > 0.0;
                if positive == self.reported_positive {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        Report::raw(
            reduce::mean(&held),
            self.name(),
            ensemble.len() as u64,
            0,
            Reduction::default(),
        )
    }

    /// The readout is an indicator on `{0, 1}`, so the mean is
    /// 1-Lipschitz in the Wasserstein-1 metric.
    fn lipschitz_w1(&self) -> Option<f64> {
        Some(1.0)
    }

    fn name(&self) -> &str {
        "sign_survival"
    }
}

// =====================================================================

fn config(n: u64, seed: u64) -> Config {
    Config {
        schema_version: "1.0.0".into(),
        n,
        seed,
        intensity: Intensity {
            kind: "uniform_interval".into(),
            params: json!({ "low": 0.0, "high": "sigma_max" }),
            null_parameter: json!(0.0),
        },
        reduction: Reduction::default(),
        lipschitz: Lipschitz {
            forward_l: Some(1.0),
            invariance_lambda: Some(1.0),
        },
        accuracy: None,
        backend: Backend::Auto,
    }
}

fn main() {
    // Twelve observations with a genuine upward trend of 0.1 per step,
    // plus some scatter that is already baked into the data.
    let series: Series = vec![
        1.02, 1.05, 1.31, 1.28, 1.55, 1.49, 1.71, 1.88, 1.94, 2.05, 2.28, 2.21,
    ];
    let reported = FitSlope.eval(&series);

    println!("Does my reported slope survive its own measurement error?");
    println!();
    println!("observations   {}", series.len());
    println!("fitted slope   {reported:+.5} per step");
    println!(
        "reported sign  {}",
        if reported > 0.0 {
            "positive"
        } else {
            "negative"
        }
    );
    println!();

    let n = 200_000;
    println!("Sign survival against the measurement error you assume");
    println!();
    println!(
        "{:>14}  {:>14}  {:>12}  verdict",
        "error sigma", "sign survives", "+/- bound"
    );
    println!("{}", "-".repeat(60));

    // The ladder runs well past the spread of the data itself. Stopping
    // where the answer is still "solid" would only show that the
    // question had not been pushed hard enough to have an answer.
    let mut limit = None;
    for sigma in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0] {
        let cfg = config(n, 20260610);
        let report = Engine::run(
            &series,
            &MeasurementError { sigma_max: sigma },
            &FitSlope,
            &SignSurvival {
                reported_positive: reported > 0.0,
            },
            &cfg,
        )
        .expect("valid config");

        // Wilson-free eyeball interval: the standard error of a
        // proportion at this n is at most 1/(2 sqrt(n)).
        let se = 0.5 / (n as f64).sqrt();
        if report.value <= 0.95 && limit.is_none() {
            limit = Some(sigma);
        }
        let verdict = if report.value > 0.99 {
            "solid"
        } else if report.value > 0.95 {
            "holds"
        } else if report.value > 0.8 {
            "shaky"
        } else {
            "GONE"
        };
        println!(
            "{sigma:14.2}  {:14.4}  {:12.5}  {verdict}",
            report.value, se
        );
    }

    println!();
    let spread = series.iter().cloned().fold(f64::MIN, f64::max)
        - series.iter().cloned().fold(f64::MAX, f64::min);
    match limit {
        Some(s) => {
            println!("Sign survival drops below 95% at sigma = {s}, which is");
            println!(
                "{:.1}x the full spread of the data ({spread:.2}).",
                s / spread
            );
            println!();
            println!("That is the number to quote. It says the reported direction");
            println!("holds unless your measurement error is several times larger");
            println!("than the entire range of what you measured -- and if it were,");
            println!("you would have bigger problems than this trend.");
        }
        None => {
            println!(
                "The sign held at every error level tried, up to {:.0}x the",
                16.0 / spread
            );
            println!("full spread of the data. Push the ladder further to find");
            println!("where it breaks; as it stands there is no measured limit.");
        }
    }

    println!();
    println!("Note what did not appear anywhere above: a null hypothesis, a");
    println!("test statistic, or a threshold someone picked by convention. The");
    println!("question was answered in the units of the actual instrument.");
}
