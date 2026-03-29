# Cookbook Recipes

End-to-end examples showing complete Scivex workflows.

## 1. Exploratory Data Analysis

Load a CSV, compute summary statistics, group and aggregate, then visualize.

```rust
use scivex::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Load data
    let df = read_csv_path("sales.csv")?;
    println!("Shape: {:?}", df.shape());

    // Descriptive statistics on a numeric column
    let revenue_col = df.column("revenue")?;
    let data = vec![100.0, 200.0, 150.0, 300.0, 250.0];
    let stats = describe(&data);
    println!("{:?}", stats);

    // Group by region and compute mean
    let grouped = df.groupby(&["region"])?;

    // Visualize: bar chart of revenue by region
    let regions = vec!["North", "South", "East", "West"];
    let totals = vec![45000.0, 32000.0, 28000.0, 51000.0];

    let fig = Figure::new().plot(
        Axes::new()
            .title("Revenue by Region")
            .x_label("Region")
            .y_label("Total Revenue ($)")
            .add_plot(
                BarPlot::new(
                    regions.into_iter().map(String::from).collect(),
                    totals,
                )
                .color(Color::BLUE),
            ),
    );

    fig.save_svg("revenue_by_region.svg")?;
    Ok(())
}
```

## 2. Linear Regression with Visualization

Fit a linear model and plot the regression line.

```rust
use scivex::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Generate sample data
    let x_data: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
    let y_data: Vec<f64> = x_data.iter().map(|&x| 2.5 * x + 1.0 + (x * 7.0).sin() * 0.3).collect();

    // Prepare tensors for regression
    let n = x_data.len();
    let x_train = Tensor::from_vec(
        x_data.iter().map(|&x| vec![x]).flatten().collect(),
        vec![n, 1],
    )?;
    let y_train = Tensor::from_vec(y_data.clone(), vec![n])?;

    // Train linear regression
    let mut model = LinearRegression::new();
    model.fit(&x_train, &y_train)?;

    // Predict
    let predictions = model.predict(&x_train)?;
    let pred_vec: Vec<f64> = predictions.as_slice().to_vec();

    // Compute R-squared
    let r2 = scivex_ml::metrics::r2_score(&y_data, &pred_vec);
    println!("R2 = {:.4}", r2);

    // Plot data + regression line
    let fig = Figure::new().plot(
        Axes::new()
            .title("Linear Regression")
            .x_label("x")
            .y_label("y")
            .add_plot(ScatterPlot::new(x_data.clone(), y_data).color(Color::BLUE))
            .add_plot(LinePlot::new(x_data, pred_vec).color(Color::RED)),
    );

    fig.save_svg("regression.svg")?;
    Ok(())
}
```

## 3. Classification Pipeline

Train a random forest with preprocessing and cross-validation.

```rust
use scivex::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Create sample data: 200 samples, 4 features
    let n = 200;
    let x = Tensor::from_vec(
        (0..n * 4).map(|i| (i as f64) * 0.01).collect(),
        vec![n, 4],
    )?;
    let y = Tensor::from_vec(
        (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect(),
        vec![n],
    )?;

    // Train/test split
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, Some(42));

    // Preprocessing: standardize features
    let mut scaler = StandardScaler::new();
    let x_train_scaled = scaler.fit_transform(&x_train)?;
    let x_test_scaled = scaler.transform(&x_test)?;

    // Train random forest
    let mut rf = RandomForestClassifier::new(100);
    rf.fit(&x_train_scaled, &y_train)?;

    // Predict and evaluate
    let preds = rf.predict(&x_test_scaled)?;
    let acc = scivex_ml::metrics::accuracy(
        &y_test.as_slice().to_vec(),
        &preds.as_slice().to_vec(),
    );
    println!("Accuracy: {:.2}%", acc * 100.0);

    // Cross-validation
    let cv_scores = cross_val_score(&mut rf, &x_train_scaled, &y_train, 5)?;
    println!("CV scores: {:?}", cv_scores);

    Ok(())
}
```

## 4. Neural Network: MNIST-Style Digit Classification

Build a simple feedforward network for digit classification.

```rust
use scivex::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Define model architecture
    let model = Sequential::new()
        .add(Linear::new(784, 256))
        .add(ReLU::new())
        .add(Dropout::new(0.3))
        .add(Linear::new(256, 128))
        .add(ReLU::new())
        .add(Dropout::new(0.2))
        .add(Linear::new(128, 10));

    // Optimizer with learning rate scheduling
    let mut optimizer = Adam::new(0.001);
    let mut scheduler = StepLR::new(0.001, 5, 0.5); // decay by 0.5 every 5 epochs

    // Training loop (using synthetic data for illustration)
    let x_batch = Variable::new(
        Tensor::<f64>::zeros(vec![32, 784]), // batch of 32 images
        false,
    );
    let y_batch = Variable::new(
        Tensor::<f64>::zeros(vec![32, 10]), // one-hot labels
        false,
    );

    for epoch in 0..20 {
        let output = model.forward(&x_batch);
        let loss = cross_entropy_loss(&output, &y_batch);

        loss.backward();
        optimizer.step(model.parameters());
        optimizer.zero_grad(model.parameters());

        let lr = scheduler.step();
        println!("Epoch {}: loss = {:.4}, lr = {:.6}", epoch, loss.data().sum(), lr);
    }

    // Save model weights
    save_weights(model.parameters(), "model.bin")?;

    Ok(())
}
```

## 5. Optimization: Curve Fitting

Fit an exponential decay model to noisy data.

```rust
use scivex::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Model: y = a * exp(-b * x) + c
    let model = |x: f64, params: &[f64]| -> f64 {
        params[0] * (-params[1] * x).exp() + params[2]
    };

    // Generate noisy data
    let x_data: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
    let y_data: Vec<f64> = x_data
        .iter()
        .enumerate()
        .map(|(i, &x)| 3.0 * (-0.5 * x).exp() + 1.0 + (i as f64 * 0.1).sin() * 0.1)
        .collect();

    // Initial parameter guess: [a, b, c]
    let p0 = vec![1.0, 1.0, 0.0];

    // Fit
    let result = curve_fit(model, &x_data, &y_data, &p0)?;
    println!("Fitted params: a={:.3}, b={:.3}, c={:.3}",
        result.params[0], result.params[1], result.params[2]);
    println!("Converged: {}, iterations: {}", result.converged, result.iterations);

    // Plot original data vs fitted curve
    let y_fit: Vec<f64> = x_data.iter().map(|&x| model(x, &result.params)).collect();

    let fig = Figure::new().plot(
        Axes::new()
            .title("Exponential Decay Fit")
            .x_label("x")
            .y_label("y")
            .add_plot(ScatterPlot::new(x_data.clone(), y_data).color(Color::BLUE))
            .add_plot(LinePlot::new(x_data, y_fit).color(Color::RED)),
    );

    fig.save_svg("curve_fit.svg")?;
    Ok(())
}
```

## 6. Text Classification with TF-IDF

Tokenize text, vectorize with TF-IDF, and classify.

```rust
use scivex::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Sample documents
    let docs = vec![
        "the cat sat on the mat",
        "the dog chased the cat",
        "birds fly in the sky",
        "fish swim in the sea",
        "cats and dogs are pets",
        "birds and fish are animals",
    ];
    let labels = vec![0.0, 0.0, 1.0, 1.0, 0.0, 1.0]; // 0 = mammals, 1 = others

    // Tokenize
    let tokenizer = WordTokenizer::new();
    let tokenized: Vec<Vec<String>> = docs
        .iter()
        .map(|d| tokenizer.tokenize(d))
        .collect();

    // TF-IDF vectorization
    let mut tfidf = TfidfVectorizer::new();
    let features = tfidf.fit_transform(&tokenized)?;

    // Train a logistic regression classifier
    let y = Tensor::from_vec(labels, vec![6])?;
    let mut model = LogisticRegression::new();
    model.fit(&features, &y)?;

    // Predict on new text
    let new_doc = tokenizer.tokenize("the parrot flies high");
    let new_features = tfidf.transform(&[new_doc])?;
    let prediction = model.predict(&new_features)?;
    println!("Prediction: {}", prediction.as_slice()[0]);

    Ok(())
}
```

## 7. Signal Processing: Spectral Analysis

Analyze a signal's frequency content using FFT and plot the spectrogram.

```rust
use scivex::prelude::*;
use scivex_core::fft;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Generate a signal: 440 Hz sine + 880 Hz sine
    let sample_rate = 8000.0;
    let duration = 1.0;
    let n = (sample_rate * duration) as usize;

    let signal: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64 / sample_rate;
            (2.0 * std::f64::consts::PI * 440.0 * t).sin()
                + 0.5 * (2.0 * std::f64::consts::PI * 880.0 * t).sin()
        })
        .collect();

    // Compute FFT
    let input = Tensor::from_vec(signal.clone(), vec![n])?;
    let spectrum = fft::rfft(&input)?;
    let magnitudes: Vec<f64> = spectrum.as_slice()
        .chunks(2)
        .map(|c| (c[0] * c[0] + c[1] * c[1]).sqrt())
        .collect();

    // Frequency axis
    let freqs: Vec<f64> = (0..magnitudes.len())
        .map(|i| i as f64 * sample_rate / n as f64)
        .collect();

    // Plot magnitude spectrum (first half)
    let half = magnitudes.len() / 2;
    let fig = Figure::new().plot(
        Axes::new()
            .title("Frequency Spectrum")
            .x_label("Frequency (Hz)")
            .y_label("Magnitude")
            .add_plot(LinePlot::new(
                freqs[..half].to_vec(),
                magnitudes[..half].to_vec(),
            ).color(Color::BLUE)),
    );

    fig.save_svg("spectrum.svg")?;
    Ok(())
}
```

## 8. Symbolic Math: Taylor Series

Compute and plot a Taylor series approximation.

```rust
use scivex_sym::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Define f(x) = sin(x)
    let x = var("x");
    let f = sin(x.clone());

    // Taylor series around x=0, order 7
    let series = taylor(&f, "x", 0.0, 7);
    let simplified = simplify(&series);

    println!("sin(x) Taylor series (order 7):");
    println!("  {}", simplified);

    // Evaluate at x = pi/4
    let val = simplified.eval(&[("x", std::f64::consts::FRAC_PI_4)]);
    println!("Approximation at pi/4: {:.6}", val);
    println!("Exact sin(pi/4):       {:.6}", std::f64::consts::FRAC_PI_4.sin());

    // Symbolic derivative
    let df = diff(&f, "x");
    let df_simplified = simplify(&df);
    println!("d/dx sin(x) = {}", df_simplified); // cos(x)

    Ok(())
}
```

## 9. Graph Analysis: Social Network

Build a graph, compute centrality, and find shortest paths.

```rust
use scivex_graph::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Build a social network graph
    let mut g = Graph::<&str, f64>::new();
    g.add_edge("Alice", "Bob", 1.0);
    g.add_edge("Alice", "Carol", 1.0);
    g.add_edge("Bob", "Dave", 1.0);
    g.add_edge("Carol", "Dave", 1.0);
    g.add_edge("Dave", "Eve", 1.0);
    g.add_edge("Carol", "Eve", 1.0);

    // Shortest path
    let path = scivex_graph::shortest::dijkstra(&g, &"Alice", &"Eve")?;
    println!("Shortest path Alice -> Eve: {:?}", path);

    // PageRank
    let pr = scivex_graph::centrality::pagerank(&g, 0.85, 100)?;
    println!("PageRank scores: {:?}", pr);

    // Connected components
    let components = scivex_graph::connectivity::connected_components(&g)?;
    println!("Number of components: {}", components.len());

    Ok(())
}
```

## 10. Image Processing Pipeline

Load an image, apply filters, detect edges, and save.

```rust
use scivex_image::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Load a BMP image
    let img = scivex_image::io::read_bmp("input.bmp")?;
    println!("Image size: {}x{}", img.width(), img.height());

    // Convert to grayscale
    let gray = scivex_image::color::grayscale(&img)?;

    // Apply Gaussian blur
    let blurred = scivex_image::filter::gaussian_blur(&gray, 3, 1.0)?;

    // Detect edges with Sobel
    let edges = scivex_image::filter::sobel_edges(&blurred)?;

    // Detect corners
    let corners = scivex_image::features::harris_corners(&gray, 3, 0.04, 0.01)?;
    println!("Found {} corners", corners.len());

    // Save result
    scivex_image::io::write_bmp("edges.bmp", &edges)?;

    Ok(())
}
```

## 11. Binary Classification with Hist-Boosting + SHAP

Train a histogram gradient boosting classifier, explain predictions with SHAP values,
and visualize feature importance.

```rust
use scivex::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Generate synthetic data: 500 samples, 8 features
    let mut rng = Rng::new(42);
    let n = 500;
    let n_features = 8;
    let x = Tensor::from_vec(
        (0..n * n_features).map(|i| (i as f64 * 0.01).sin()).collect(),
        vec![n, n_features],
    )?;
    let y = Tensor::from_vec(
        (0..n).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect(),
        vec![n],
    )?;

    // Train/test split
    let (x_train, x_test, y_train, y_test) = train_test_split(&x, &y, 0.2, Some(42));

    // Histogram gradient boosting (LightGBM-style)
    let mut model = HistGradientBoostingClassifier::new()
        .n_estimators(100)
        .learning_rate(0.1)
        .max_depth(5)
        .max_bins(255);
    model.fit(&x_train, &y_train)?;

    // Evaluate
    let preds = model.predict(&x_test)?;
    let acc = accuracy(&y_test.as_slice().to_vec(), &preds.as_slice().to_vec());
    println!("Accuracy: {:.2}%", acc * 100.0);

    // SHAP values for explainability
    let shap_values = tree_shap(&model, &x_test)?;
    println!("SHAP values shape: {:?}", shap_values.shape());

    // Feature importance (mean |SHAP|)
    let importances = permutation_importance(&model, &x_test, &y_test, 10)?;
    println!("Feature importances: {:?}", importances);

    Ok(())
}
```

## 12. Time Series Forecasting (Prophet-style)

Decompose a time series into trend + seasonality and forecast future values.

```rust
use scivex::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Generate daily data with trend + weekly seasonality
    let n = 365;
    let t: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let y: Vec<f64> = t.iter().map(|&ti| {
        // Linear trend + weekly seasonality + noise
        0.05 * ti
            + 3.0 * (2.0 * std::f64::consts::PI * ti / 7.0).sin()
            + (ti * 0.1).sin() * 0.5
    }).collect();

    // Prophet-style forecasting
    let mut prophet = Prophet::new()
        .changepoint_prior_scale(0.05)
        .seasonality_prior_scale(10.0)
        .n_changepoints(25);
    prophet.fit(&t, &y)?;

    // Forecast 30 days ahead
    let future_t: Vec<f64> = (n..n + 30).map(|i| i as f64).collect();
    let forecast = prophet.predict(&future_t)?;
    println!("Forecast (next 7 days): {:?}", &forecast[..7]);

    // Access decomposition
    let trend = prophet.trend(&future_t)?;
    let seasonal = prophet.seasonality(&future_t)?;
    println!("Trend component: {:?}", &trend[..7]);

    // Time series feature extraction
    use scivex_stats::ts_features::{extract_features, TsFeature};
    let features = vec![
        TsFeature::Mean,
        TsFeature::StdDev,
        TsFeature::LinearTrendSlope,
        TsFeature::AutoCorrelation(7), // weekly lag
    ];
    let result = extract_features(&y, 30, 7, &features)?;
    println!("Rolling features: {} windows x {} features",
        result.features.len(), result.feature_names.len());

    Ok(())
}
```

## 13. Bayesian Inference with NUTS

Use the No-U-Turn Sampler (NUTS) for Bayesian parameter estimation.

```rust
use scivex::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Log-posterior for a normal model: y ~ Normal(mu, sigma)
    // Priors: mu ~ Normal(0, 10), sigma ~ HalfCauchy(5)
    let data = vec![2.1, 3.4, 2.8, 3.1, 2.5, 3.8, 2.9, 3.3, 2.7, 3.0];

    let log_posterior = |params: &[f64]| -> f64 {
        let mu = params[0];
        let log_sigma = params[1];
        let sigma = log_sigma.exp();

        // Log-likelihood
        let ll: f64 = data.iter()
            .map(|&y| -0.5 * ((y - mu) / sigma).powi(2) - log_sigma)
            .sum();

        // Log-priors
        let prior_mu = -0.5 * (mu / 10.0).powi(2);    // Normal(0, 10)
        let prior_sigma = -(1.0 + (sigma / 5.0).powi(2)).ln();  // HalfCauchy(5)

        ll + prior_mu + prior_sigma
    };

    let grad_log_posterior = |params: &[f64]| -> Vec<f64> {
        // Numerical gradient
        let eps = 1e-5;
        let mut grad = vec![0.0; params.len()];
        let f0 = log_posterior(params);
        for i in 0..params.len() {
            let mut p = params.to_vec();
            p[i] += eps;
            grad[i] = (log_posterior(&p) - f0) / eps;
        }
        grad
    };

    // Run NUTS sampler
    let initial = vec![0.0, 0.0]; // [mu, log_sigma]
    let samples = nuts(
        log_posterior,
        grad_log_posterior,
        &initial,
        2000,   // n_samples
        500,    // warmup
        0.8,    // target_accept
    )?;

    // Posterior summaries
    let mu_samples: Vec<f64> = samples.iter().map(|s| s[0]).collect();
    let sigma_samples: Vec<f64> = samples.iter().map(|s| s[1].exp()).collect();

    let mu_mean = descriptive::mean(&mu_samples)?;
    let sigma_mean = descriptive::mean(&sigma_samples)?;
    println!("Posterior mean mu: {:.3}", mu_mean);
    println!("Posterior mean sigma: {:.3}", sigma_mean);

    Ok(())
}
```

## 14. CatBoost Regression with Categorical Features

Train a CatBoost-style model on mixed numeric + categorical data.

```rust
use scivex::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Simulate housing data: [sqft, bedrooms, neighborhood_encoded, year_built]
    let n = 300;
    let x = Tensor::from_vec(
        (0..n * 4).map(|i| {
            let row = i / 4;
            let col = i % 4;
            match col {
                0 => 800.0 + (row as f64) * 5.0,       // sqft
                1 => ((row % 4) + 1) as f64,             // bedrooms
                2 => (row % 5) as f64,                    // neighborhood (categorical)
                3 => 1960.0 + (row as f64) * 0.2,        // year_built
                _ => 0.0,
            }
        }).collect(),
        vec![n, 4],
    )?;
    let y = Tensor::from_vec(
        (0..n).map(|i| 100_000.0 + (i as f64) * 500.0 + ((i % 5) as f64) * 10_000.0).collect(),
        vec![n],
    )?;

    // CatBoost with categorical feature indices
    let mut model = CatBoostRegressor::new()
        .iterations(200)
        .learning_rate(0.05)
        .depth(6)
        .cat_features(vec![2]); // column 2 is categorical
    model.fit(&x, &y)?;

    // Feature importance
    let importances = model.feature_importances();
    println!("Feature importances: {:?}", importances);

    // Predict
    let preds = model.predict(&x)?;
    let r2 = r2_score(y.as_slice(), preds.as_slice());
    println!("R2 score: {:.4}", r2);

    Ok(())
}
```

## 15. Data Wrangling: Lazy Eval + Joins + GroupBy

Demonstrate DataFrame lazy evaluation chains and complex transformations.

```rust
use scivex::prelude::*;

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Create orders DataFrame
    let orders = DataFrame::new(vec![
        Series::new("order_id", vec![1i64, 2, 3, 4, 5, 6]),
        Series::new("customer_id", vec![101i64, 102, 101, 103, 102, 101]),
        Series::new("product", vec!["Widget", "Gadget", "Widget", "Doohickey", "Widget", "Gadget"]),
        Series::new("amount", vec![25.50, 45.00, 30.00, 15.75, 45.00, 52.00]),
    ])?;

    // Create customers DataFrame
    let customers = DataFrame::new(vec![
        Series::new("customer_id", vec![101i64, 102, 103]),
        Series::new("name", vec!["Alice", "Bob", "Carol"]),
        Series::new("region", vec!["North", "South", "North"]),
    ])?;

    // Join orders with customers
    let joined = orders.inner_join(&customers, "customer_id", "customer_id")?;
    println!("Joined shape: {:?}", joined.shape());

    // GroupBy: total spend per customer per region
    let summary = joined
        .groupby(&["name", "region"])?
        .agg(&[("amount", "sum"), ("order_id", "count")])?;
    println!("Summary:\n{}", summary);

    // Filter: orders over $30
    let big_orders = orders.filter("amount", |v: &f64| *v > 30.0)?;
    println!("Orders > $30: {} rows", big_orders.shape().0);

    // Sort by amount descending
    let sorted = orders.sort_by("amount", false)?;
    println!("Top order: ${}", sorted.column("amount")?.get::<f64>(0)?);

    // Pivot table style: product x region totals
    let pivot = joined
        .groupby(&["product", "region"])?
        .agg(&[("amount", "sum")])?;
    println!("Pivot:\n{}", pivot);

    Ok(())
}
```
