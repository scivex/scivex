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
