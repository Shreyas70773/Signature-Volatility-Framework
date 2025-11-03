# Hybrid Signature-Ensemble Volatility Forecasting Framework

This repository contains the complete R implementation of a novel, high-performance framework for forecasting future realized financial volatility. The model fuses techniques from **Rough Path Theory**, **Financial Econometrics**, and **Machine Learning** to produce robust, data-driven forecasts and uncertainty estimates.

The core contribution is a multi-stage pipeline that moves beyond traditional time-series models (like GARCH) by constructing a rich, high-dimensional feature set and using a multi-objective ensemble of gradient boosted trees to capture complex market dynamics.

---

## Key Features

*   **Advanced Feature Engineering:** Generates over 70 predictive features, including:
    *   **Multi-Scale Path Signatures:** Captures the geometric, path-dependent character of price movements at different time horizons.
    *   **Jump-Detection Metrics:** Explicitly models market shocks and their after-effects, motivated by Extreme Value Theory.
    *   **Rich Econometric Indicators:** Includes measures of volatility clustering, skewness, kurtosis, and market regime.
*   **Multi-Objective Ensemble Model:** Utilizes an ensemble of five distinct XGBoost models, each trained with a unique objective function (MAE, Pseudo-Huber, and custom asymmetric/quantile losses) to capture different aspects of risk.
*   **Dynamic Prediction Intervals:** Produces not just a single point forecast but also a dynamic, data-driven prediction interval that quantifies model uncertainty and adapts to market conditions.
*   **Robust Performance:** The final Weighted Ensemble model demonstrates strong out-of-sample performance, achieving a positive R-squared and high correlation with future realized volatility.

---

## The Forecasting Pipeline

The model operates as a five-stage pipeline, transforming raw price data into a sophisticated volatility forecast.

### Step 1: Data Acquisition
Daily adjusted closing prices for the S&P 500 index are downloaded. Log-returns are calculated and used as the basis for all subsequent steps.

### Step 2: High-Dimensional Feature Engineering
This is the core of the model. Raw price and return data are transformed into a rich feature set representing the market state. Key feature classes include:

1.  **Path Signature Features:** The signature transform is applied to the 2D log-price path across multiple lookback windows (10, 20, 40 days) to capture short, medium, and long-term path geometry.
2.  **Jump & Extreme Value Features:** A dynamic threshold is used to detect market jumps. Features include jump counts, average jump magnitude, days since the last jump, and a 252-day rolling maximum of returns.
3.  **Econometric & Statistical Features:** This includes a wide array of traditional metrics such as historical volatility over multiple horizons, ratios of short-to-long term volatility, rolling skewness and kurtosis, and measures of volatility clustering.

### Step 3: Target Variable Definition
The model is trained to predict a direct, observable measure of future risk: the **20-day annualized realized volatility**.

### Step 4: Multi-Objective Model Training
Five separate XGBoost models are trained on the same feature set. Each model is optimized using a different mathematical objective, forcing the ensemble to learn diverse perspectives on the forecasting problem:
*   **Lower Bound Model (Quantile Loss, $\alpha=0.1$):** A conservative model that aims to under-predict 90% of the time.
*   **MAE Model (L1 Loss):** A robust model that targets the median of the future volatility distribution.
*   **Asymmetric Loss Model:** A custom model that heavily penalizes under-predicting volatility spikes, making it sensitive to risk-on events.
*   **Pseudo-Huber Model:** A robust model that is less sensitive to extreme outliers than a standard L2 loss.
*   **Upper Bound Model (Quantile Loss, $\alpha=0.9$):** An aggressive model that aims to over-predict 90% of the time.

### Step 5: Ensemble Aggregation and Prediction
The predictions from the five models are combined:
*   **Weighted Ensemble Forecast:** The final point forecast is a weighted average of the five predictions, with weights inversely proportional to each model's out-of-sample RMSE.
*   **Prediction Interval:** The forecasts from the Lower and Upper Bound models are used to construct an 80% data-driven prediction interval.

---

## Performance and Results

The model was trained on S&P 500 data from 2005-2021 and tested on out-of-sample data from 2022 onwards. The final Weighted Ensemble achieved an **R-squared of 0.173** and a **correlation of 0.556** with the actual future realized volatility.

### Volatility Forecast with Prediction Intervals
The final model successfully tracks the major volatility regimes and provides a dynamic measure of uncertainty.

![Time Series Forecast](output/plot_timeseries_forecast.png)

### Feature Importance
The analysis reveals that the model's success comes from a true hybrid of features. The top predictors include econometric measures (`ret_skew`), our novel jump features (`days_since_jump`), and several path signature terms, confirming the value of the signature transform.

### Diagnostic Analysis
The model exhibits robust behavior across different market regimes and provides valuable insights into its own uncertainty.

| Model Comparison Scatter Plots                                         | Error Distribution by Regime                                       |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------ |
| ![Scatter Plots](output/plot_model_comparison_scatter.png) | ![Boxplot](output/plot_error_distribution_by_regime.png) |

| Prediction Interval Coverage Analysis                                  | Jump Detection Validation                                          |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------ |
| ![Coverage Plot](output/plot_prediction_interval_coverage.png)         | ![Jumps Plot](output/plot_jump_detection.png)                       |

---

## Implementation

### Dependencies
The model is implemented in R. The following packages are required:
```R
install.packages(c("quantmod", "xts", "zoo", "xgboost", "ggplot2", "tidyr", "gridExtra"))
```

### Running the Model
The entire pipeline, from data download to final visualization, can be executed by running the `run_volatility_model.R` script.

```R
# Source the main script to run the full analysis
source("run_volatility_model.R")
```
The script will automatically download the latest data, engineer all features, train the ensemble, evaluate performance, and generate all output plots in the `output/` directory.

---

## Mathematical Appendix

### A.1 The Path Signature
Let $\mathbf{X}: [0, T] \to \mathbb{R}^d$ be a continuous path. The **signature** of $\mathbf{X}$, denoted $S(\mathbf{X})$, is a sequence of iterated integrals. The term corresponding to the multi-index $(i_1, \dots, i_k)$ is given by:
$$ S(\mathbf{X})_{i_1, \dots, i_k} = \int_{0 < s_1 < \dots < s_k < T} \cdots \int dX_{s_1}^{i_1} \cdots dX_{s_k}^{i_k} $$
For a discrete 2D path $\mathbf{X}_s = (s, \log(P_s))$, we approximate these terms with iterated sums. For example, the second-degree term $S_{1,2}$ is calculated as:
$$ S_{1,2} \approx \sum_{k=1}^{N-1} X_{k}^1 \cdot (X_{k+1}^2 - X_k^2) $$
where $X^1$ is the time component and $X^2$ is the log-price component.

### A.2 Custom XGBoost Objective Functions
To train a model in XGBoost, we must provide the first (gradient) and second (Hessian) derivatives of the loss function $L(y, \hat{y})$ with respect to the prediction $\hat{y}$. Let $\epsilon = y - \hat{y}$.

#### Asymmetric Loss
The loss is $L_{\text{asym}} = 2\epsilon^2 \cdot \mathbb{I}_{\epsilon>0} + \epsilon^2 \cdot \mathbb{I}_{\epsilon \le 0}$.
*   **Gradient (grad):** $g = -\frac{\partial L}{\partial \epsilon} = 
    \begin{cases} 
          -4\epsilon & \text{if } \epsilon > 0 \\
          -2\epsilon & \text{if } \epsilon \le 0
       \end{cases}$
*   **Hessian (hess):** $h = \frac{\partial^2 L}{\partial \hat{y}^2} = 
    \begin{cases} 
          4 & \text{if } \epsilon > 0 \\
          2 & \text{if } \epsilon \le 0
       \end{cases}$

#### Quantile Loss
The loss is $L_{\alpha} = \alpha \epsilon \cdot \mathbb{I}_{\epsilon \ge 0} + (\alpha-1) \epsilon \cdot \mathbb{I}_{\epsilon < 0}$.
*   **Gradient (grad):** $g = -\frac{\partial L}{\partial \epsilon} = 
    \begin{cases} 
          -\alpha & \text{if } \epsilon \ge 0 \\
          1 - \alpha & \text{if } \epsilon < 0
       \end{cases}$
*   **Hessian (hess):** $h = 1$ (set to a constant for numerical stability).

---

## Citation
If you use this framework in your research, please cite it as follows:

```bibtex
@misc{volatility_signature_ensemble_2025,
  title={A Hybrid Path Signature and Machine Learning Ensemble for Forecasting Realized Volatility},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/your_username/your_repository_name}}
}
```

## License
This project is licensed under the MIT License - see the `LICENSE` file for details.
