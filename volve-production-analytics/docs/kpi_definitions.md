# KPI Definitions

Reference guide for all key performance indicators used in the Volve Production Analytics dashboard.

## Oil, Gas, and Water Production

Production volumes are measured in **Sm3** (standard cubic meters) -- volume normalized to standard temperature and pressure conditions. Monthly totals are aggregated across all active wellbores in Total Field mode, or shown per-wellbore in Single Wellbore mode.

## Month-over-Month (MoM) Change

MoM change measures the percentage difference between the current month's production and the prior month's production:

    MoM% = ((current_month - prior_month) / prior_month) * 100

A positive MoM indicates production growth; negative indicates decline. MoM is sensitive to short-term events (shutdowns, workovers) and should be interpreted alongside YoY for trend context.

## Year-over-Year (YoY) Change

YoY change compares the current month to the same month one year ago:

    YoY% = ((current_month - same_month_last_year) / same_month_last_year) * 100

YoY removes seasonal effects and reveals long-term production trajectory. A sustained negative YoY trend may indicate reservoir depletion.

## MAPE (Mean Absolute Percentage Error)

    MAPE = mean(|actual - predicted| / |actual|) * 100

MAPE expresses forecast error as a percentage of actual values. It is intuitive but has a known limitation: it is undefined when actual values are zero. The dashboard excludes zero-production months from MAPE calculation. For datasets with many near-zero values, prefer WAPE.

## WAPE (Weighted Absolute Percentage Error)

    WAPE = sum(|actual - predicted|) / sum(|actual|) * 100

WAPE aggregates errors relative to total production volume. It is more robust than MAPE because it handles near-zero values gracefully and weights errors by magnitude. Lower WAPE indicates better overall forecast accuracy.

## MAE (Mean Absolute Error)

    MAE = mean(|actual - predicted|)

MAE is expressed in the same units as the target variable (Sm3). It gives an average error magnitude without percentage normalization. Useful for understanding absolute forecast deviation in production volume terms.

## RMSE (Root Mean Squared Error)

    RMSE = sqrt(mean((actual - predicted)^2))

RMSE penalizes large errors more heavily than MAE due to the squaring operation. If RMSE is much larger than MAE, it indicates the model has occasional large misses. Also expressed in Sm3.

## Z-Score Anomaly Detection

The dashboard uses a rolling z-score method to flag production anomalies:

    z-score = (value - rolling_mean) / rolling_std

Where `rolling_mean` and `rolling_std` are computed over a 6-month window (minimum 3 observations). Points with |z-score| exceeding the user-selected threshold (default 2.5) are flagged as anomalies.

A z-score > 0 means production is above the rolling average; z-score < 0 means below average. Statistical anomalies are not confirmed operational failures -- they serve as triage signals.

## ETS Confidence Intervals

The Exponential Smoothing (ETS) model produces 95% confidence intervals based on the standard deviation of model residuals:

    upper = forecast + 1.96 * residual_std
    lower = forecast - 1.96 * residual_std

Wider intervals indicate greater model uncertainty. All forecast values and bounds are clipped to zero (production floor -- negative production is physically impossible).
