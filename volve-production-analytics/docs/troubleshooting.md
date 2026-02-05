# Troubleshooting Guide

Common issues and their solutions for the Volve Production Analytics pipeline and dashboard.

## No Data Found

**Symptom**: Dashboard shows "No data found" or pipeline fails at Step 1.

**Causes and fixes**:
- Verify raw data file exists in `data/raw/` (expected: CSV with production columns)
- Check that column names match one of the patterns in `src/config.py` COLUMN_MAPPING
- If using processed data, ensure `data/processed/volve_monthly.parquet` exists (run the pipeline first)
- File encoding issues: the pipeline expects UTF-8 encoded CSV files

## SharePoint Authentication Errors

**Symptom**: "Token acquisition failed" or 401 Unauthorized from Microsoft Graph API.

**Causes and fixes**:
- Verify `.env` contains valid AZURE_TENANT_ID, AZURE_CLIENT_ID, and AZURE_CLIENT_SECRET
- Check that the Azure AD app registration has Sites.ReadWrite.All permission granted with admin consent
- Client secrets expire -- check the expiry date in Azure Portal under App Registrations > Certificates & secrets
- Token refresh is automatic (5-minute buffer before expiry), but a fully expired secret requires rotation in Azure Portal

## Token Refresh Issues

**Symptom**: SharePoint calls fail intermittently after working initially.

**Causes and fixes**:
- The SharePointClient caches tokens and refreshes 5 minutes before expiry
- If the system clock is significantly off, token expiry checks may behave unexpectedly
- Network interruptions during token refresh will raise RuntimeError -- the dashboard falls back to local mode automatically

## SharePoint I/O Failures

**Symptom**: "Network unreachable" or timeout errors during file sync.

**Causes and fixes**:
- Verify SHAREPOINT_SITE_URL is correct (format: https://yourtenant.sharepoint.com/sites/YourSite)
- Check network connectivity to Microsoft 365 endpoints
- The pipeline and dashboard both have local fallback -- if SharePoint is unavailable, operations continue using `data/` directory
- For persistent issues, check Azure AD > Enterprise Applications > sign-in logs for error details

## Forecast Returns All Zeros

**Symptom**: All forecast values are 0 Sm3.

**Causes and fixes**:
- The forecast floor clips negative predictions to 0. If the model predicts negative values (common for declining fields), all forecasts may be zero
- Check that the training data has at least 12 months of non-zero production (MIN_HISTORY_MONTHS)
- Try the Seasonal Naive baseline model as a sanity check -- if it also returns zeros, the underlying data may have insufficient signal
- Verify the selected wellbore has recent production (some wells ceased production before 2016)

## MAPE Shows N/A

**Symptom**: MAPE metric displays "N/A" in the Model Performance section.

**Causes and fixes**:
- MAPE is undefined when actual values are zero. If the backtest period includes zero-production months, those are excluded from MAPE calculation
- If all backtest actuals are zero, MAPE cannot be computed and shows N/A
- WAPE is always available as an alternative -- it handles zero values by aggregating across all observations
- Switch to Total Field mode for more robust metrics (aggregation reduces zero-value frequency)

## Adjusting Anomaly Sensitivity

**Symptom**: Too many or too few anomalies flagged.

**Causes and fixes**:
- Use the threshold slider in the sidebar (range: 1.0 to 4.0, default 2.5)
- Lower threshold (e.g., 2.0) flags more points -- useful for conservative monitoring
- Higher threshold (e.g., 3.0) flags only extreme outliers -- reduces noise
- The Threshold Sensitivity table in the Anomaly Detection section shows flag counts at different thresholds to help calibrate
- Statistical anomalies are not confirmed failures -- use them as triage signals, not definitive diagnoses

## Power Automate Flow Errors

**Symptom**: Weekly email report not delivered, or flow shows failed status.

**Causes and fixes**:
- Check the flow run history in Power Automate portal for specific error messages
- Verify the SharePoint file paths in the flow match the actual upload locations
- Ensure the service account has Send.Mail permission if sending via Outlook
- The flow uses scope-based error handling -- check the "Catch" scope for logged error details
- Admin notification action should fire on failure -- verify the admin email address is correct
