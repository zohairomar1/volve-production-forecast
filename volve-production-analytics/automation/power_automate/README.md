# Power Automate Integration

This guide explains how to set up automated reporting using Microsoft Power Automate with SharePoint and email.

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   SharePoint    │────▶│  Python Pipeline │────▶│   SharePoint    │
│   (Raw Data)    │     │   (Scheduled)    │     │  (Processed)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
                        ┌──────────────────┐     ┌─────────────────┐
                        │      Email       │◀────│  Power Automate │
                        │   (Recipients)   │     │     (Flow)      │
                        └──────────────────┘     └─────────────────┘
```

## Implementation Pattern A: Scheduled Python + Power Automate Email

This is the recommended approach for most organizations.

### Step 1: Schedule Python Pipeline

**Option A: Azure VM / On-Premises Server**
```bash
# crontab entry for weekly execution (Monday 8:00 AM)
0 8 * * 1 /path/to/venv/bin/python -m src.scripts.run_pipeline >> /var/log/volve_pipeline.log 2>&1
```

**Option B: GitHub Actions**
```yaml
# .github/workflows/pipeline.yml
on:
  schedule:
    - cron: '0 8 * * 1'  # Every Monday at 8 AM UTC
```

### Step 2: Create Power Automate Flow

#### Flow Name: "Volve Weekly Production Report"

**Trigger: Recurrence**
- Frequency: Week
- Interval: 1
- On these days: Monday
- At these hours: 9 (after pipeline completes)

#### Actions:

1. **Get file metadata (SharePoint)**
   - Site Address: `https://yourcompany.sharepoint.com/sites/VolveAnalytics`
   - Library Name: `Shared Documents`
   - Folder Path: `/Processed Data`

2. **Condition: Check if email_summary.txt exists**
   - If yes: Continue
   - If no: Send error notification

3. **Get file content (SharePoint)**
   - File Identifier: email_summary.txt

4. **Get file content (SharePoint)**
   - File Identifier: volve_monthly.csv (for attachment)

5. **Send an email (V2) - Outlook**
   - To: `production-team@yourcompany.com`
   - Subject: `Volve Production Report - @{formatDateTime(utcNow(), 'MMMM yyyy')}`
   - Body: See email template below
   - Attachments: volve_monthly.csv

#### Error Handling:

Add a parallel branch for error handling:

1. **Scope: Try processing**
   - Contains all the steps above

2. **Scope: Catch errors** (Configure to run if Try fails)
   - Send email notification to admin
   - Log error to SharePoint list

### Step 3: Configure SharePoint Structure

```
SharePoint Site: VolveAnalytics
├── Shared Documents/
│   ├── Raw Data/
│   │   └── Volve production data.csv
│   ├── Processed Data/
│   │   ├── volve_monthly.csv
│   │   ├── forecasts.csv
│   │   ├── metrics.json
│   │   └── email_summary.txt
│   └── Reports/
│       └── (archived reports)
```

## Implementation Pattern B: HTTP Endpoint (Advanced)

For real-time processing triggered by Power Automate.

### Step 1: Deploy Python as Azure Function

```python
# function_app.py
import azure.functions as func
from src.scripts.run_pipeline import run_pipeline

def main(req: func.HttpRequest) -> func.HttpResponse:
    results = run_pipeline(verbose=False)
    return func.HttpResponse(
        json.dumps(results),
        mimetype="application/json"
    )
```

### Step 2: Power Automate Flow

1. **Trigger: Recurrence** (weekly)
2. **HTTP Action**: Call Azure Function endpoint
3. **Parse JSON**: Extract summary from response
4. **Send Email**: Use parsed data in email body

## Flow Screenshots

See the `screenshots/` folder for visual guides:
- `01_trigger_config.png` - Recurrence trigger setup
- `02_sharepoint_connector.png` - SharePoint file operations
- `03_email_action.png` - Email composition
- `04_error_handling.png` - Error branch configuration

(Note: Screenshots are placeholders - replace with actual captures from your environment)

## Testing the Flow

1. **Manual Trigger**
   - Click "Test" in Power Automate
   - Select "Manually"
   - Verify email arrives with correct content

2. **Verify Data Flow**
   - Check SharePoint for updated processed files
   - Confirm email contains latest data
   - Verify attachment opens correctly

## Troubleshooting

| Issue | Solution |
|-------|----------|
| File not found | Check SharePoint permissions and paths |
| Email not sending | Verify Outlook connector authorization |
| Stale data | Ensure Python pipeline ran successfully |
| Formatting issues | Check email template HTML encoding |

## Security Considerations

1. **SharePoint Permissions**
   - Service account should have read access to Raw Data
   - Service account should have write access to Processed Data

2. **Connection Security**
   - Use managed identities where possible
   - Store secrets in Azure Key Vault
   - Review connector permissions regularly

3. **Data Classification**
   - Production data may be confidential
   - Ensure email recipients are authorized
   - Consider DLP policies on SharePoint

## Cost Estimation

Power Automate licenses:
- Per-flow plan: ~$15/month for single flow
- Per-user plan: Included with Microsoft 365

Storage:
- SharePoint: Included with Microsoft 365
- Additional costs minimal for this data volume
