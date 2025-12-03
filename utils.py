# Utility helpers for the nowcasting pipeline. Includes time formatting for the output
# 'Future Date' (New York local time rounded to hour, +1 hour), and other small helpers.
from datetime import datetime, timedelta, timezone

try:
    from zoneinfo import ZoneInfo  # stdlib (Py3.9+); on Windows: pip install tzdata
except Exception:
    ZoneInfo = None

# This function: Format future timestamps (local NY time, rounded to hour, +1 hour).
def get_current_ny_date():

    if ZoneInfo is not None: # Conditional branch to handle specific case(s)
        tz = ZoneInfo("America/New_York")
        future_dt = datetime.now(tz).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

        return future_dt.strftime("%Y-%m-%d %H:%M %Z")  # For example. 2025-11-30 18:00 EST  # Return
    else:
        # fallback to UTC, if tzdata not available
        future_dt = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        return future_dt.strftime("%Y-%m-%d %H:%M UTC")
