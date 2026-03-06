#!/bin/bash
# Add 5-minute options monitor cron to Pi
# Run this on Pi: bash scripts/install_options_cron.sh

CRON_LINE="*/5 9-15 * * 1-5 /bin/bash ~/financial-agent-india/scripts/options_monitor_cron.sh"

# Check if already installed
if crontab -l 2>/dev/null | grep -q "options_monitor_cron"; then
    echo "Options monitor cron already installed."
    echo "Current entry:"
    crontab -l | grep "options_monitor_cron"
else
    (crontab -l 2>/dev/null; echo "$CRON_LINE") | crontab -
    echo "Options monitor cron installed: every 5 min, Mon-Fri 9-15 IST"
fi
