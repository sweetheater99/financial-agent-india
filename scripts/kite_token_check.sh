#!/bin/bash
# Check Kite access token validity and alert via Telegram if expired.
# Crontab: 50 3 * * 1-5 (8:50 AM IST = 3:20 UTC)
cd ~/financial-agent-india
source venv/bin/activate
source ~/.config/env/global.env

python3 -c "
from kite_data import get_kite
try:
    kite = get_kite()
    print('Kite token: OK')
except Exception as e:
    import os, requests
    token = os.getenv('DEAL_BOT_TOKEN') or os.getenv('TELEGRAM_BOT_TOKEN')
    chat = os.getenv('TELEGRAM_FORUM_CHAT_ID') or os.getenv('DEAL_BOT_CHAT_ID')
    topic = os.getenv('TELEGRAM_TOPIC_STOCKS', '')
    msg = '⚠️ Kite access token expired! Run: python kite_auth.py on Pi'
    data = {'chat_id': chat, 'text': msg, 'parse_mode': 'HTML'}
    if topic:
        data['message_thread_id'] = topic
    requests.post(f'https://api.telegram.org/bot{token}/sendMessage', data=data)
    print(f'Kite token: EXPIRED - {e}')
"
