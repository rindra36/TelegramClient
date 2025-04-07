from telethon import TelegramClient, events
import logging

# Set up logging
logging.basicConfig(
    format='[%(levelname) %(asctime)s] %(name)s: %(message)s',
    level=logging.WARNING
)

# YouseffTrader : -1002074799242
# OTC Pro Trading Robot : -1001851061994
# Notorious Silva Signals 🥇 : -1001621425154
# VIPERION📈POCKET OPTION : -1001748950801
# 🔥🔥👇POCKET OPTION WINNING SIGNALS👇🔥🔥 : -1001674492582

# Remember to use your own values from my.telegram.org!
api_id = 17895806
api_hash = 'ed08e80c65dafa30895f20eee5366aa1'

# Create Telegram Client
client = TelegramClient('anon', api_id, api_hash)

# Main function
async def main():
    # Getting information about myself
    me = await client.get_me()

    # "me" is a user object. You can pretty-print
    # any Telegram object with the "stringify" method:
    print(me.stringify())
