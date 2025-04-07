import logging, re, json, asyncio, pytz

from telethon import TelegramClient, events
from datetime import datetime
from zoneinfo import ZoneInfo

from telethon.tl.types import InputBotInlineMessageText

from PocketOptionAPI import PocketOptionAPI

def get_wait_time(entry_time_str, timezone_offset=-4):
    """
    Convert entry time from UTC-3 or UTC-4 to local time and calculate seconds to wait

    Args:
        entry_time_str: Time string in format "HH:MM"
        timezone_offset: Timezone offset (-3 or -4)
    Returns:
        Seconds to wait, or None if the time has passed for today
    """
    if timezone_offset not in [-3, -4]:
        raise ValueError("Timezone offset must be -3 or -4")

    # Get current date
    now = datetime.now()
    today = now.date()

    # Parse entry time
    hour, minute = map(int, entry_time_str.split(':'))

    # Select correct timezone based on offset
    tz_map = {
        -4: 'America/New_York',  # UTC-4
        -3: 'America/Sao_Paulo'  # UTC-3
    }

    # Create datetime in source timezone
    tz_entry = pytz.timezone(tz_map[timezone_offset])
    entry_time = datetime.combine(today, datetime.min.time().replace(hour=hour, minute=minute))
    entry_time = tz_entry.localize(entry_time)

    # Get current time in local timezone
    now_local = datetime.now(tz=datetime.now().astimezone().tzinfo)

    # Convert entry time to local timezone
    local_entry = entry_time.astimezone()

    # Calculate wait time
    wait_seconds = (local_entry - now_local).total_seconds()

    # If time has passed today, return None
    if wait_seconds < 0:
        return None

    return wait_seconds

async def waiting_time(entry_time, timezone_offset=-4):
    try:
        wait_time = get_wait_time(entry_time, timezone_offset)
        if wait_time is None:
            print("Entry time has already passed")
            return False

        print(f"Waiting {wait_time} seconds...")
        await asyncio.sleep(wait_time - 4) # The time to let the API waiting

        return True
    except ValueError as e:
        print(f"Error: {e}")
        return False

async def print_message(id, limit = 100):
    # You can print the message history of any chat:
    async for message in client.iter_messages(id, limit=limit):
        print(message.id, message.text, message.date)

# Set up logging
logging.basicConfig(
    format='[%(levelname)s %(asctime)s] %(name)s: %(message)s',
    level=logging.WARNING
)

# Get access
with open('../assets/telegramCredentials.json', 'r', encoding='utf-8') as f:
    jsonAccess = json.loads(f.read())
    api_id = jsonAccess['id']
    api_hash = jsonAccess['hash']


if api_id != '' and api_hash != '':
    with open('../assets/chats.json', 'r', encoding='utf-8') as f:
        chatsList = json.loads(f.read())

    client = TelegramClient('anon', api_id, api_hash)
    pocketOption = PocketOptionAPI()

    async def main():
        await print_message(-1001674492582, 1000)


    # Handle for Notorious Silva Signals 🥇
    # @client.on(events.NewMessage(chats=chatsList['NSilva']))
    # async def handler(event):
    #     action = entry = None
    #     channel = 'NSilva'
    #
    #     message = event.raw_text
    #     pattern = r'Martingale levels'
    #     matchpattern = re.search(pattern, message, re.IGNORECASE)
    #
    #     if matchpattern:
    #         # Get the action (CALL/PUT)
    #         regexaction = r'(CALL|PUT)'
    #         matchaction = re.search(regexaction, message, re.IGNORECASE)
    #
    #         if matchaction:
    #             action = f'{matchaction.group(1)}'
    #         else:
    #             print(
    #                 f'message : {message}'
    #                 'No action found'
    #             )
    #             return False
    #
    #         # Get the entry point (H:M)
    #         regexentry = r'(\d{1,2}:\d{2})'
    #         matchentry = re.search(regexentry, message)
    #
    #         if matchentry:
    #             entry = f'{matchentry.group(1)}'
    #         else:
    #             print(
    #                 f'message : {message}'
    #                 'No entry found'
    #             )
    #             return False

    # Handle generally for TEST : TO DELETE
    # @client.on(events.NewMessage)
    # async def handler(event):
    #     message = event.raw_text
    #     regex = r'(?:Preparing trading asset|Summary:|Result:)'
    #     print(
    #         'New message General',
    #         f'message: {message}',
    #         re.search(regex, message, re.IGNORECASE)
    #     )

    # client.start()
    # client.run_until_disconnected()
    with client:
        client.loop.run_until_complete(main())
else:
    print('Empty credentials')
