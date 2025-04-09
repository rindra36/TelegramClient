from telethon import events
from src.utils import load_credentials, setup_client, load_chats

SESSION_NAME = 'AureaBot'
credentials = load_credentials()
client = setup_client('anon', credentials['id'], credentials['hash'])
chats = load_chats

@client.on(events.NewMessage(chats=chats[SESSION_NAME]))
async def handle_new_message(event):
    """Handle new messages in the specified chat."""
    message = event.raw_text
    
    if 'Choose a new trading pair' in message:
        await event.respond('EURCHF_otc')
        
    if 'Choose an expiry time' in message:
        await event.respond('1M')
            
    print(f"New message in {SESSION_NAME}: {message}")
    
client.start()
client.run_until_disconnected()