from src.utils import load_credentials, setup_client

creadentials = load_credentials()
client = setup_client('anon', creadentials['id'], creadentials['hash'])

async def main():
    # Get all dialogs
    # async for dialog in client.iter_dialogs():
    #     print(f'{dialog.id}:"{dialog.name}"')

    # Get messages from a specific chat
    id = -1001748950801  # Replace with the chat ID you want to check
    limit = 100  # Number of messages to retrieve
    # You can print the message history of any chat:
    async for message in client.iter_messages(id, limit=limit):
        print(message.id, message.text, message.date)


with client:
    client.loop.run_until_complete(main())