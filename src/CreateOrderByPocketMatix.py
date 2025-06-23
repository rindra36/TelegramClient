# pip install pytest-playwright
# .venv/Scripts/playwright.exe install # Install with Playwright in the virtual environment

from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync
from BinaryOptionsToolsV2.validator import Validator
from PocketOptionMethod import PocketOptionMethod
from playwright.async_api import async_playwright, Playwright, WebSocket, WebSocketRoute

import asyncio
import json
import traceback

api_app = None  # Placeholder for the PocketOptionMethod instance

# Hanlder for WebSocket messages
async def handle_message(message: str | bytes):    
    global api_app
    
    # If the message is a byte string, decode it to a regular string
    if isinstance(message, bytes):
        message = message.decode('utf-8')
        
    # Check if the message contains a requestId
    if '"requestId"' in message:
        print(f"Message with requestId: {message}")
        
        # Transform the response to a dictionary
        response = json.loads(message)
        asset = response['asset']
        command = response['command']
        time = response['closeTimestamp'] - response['openTimestamp']
        amount = response['amount']

        try:
            # Create the trade depending of the command
            if command == 0:
                # Buy order
                try:
                    print(f"Buying {asset} for {amount}$ at {time} seconds")
                    trade_id = await api_app.buy(asset, amount, time, check_win=False)
                    print(f"Trade placed ID: {trade_id}")
                except Exception as e:
                    print(f"Buy order failed: {e}")

            elif command == 1:
                # Sell order
                try:
                    print(f"Selling {asset} for {amount}$ at {time} seconds")
                    trade_id = await api_app.sell(asset, amount, time, check_win=False)
                    print(f"Trade placed ID: {trade_id}")
                except Exception as e:
                    print(f"Sell order failed: {e}")
            else:
                print("Unknown command")
        except Exception as e:
            traceback_handler = traceback.format_exc()
            print(f"Trade error: {e}", traceback_handler)

async def handle_websocket(websocket: WebSocket):
    print(f'WebSocket connected: {websocket.url}')
    
    try:
        while True:
            # Listen for WebSocket messages
            await websocket.on('framereceived', handle_message)
                
    except Exception as e:
        traceback_handler = traceback.format_exc()
        print(f"WebSocket error: {e}", traceback_handler)

async def run(playwright: Playwright):
    browser = await playwright.chromium.launch(headless=False)
    context = await browser.new_context()
    
    # Enable WebSocket tracking
    await context.route("**/*", lambda route: route.continue_())
    
    # Open a new page
    page = await context.new_page()
    
    # Listen for WebSocket connections
    page.on("websocket", handle_websocket)
    
    # Navigate to the PocketOption website
    await page.goto("https://pocketoption.com/en")
    
    # Keep the browser open
    await asyncio.sleep(float('inf'))


async def main():
    global api_app

    # Initialize the application API client
    account_id = input('Choose the account ID [1, 2]: ')
    wallet_type = input('Choose the wallet type [demo, real]: ')

    if account_id not in ['1', '2']:
        raise ValueError("Invalid account ID. Choose either '1' or '2'.")

    if wallet_type not in ['demo', 'real']:
        raise ValueError("Invalid wallet type. Choose either 'demo' or 'real'.")
    
    api_app = PocketOptionMethod(int(account_id), wallet_type)
    await asyncio.sleep(5)  # Wait for connection to establish
    
    async with async_playwright() as playwright:
        await run(playwright)

if __name__ == '__main__':
    asyncio.run(main())