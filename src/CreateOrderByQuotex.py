# pip install git+https://github.com/cleitonleonel/pyquotex.git

from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync
from BinaryOptionsToolsV2.validator import Validator
from PocketOptionMethod import PocketOptionMethod
from quotexapi.stable_api import Quotex

import asyncio
import json

async def connect_with_retries(client: Quotex, attempts=5):
    check_connect, message = await client.connect()
    
    if not check_connect:
        attempt = 0
        while attempt <= attempts:
            if not await client.check_connect():
                check_connect, message = await client.connect()
                if check_connect:
                    print("Reconnection successful!")
                    break
                else:
                    attempt += 1
                    print(f"Retrying connection {attempt} of {attempts}")
            await asyncio.sleep(5)
    
    return check_connect, message

async def main():
    # # Initialize the copying API client
    # ssid_copy = input('Please enter the SSID for copying: ')
    # api = PocketOptionAsync(ssid_copy)
    # await asyncio.sleep(5)  # Wait for connection to establish

    # Initialize the Quotex API client
    client = Quotex(
        email='laryseulement@gmail.com',
        password='Motdepasse!01',
        lang='en'
    )

    # Initialize the application API client
    account_id = input('Choose the account ID [1, 2]: ')
    wallet_type = input('Choose the wallet type [demo, real]: ')

    if account_id not in ['1', '2']:
        raise ValueError("Invalid account ID. Choose either '1' or '2'.")

    if wallet_type not in ['demo', 'real']:
        raise ValueError("Invalid wallet type. Choose either 'demo' or 'real'.")
    
    api_app = PocketOptionMethod(int(account_id), wallet_type)
    await asyncio.sleep(5)  # Wait for connection to establish

    try:
        # Connection with retries
        while True:            
            check_connect, message = await connect_with_retries(client)
            
            if check_connect:
                print("Beginning raw order ...")
                response = await client.start_realtime_order('EURUSD')
                print(f"Basic raw order response: {response}")
                
                asset = response['asset']
                command = response['command']
                print(f'Original expiration time: {response["closeTimestamp"] - response["openTimestamp"]} seconds')
                time = max(response['closeTimestamp'] - response['openTimestamp'], 60)
                amount = 1
                
                # Create the trade depending of the command
                if command == 0:
                    # Buy order
                    print(f"Buying {asset} for {amount} at {time} seconds")
                    trade_id = await api_app.buy(asset, amount, time, check_win=False)
                    print(f"Trade placed ID: {trade_id}")
                elif command == 1:
                    # Sell order
                    print(f"Selling {asset} for {amount} at {time} seconds")
                    trade_id = await api_app.sell(asset, amount, time, check_win=False)
                    print(f"Trade placed ID: {trade_id}")
                else:
                    print("Unknown command")
            
    except Exception as e:
        print(f"Error: {e}")

    # Send basic raw order and get response if containning "requestId" and then create order for the application API
    # while True:
    #     try:
    #         print("Beginning raw order ...")
    #         validator = Validator.contains('requestId')
    #         response = await api.create_raw_order(
    #             '42["ping"]',
    #             validator
    #         )
    #         print(f"Basic raw order response: {response}")

    #         # Parse the response
    #         # Response example: {"id":"3b53adbb-8a89-4fd6-b65a-b25811151463","openTime":"2025-04-20 21:06:20","closeTime":"2025-04-20 21:07:10","openTimestamp":1745183180,"closeTimestamp":1745183230,"uid":96282099,"isDemo":1,"amount":20,"profit":18.4,"percentProfit":92,"percentLoss":100,"openPrice":15699.5,"copyTicket":"","closePrice":0,"command":0,"asset":"USDIDR_otc","requestId":"buy","openMs":273,"optionType":100,"isCopySignal":false,"currency":"USD"}
    #         # Getting the asset, command
    #         # Transform the response to a dictionary
    #         response_json = json.loads(response)
    #         asset = response_json['asset']
    #         command = response_json['command']
    #         time = 50
    #         amount = 1

    #         # Create the trade depending of the command
    #         if command == 0:
    #             # Buy order
    #             print(f"Buying {asset} for {amount} at {time} seconds")
    #             trade_id = await api_app.buy(asset, amount, time, check_win=False)
    #             print(f"Trade placed ID: {trade_id}")
    #         elif command == 1:
    #             # Sell order
    #             print(f"Selling {asset} for {amount} at {time} seconds")
    #             trade_id = await api_app.sell(asset, amount, time, check_win=False)
    #             print(f"Trade placed ID: {trade_id}")
    #         else:
    #             print("Unknown command")

    #         # Ensure balance is not below $5
    #         balance = await api_app.get_balance()
    #         if balance < 5:
    #             print(f'Reached the stop balance of $5. Stopping the bot.')
    #             print(f'Current balance: {balance}')
    #             break
    #     except Exception as e:
    #         print(f"Basic raw order failed: {e}")

if __name__ == '__main__':
    asyncio.run(main())