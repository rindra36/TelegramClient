import asyncio
from lib.PocketOptionAPI.pocketoptionapi_async.client import AsyncPocketOptionClient
# from lib.PocketOptionAPI.pocketoptionapi_async.models import OrderDirection, TimeFrame
from lib.PocketOptionAPI.pocketoptionapi_async.models import OrderDirection
from datetime import datetime
import os
import pandas as pd

async def main():
    # Complete SSID format (get from browser dev tools)
    ssid = r'42["auth",{"session":"j079fsgog45pjnbsj9a2hvpnnb","isDemo":1,"uid":102766033,"platform":3,"isFastHistory":true}]'
    
    # Create client with persistent connection
    client = AsyncPocketOptionClient(
        ssid, 
        is_demo=True,
        persistent_connection=True,  # Enable keep-alive
        auto_reconnect=True         # Enable auto-reconnection
    )
    
    try:
        # Connect
        await client.connect()
        
        # Get balance
        # balance = await client.get_balance()
        # print(f"Balance: ${balance.balance}")
        
        # Get candles
        # candles = await client.get_candles("EURUSD", '1m', 100)
        # print(f"Retrieved {len(candles)} candles")
        
        # Place order (demo)
        # order = await client.place_order("EURUSD", 10, OrderDirection.CALL, 60)
        # print(f"Order placed: {order.order_id}")

        # Create output directory if it doesn't exist
        output_dir = "candles_analysis"
        os.makedirs(output_dir, exist_ok=True)

        # Create output file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"candles_v0_analysis_{timestamp}.txt")
        
        # Test different time frames and periods
        times = [3600 * i for i in range(1, 25)]  # 1-24 hours back
        time_frames = [1, 5, 15, 30, 60, 300]  # Different timeframes in seconds
        times = [86400]  # 1-24 hours back
        time_frames = [1, 5, 15, 30, 60, 120, 180, 300, 600, 900, 1800, 3600, 14400, 86400]  # Different timeframes in seconds
        
        with open(output_file, 'w') as f:
            for time in times:
                for frame in time_frames:
                    output = f"\nBegin datetime : {datetime.now()}\n"

                    # Get candles data
                    candles = await client.get_candles("EURUSD_otc", frame, time)

                    output += f"End datetime : {datetime.now()}\n"
                    
                    # Convert to pandas DataFrame for analysis
                    candles_df = pd.DataFrame.from_dict(candles)
                    
                    # Prepare output text
                    output += f"Timeframe: {frame}s, Period: {time}s\n"
                    output += f"First {min(5, len(candles_df))} candles:\n{candles_df.head().to_string()}\n"
                    output += f"Last {min(5, len(candles_df))} candles:\n{candles_df.tail().to_string()}\n"
                    output += f"Number of candles: {len(candles_df)}\n"
                    output += "-" * 50 + "\n"
                    
                    # Print to console and write to file
                    print(output)
                    f.write(output)
                    f.flush()  # Ensure the output is written immediately
        
    finally:
        await client.disconnect()

asyncio.run(main())