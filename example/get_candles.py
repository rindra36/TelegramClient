from BinaryOptionsToolsV2.pocketoption import PocketOptionAsync
from BinaryOptionsTools import pocketoption

import pandas as pd
import asyncio
import nest_asyncio
from datetime import datetime
import os

nest_asyncio.apply()

# Main part of the code
async def main(ssid: str):
    # Create output directory if it doesn't exist
    output_dir = 'candles_data'
    os.makedirs(output_dir, exist_ok=True)

    time = 3600
    frame = 60

    # The api automatically detects if the 'ssid' is for real or demo account
    api = PocketOptionAsync(ssid)
    await asyncio.sleep(5)

    candles = await api.get_candles("EURUSD_otc", frame, time)
    # print(f"Raw Candles: {candles}")
    candles_pd = pd.DataFrame.from_dict(candles)

    # Create filename with timestamp and parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/candles_V2_EURUSD_otc_time{time}_frame{frame}_{timestamp}.csv"

    # Save to CSV file
    candles_pd.to_csv(filename, index=False)
    print(f"Saved candles to: {filename}")

    await asyncio.sleep(5)
    api_v1 = pocketoption(ssid)

    candles_V1 = api_v1.GetCandles("EURUSD_otc", frame, None, time, 1, '1min')
    print(candles_V1)

    # Create filename with timestamp and parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/candles_V1_EURUSD_otc_time{time}_frame{frame}_{timestamp}.csv"
    # Save to CSV file
    candles_V1.to_csv(filename)
    print(f"Saved candles to: {filename}")
    
    # # Candñes are returned in the format of a list of dictionaries
    # # times = [ 3600 * i for i in range(1, 11)]
    # # time_frames = [ 1, 5, 15, 30, 60, 300]
    # times = [3600]
    # time_frames = [60]
    # for time in times:
    #     for frame in time_frames:
            
    #         candles = await api.get_candles("EURUSD_otc", frame, time)
    #         # print(f"Raw Candles: {candles}")
    #         candles_pd = pd.DataFrame.from_dict(candles)

    #         candles_V1 = await api_v1.GetCandles("EURUSD_otc", frame, None, time)
    #         candles_pd_V1 = pd.DataFrame.from_dict(candles_V1)

    #         # Create filename with timestamp and parameters
    #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #         filename = f"{output_dir}/candles_V2_EURUSD_otc_time{time}_frame{frame}_{timestamp}.csv"

    #         # Create filename with timestamp and parameters
    #         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #         filename = f"{output_dir}/candles_V1_EURUSD_otc_time{time}_frame{frame}_{timestamp}.csv"

    #         # Save to CSV file
    #         candles_pd.to_csv(filename, index=False)
    #         print(f"Saved candles to: {filename}")

    #         # Save to CSV file
    #         candles_pd_V1.to_csv(filename, index=False)
    #         print(f"Saved candles to: {filename}")

    #         # Still print to console for immediate feedback
    #         print(f"Candles: {candles_pd}")
    
if __name__ == '__main__':
    ssid = input('Please enter your ssid: ')
    asyncio.run(main(ssid))
    