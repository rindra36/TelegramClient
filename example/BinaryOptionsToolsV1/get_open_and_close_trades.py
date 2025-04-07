import asyncio

from PocketOptionMethod import PocketOptionMethod
from datetime import datetime as Datetime
import time

async def main():
    # The api automatically detects if the 'ssid' is for real or demo account
    print(f'Initialization : ', Datetime.now())
    api = PocketOptionMethod()
    print(f'After initialization : ', Datetime.now())
    _ = await api.buy(asset="EURUSD_otc", amount=1.0, time=60, check_win=False)
    print(f'After buy : ', Datetime.now())
    _ = await api.sell(asset="EURUSD_otc", amount=1.0, time=60, check_win=False)
    print(f'After sell : ', Datetime.now())
    # This is the same as setting checkw_win to true on the api.buy and api.sell functions
    await asyncio.sleep(10)
    opened_deals = api.api_v1.api.check_open()
    print(f'After opened deals : ', Datetime.now())
    print(f"Opened deals: {opened_deals}\nNumber of opened deals: {len(opened_deals)} (should be at least 2)")
    # time.sleep(62)  # Wait for the trades to complete
    # print(f'After sleep : ', Datetime.now())
    # closed_deals = api.api_v1.api.check_order_closed(_)
    # print(f'After closed deals : ', Datetime.now())
    # print(f"Closed deals: {closed_deals}\nNumber of closed deals: {len(closed_deals)} (should be at least 2)")

if __name__ == '__main__':
    asyncio.run(main())