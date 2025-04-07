# from PocketOptionAPI import PocketOptionAPI
from PocketOptionMethod import PocketOptionMethod
import asyncio

async def main():
    # api = PocketOptionAPI()
    api = PocketOptionMethod()
    # channel = 'Youssef'
    # api.set_channel_data(channel, {
    #     'action': 'BUY',
    #     'asset': 'EURJPY_otc',
    #     'expiration': 60,
    #     'amount': 1,
    # })
    # trade_id = await api.trade(channel, use_v1=True, check_win=False)
    # print(trade_id)

if __name__ == "__main__":
    asyncio.run(main())