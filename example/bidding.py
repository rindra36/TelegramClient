from BinomoAPI import BinomoAPI
import asyncio, time

async def main():
    api = BinomoAPI(AddLogging=True)
    # balance = api.getCurrentBalance()
    # print(balance)
    # time.sleep(1)
    await api.Call("Z-CRY/IDX", 60, 1000, is_demo=True)
    time.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())