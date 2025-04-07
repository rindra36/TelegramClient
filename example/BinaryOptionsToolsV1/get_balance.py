from BinaryOptionsTools import pocketoption
import asyncio
import nest_asyncio
from datetime import datetime as Datetime

async def main():
    nest_asyncio.apply()
    ssid = input("Enter your ssid: ")
    demo = not bool(int(input("Do you want to use demo or real account? (0: demo, 1: real) ")))
    print(f'After input : ', Datetime.now())
    api = pocketoption(ssid, demo)
    print(f'After initialization : ', Datetime.now())
    try:
        balance = api.GetBalance()
        print(f"Balance: {balance}", Datetime.now())

        # print(f'Before calling Call : ', Datetime.now())
        # buy = api.Call(amount=1)
        # print(f"Buy: {buy}", Datetime.now())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())