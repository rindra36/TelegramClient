# Import library
import traceback
import asyncio
import logging
from telethon import events

# Import function from src
from src.utils import load_chats, load_credentials, setup_logging, setup_client

# Import class from src/class
from PocketOptionAPI import PocketOptionAPI

# Import class from src
from src.YoussefTrading import YoussefTradingBot
from src.ViperionTrading import ViperionTradingBot
from src.AlexTrading import AlexTradingBot
from src.AxelTrading import AxelTradingBot
from src.POWSTrading import POWSTradingBot
from src.OPTRTrading import OPTRTradingBot
from src.MatthewTrading import MatthewTradingBot
from src.VictorTrading import VictorTradingBot
from src.JuliaTrading import JuliaTradingBot

# Constants
SESSION_NAME = 'MainTradingAccountPrimary'

class MainTradingAccountPrimary():
    def __init__(self):
        setup_logging(SESSION_NAME)
        self.credentials = load_credentials()
        self.chats = load_chats()
        self.client = setup_client(SESSION_NAME, self.credentials['id'], self.credentials['hash'])
        self.pocket_option = PocketOptionAPI(1, 'real')

    async def start(self):
        """Start the trading bot."""

        # Handler of message from Youseff
        @self.client.on(events.NewMessage(chats=self.chats['Youseff']))
        async def message_handler(event):
            bot = YoussefTradingBot(self.pocket_option)
            await bot.start(event.raw_text)

        # Handler of message from Viperion
        @self.client.on(events.NewMessage(chats=self.chats['Viperion']))
        async def message_handler(event):
            bot = ViperionTradingBot(self.pocket_option)
            await bot.start(event.raw_text)

        # Handler of message from Alex
        @self.client.on(events.NewMessage(chats=self.chats['Alex']))
        async def message_handler(event):
            bot = AlexTradingBot(self.pocket_option)
            await bot.start(event.raw_text)

        # Handler of message from Axel
        @self.client.on(events.NewMessage(chats=self.chats['Axel']))
        async def message_handler(event):
            bot = AxelTradingBot(self.pocket_option)
            await bot.start(event.raw_text)

        # Handler of message from POWS
        @self.client.on(events.NewMessage(chats=self.chats['POWS']))
        async def message_handler(event):
            bot = POWSTradingBot(self.pocket_option)
            await bot.start(event.raw_text)

        # Handler of message from Matthew
        @self.client.on(events.NewMessage(chats=self.chats['Matthew']))
        async def message_handler(event):
            bot = MatthewTradingBot(self.pocket_option)
            await bot.start(event.raw_text)
            
        # Handler of message from Victor
        @self.client.on(events.NewMessage(chats=self.chats['Victor']))
        async def message_handler(event):
            bot = VictorTradingBot(self.pocket_option)
            await bot.start(event.raw_text)
            
        # Handler of message from Julia
        @self.client.on(events.NewMessage(chats=self.chats['Julia']))
        async def message_handler(event):
            bot = JuliaTradingBot(self.pocket_option)
            await bot.start(event.raw_text)

        await self.client.start()
        await self.client.run_until_disconnected()


def main():
    """Main entry point for the trading bot."""
    try:
        bot = MainTradingAccountPrimary()
        asyncio.run(bot.start())
    except Exception as e:
        traceback_exception = traceback.format_exc()
        logging.error(f"MainTradingAccountPrimary initialization failed: {e}", traceback_exception)

if __name__ == "__main__":
    main()