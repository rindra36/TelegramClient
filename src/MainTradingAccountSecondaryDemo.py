# Import library
import sys
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
from src.TTSPTrading import TTSPTradingBot
from src.TTSPVIPTrading import TTSPVIPTradingBot
from src.PFATTrading import PFATTradingBot
from src.MikhailTrading import MikhailTradingBot
from src.GavinBotM1Trading import GavinBotM1TradingBot
from src.GavinBotM5Trading import GavinBotM5TradingBot
from src.PrinceTrading import PrinceTradingBot
from src.AureaVIPMoneyMakerTrading import AureaVIPMoneyMakerTradingBot
from src.TPSAutoTrading import TPSAutoTradingBot
from src.TPSTrading import TPSTradingBot

# Constants
SESSION_NAME = 'MainTradingAccountSecondaryDemo'

class MainTradingAccountSecondaryDemo():
    def __init__(self):
        setup_logging(SESSION_NAME)
        self.credentials = load_credentials()
        self.chats = load_chats()
        self.client = setup_client(SESSION_NAME, self.credentials['id'], self.credentials['hash'])
        self.pocket_option = PocketOptionAPI(2)

    async def start(self):
        """Start the trading bot."""

        # Handler of message from Youseff
        # @self.client.on(events.NewMessage(chats=self.chats['Youseff']))
        # async def message_handler(event):
        #     bot = YoussefTradingBot(self.pocket_option)
        #     bot.MAX_RETRY = 5
        #     await bot.start(event.raw_text)

        # Handler of message from Viperion
        # @self.client.on(events.NewMessage(chats=self.chats['Viperion']))
        # async def message_handler(event):
        #     bot = ViperionTradingBot(self.pocket_option)
        #     await bot.start(event.raw_text)

        # Handler of message from Alex
        @self.client.on(events.NewMessage(chats=self.chats['Alex']))
        @self.client.on(events.NewMessage(chats='me'))
        async def message_handler(event):
            bot = AlexTradingBot(self.pocket_option)
            await bot.start(event.raw_text)

        # Handler of message from Axel
        # @self.client.on(events.NewMessage(chats=self.chats['Axel']))
        # async def message_handler(event):
        #     bot = AxelTradingBot(self.pocket_option)
        #     await bot.start(event.raw_text)

        # Handler of message from POWS
        # @self.client.on(events.NewMessage(chats=self.chats['POWS']))
        # async def message_handler(event):
        #     bot = POWSTradingBot(self.pocket_option)
        #     await bot.start(event.raw_text)

        # Handler of message from Matthew
        # @self.client.on(events.NewMessage(chats=self.chats['Matthew']))
        # async def message_handler(event):
        #     bot = MatthewTradingBot(self.pocket_option)
        #     bot.MAX_RETRY = 5
        #     await bot.start(event.raw_text)
            
        # Handler of message from Victor
        # @self.client.on(events.NewMessage(chats=self.chats['Victor']))
        # async def message_handler(event):
        #     bot = VictorTradingBot(self.pocket_option)
        #     bot.MAX_RETRY = 5
        #     await bot.start(event.raw_text)
            
        # Handler of message from Julia
        # @self.client.on(events.NewMessage(chats=self.chats['Julia']))
        # async def message_handler(event):
        #     bot = JuliaTradingBot(self.pocket_option)
        #     await bot.start(event.raw_text)

        # Handler of message from TTPS
        @self.client.on(events.NewMessage(chats=self.chats['TTPS']))
        @self.client.on(events.NewMessage(chats='me'))
        async def message_handler(event):
            bot = TTSPTradingBot(self.pocket_option)
            await bot.start(event.raw_text)

        # Handler of message from TTPSVIP
        @self.client.on(events.NewMessage(chats=self.chats['TTPSVIP']))
        @self.client.on(events.NewMessage(chats='me'))
        async def message_handler(event):
            bot = TTSPVIPTradingBot(self.pocket_option)
            await bot.start(event.raw_text)

        # Handler of message from PFAT
        # @self.client.on(events.NewMessage(chats=self.chats['PFAT']))
        # async def message_handler(event):
        #     bot = PFATTradingBot(self.pocket_option)
        #     # bot.MAX_RETRY = 5
        #     await bot.start(event.raw_text)

        # Handler of message from Mikhail
        # @self.client.on(events.NewMessage(chats=self.chats['Mikhail']))
        # async def message_handler(event):
        #     bot = MikhailTradingBot(self.pocket_option)
        #     bot.MAX_RETRY = 5
        #     await bot.start(event.raw_text)

        # Handler of message from GavinBotM1
        # @self.client.on(events.NewMessage(chats=self.chats['GavinBotM1']))
        # async def message_handler(event):
        #     bot = GavinBotM1TradingBot(self.pocket_option)
        #     await bot.start(event.raw_text)

        # Handler of message from GavinBotM5
        # @self.client.on(events.NewMessage(chats=self.chats['GavinBotM5']))
        # async def message_handler(event):
        #     bot = GavinBotM5TradingBot(self.pocket_option)
        #     await bot.start(event.raw_text)

        # Handler of message from Prince
        # @self.client.on(events.NewMessage(chats=self.chats['Prince']))
        # async def message_handler(event):
        #     bot = PrinceTradingBot(self.pocket_option)
        #     await bot.start(event.raw_text)

        # Handler of message from AureaVIPMoneyMaker
        # @self.client.on(events.NewMessage(chats=self.chats['AureaVIPMoneyMaker']))
        # async def message_handler(event):
        #     bot = AureaVIPMoneyMakerTradingBot(self.pocket_option)
        #     await bot.start(event.raw_text)

        # Handler of message from TPSAuto
        @self.client.on(events.NewMessage(chats=self.chats['TPSAuto']))
        @self.client.on(events.NewMessage(chats='me'))
        async def message_handler(event):
            bot = TPSAutoTradingBot(self.pocket_option)
            await bot.start(event.raw_text)

        # Handler of message from TPS
        @self.client.on(events.NewMessage(chats=self.chats['TPS']))
        @self.client.on(events.NewMessage(chats='me'))
        async def message_handler(event):
            bot = TPSTradingBot(self.pocket_option)
            await bot.start(event.raw_text)

        await self.client.start()
        await self.client.run_until_disconnected()


def main():
    """Main entry point for the trading bot."""
    try:
        bot = MainTradingAccountSecondaryDemo()
        asyncio.run(bot.start())
    except Exception as e:
        traceback_exception = traceback.format_exc()
        logging.error(f"MainTradingAccountSecondaryDemo initialization failed: {e}", traceback_exception)

if __name__ == "__main__":
    main()