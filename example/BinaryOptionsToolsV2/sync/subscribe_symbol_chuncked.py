from BinaryOptionsToolsV2.pocketoption import PocketOption

# Main part of the code
def main(ssid: str):
    # The api automatically detects if the 'ssid' is for real or demo account
    api = PocketOption(ssid)    
    stream = api.subscribe_symbol_chuncked("EURUSD_otc", 15) # Returns a candle obtained from combining 15 (chunk_size) candles
    
    # This should run forever so you will need to force close the program
    for candle in stream:
        print(f"Candle: {candle}") # Each candle is in format of a dictionary 
    
if __name__ == '__main__':
    ssid = input('Please enter your ssid: ')
    main(ssid)
    