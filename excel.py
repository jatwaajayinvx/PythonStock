import time
import xlwings as xw
import yfinance as yf

# Open a new temporary Excel workbook (no file created on disk)
wb = xw.Book()
sheet = wb.sheets[0]

# Set up headers immediately
sheet.range("A1").value = ["Symbol", "Live Price", "Last Updated"]

tickers = ["RELIANCE.NS", "TCS.NS"]

print("Streaming live prices to Excel... Press Ctrl+C in terminal to stop.")

while True:
    row = 2
    for symbol in tickers:
        try:
            # Fetch the latest price
            stock = yf.Ticker(symbol)
            price = stock.fast_info.get("last_price")
            
            # Fallback if fast_info returns None
            if price is None:
                price = stock.history(period="1d")["Close"].iloc[-1]
                
            timestamp = time.strftime("%H:%M:%S")
            
            # Write directly to Excel cells
            sheet.range(f"A{row}").value = [symbol, round(float(price), 2), timestamp]
            print(f"Updated {symbol}: {round(float(price), 2)} at {timestamp}")
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            
        row += 1
        
    # Wait 5 seconds before the next update
    time.sleep(5)
