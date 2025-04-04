import yfinance as yf                                                                           # For fetching stock prices
import pandas as pd                                                                             # For handling data in tables
import numpy as np                                                                              # For math calculations
import matplotlib.pyplot as plt                                                                 # For making graphs
from datetime import datetime, timedelta                                                        # For working with dates


class PortfolioSimulator:                                                                       # Define the PortfolioSimulator class to manage the app 
    
    def __init__(self):                                                                         # Initialize the class with no default cash (user will input it)
        
        self.initial_cash = None                                                                # Set starting money to None (will be set by user)
        
        self.tickers = []                                                                       # Create an empty list for stock tickers
                                                                                   
        self.weights = []                                                                       # Create an empty list for weights
        
        self.data = None                                                                        # Set a variable to hold stock price data (starts as None)
        
        self.portfolio_value = None                                                             # Set a variable to track portfolio value (starts as None)
        self.sharpe_ratio = None                                                                # Set a variable for the Sharpe Ratio (starts as None)                                                 
        self.best_stock = None                                                                  # Set a variable for the best-performing stock (starts as None)
        
        self.worst_stock = None                                                                 # Set a variable for the worst-performing stock (starts as None)
        
        self.start_date = None                                                                  # Set a variable for the start date (starts as None)
        
        self.end_date = datetime.now().strftime("%Y-%m-%d")                                     # Set the end date to today’s date in "YYYY-MM-DD" format

    
    def get_user_input(self):                                                                   # Method to get input from the user
        
        print("\n=== Portfolio Simulator ===")
        
        while True:                                                                             # Ask for initial investment amount
            
            cash_input = input("Enter initial investment amount (e.g., 10000): ").strip()       # Get the initial cash input from the user
            
            try:                                                                                # Try to convert it to a number
                
                self.initial_cash = float(cash_input)                                           # Convert input to a float
                
                if self.initial_cash <= 0:                                                      # Check if it’s positive
                    
                    print("Error: Investment must be positive!")                                
                    continue
                
                break
            
            except ValueError:
                print("Error: Enter a valid number (e.g., 10000)!")

        
        print(f"Starting with ${self.initial_cash:.2f}. Enter stock tickers and weights (e.g., 'AAPL 0.5' for 50%). Type 'done' when finished.")
        
        while True:                                                                             # Keep asking for stocks until 'done'
            
            entry = input("Stock ticker and weight (e.g., 'AAPL 0.5'): ").strip().lower()       # Get input and clean it up
            
            if entry == "done":                                                                 # Check if user is done
                
                if not self.tickers:                                                            # Ensure at least one stock was entered
                    print("Error: You must enter at least one stock!")
                    continue
                
                break
            
            try:
                ticker, weight = entry.split()                                                # Split input into ticker and weight
                
                weight = float(weight)                                                        # Convert weight to a number
                
                if weight <= 0:
                    print("Error: Weight must be positive!")
                    continue
                self.tickers.append(ticker.upper())                                           # Add ticker (uppercase) to list
                
                self.weights.append(weight)                                                   # Add weight to list
            except ValueError:
                print("Invalid input! Use format 'TICKER WEIGHT' (e.g., 'AAPL 0.5').")

        
        total_weight = sum(self.weights)                                                      # Calculate total weight
        
        if total_weight == 0:                                                                 # Check if total weight is valid
            print("Error: Weights must sum to a positive value! Starting over.")
            self.tickers = []
            self.weights = []
            self.initial_cash = None
            return
        
        self.weights = [w / total_weight for w in self.weights]                               # Normalize weights to add up to 1

        
        while True:                                                                           # Ask for time period
            
            period = input("Enter time period (e.g., '1y' for 1 year, '6m' for 6 months): ").lower() # Get time period input
    
            try:
                if period.endswith('y'):
                    years = int(period[:-1])
                    self.start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
                elif period.endswith('m'):
                    months = int(period[:-1])
                    self.start_date = (datetime.now() - timedelta(days=months * 30)).strftime("%Y-%m-%d")
                else:
                    print("Invalid period! Use '1y', '6m', etc.")
                    continue
                break
            except ValueError:
                print("Invalid period! Use '1y', '6m', etc.")

    
    def fetch_data(self):                                                                     # Method to fetch stock data
        
        if not self.tickers:                                                                  # Check if tickers were entered
            print("Error: No stocks entered yet! Run get_user_input() first.")
            return
        try:
            
            self.data = yf.download(self.tickers, start=self.start_date, end=self.end_date)["Close"] # Fetch closing prices for all tickers
            
            if self.data.empty:                                                               # Check if data is empty
                print("Error: No valid data returned for these tickers!")
                self.data = None
                return
            if self.data.isna().all().any():                                                  # Check for missing data
                print("Error: Some tickers have no data!")
                self.data = None
                return
            
            self.data = self.data.dropna()                                                    # Remove rows with missing values
            
            if len(self.data) < 2:                                                            # Ensure enough data points
                print("Error: Not enough data points to analyze!")
                self.data = None
                return
        except Exception as e:                                                                # If fetching fails, print error
            print(f"Error fetching data: {str(e)}")
            self.data = None

    
    def simulate_portfolio(self):                                                             # Method to simulate the portfolio
        
        if self.data is None:                                                                 # Check if data exists
            print("Error: No data available! Run fetch_data() first.")
            return
        
        shares = [self.initial_cash * w / self.data[ticker].iloc[0] for w, ticker in zip(self.weights, self.tickers)] # Calculate shares based on initial prices and weights
        
        self.portfolio_value = (self.data * shares).sum(axis=1)                               # Calculate daily portfolio value
        
        daily_returns = self.portfolio_value.pct_change().dropna()                            # Calculate daily returns
        
        total_return = (self.portfolio_value.iloc[-1] - self.initial_cash) / self.initial_cash # Calculate total return
        
        annual_return = (1 + total_return) ** (252 / len(self.data)) - 1                      # Calculate annualized return
        
        annual_vol = daily_returns.std() * np.sqrt(252)                                       # Calculate annualized volatility
        
        risk_free_rate = 0.02                                                                 # Set risk-free rate
        
        self.sharpe_ratio = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0 # Calculate Sharpe Ratio
        
        individual_returns = [(self.data[t].iloc[-1] / self.data[t].iloc[0] - 1) * 100 for t in self.tickers] # Calculate individual stock returns
        
        self.best_stock = self.tickers[np.argmax(individual_returns)]                         # Find best stock
        
        self.worst_stock = self.tickers[np.argmin(individual_returns)]                        # Find worst stock

    
    def compare_to_benchmark(self):                                                           # Method to get benchmark data
        
        if self.data is None:                                                                 # Check if portfolio data exists
            print("Error: No portfolio data to compare!")
            return None
        
        benchmark_ticker = "SPY"                                                              # Set benchmark ticker
        try:
            benchmark_data = yf.download(benchmark_ticker, start=self.start_date, end=self.end_date)["Close"] # Download SPY closing prices
            
            benchmark_value = (benchmark_data / benchmark_data.iloc[0]) * self.initial_cash   # Scale to initial cash
            
            return benchmark_value                                                            # Return benchmark value
        except Exception as e:
            print(f"Error fetching benchmark data: {str(e)}")
            return None

    
    def display_and_save_results(self):                                                       # Method to display and save results
        
        if self.portfolio_value is None:                                                      # Check if portfolio was simulated
            print("Error: Run simulate_portfolio() first!")
            return
        
        total_return = (self.portfolio_value.iloc[-1] - self.initial_cash) / self.initial_cash # Calculate total return
        
        daily_returns = self.portfolio_value.pct_change().dropna()                            # Calculate daily returns
        
        annual_return = (1 + total_return) ** (252 / len(self.data)) - 1                      # Calculate annualized return
        
        annual_vol = daily_returns.std() * np.sqrt(252)                                       # Calculate annualized volatility
        
        max_drawdown = ((self.portfolio_value.cummax() - self.portfolio_value) / self.portfolio_value.cummax()).max() # Calculate maximum drawdown

        
        print(f"\n=== Portfolio Simulator Results ===\n")
        
        print(f"Stocks: {', '.join([f'{t} ({w:.0%})' for t, w in zip(self.tickers, self.weights)])}")
        print(f"Time Period: {self.start_date} to {self.end_date}")
        print(f"Initial Investment: ${self.initial_cash:.2f}")
        print(f"Final Portfolio Value: ${self.portfolio_value.iloc[-1]:.2f}")
        print(f"Total Return: {total_return * 100:.2f}%")
        print(f"Annualized Return: {annual_return * 100:.2f}%")
        print(f"Annualized Volatility: {annual_vol * 100:.2f}%")
        print(f"Maximum Drawdown: {max_drawdown * 100:.2f}%")
        print(f"Sharpe Ratio: {self.sharpe_ratio:.2f}")
        print(f"Best Performer: {self.best_stock}")
        print(f"Worst Performer: {self.worst_stock}")

        
        plt.figure(figsize=(10, 6))                                                           # Set up plot
        plt.plot(self.portfolio_value, label="Portfolio Value", color="teal")
        
        benchmark_value = self.compare_to_benchmark()                                         # Get and plot benchmark
        if benchmark_value is not None:
            plt.plot(benchmark_value, label="S&P 500 (SPY)", color="gray")
        plt.title("Portfolio Performance vs. S&P 500")
        plt.xlabel("Date")
        plt.ylabel("Value ($)")
        plt.legend()
        plt.grid(True)
        plt.savefig("portfolio_result.png")
        print("Plot saved as 'portfolio_result.png'")
        self.portfolio_value.to_csv("portfolio_value.csv")
        print("Portfolio data saved as 'portfolio_value.csv'")
        plt.show()

# Run the app
if __name__ == "__main__":
    
    sim = PortfolioSimulator()                                                               # Create simulator instance
    sim.get_user_input()                                                                     # Get user input
    sim.fetch_data()                                                                         # Fetch data
    sim.simulate_portfolio()                                                                 # Simulate portfolio
    sim.display_and_save_results()                                                           # Show results