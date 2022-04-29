# -*- coding: utf-8 -*-
"""
Created on Sat Nov  6 12:49:42 2021

@author: marcu
"""

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from stock_pack import stock
from scipy.optimize import minimize

class portfolio:

    """
    Finance class to perform calculations and visualizations over multiple
    stocks using data from yahoo finance.

    ...
    Attributes
    ----------
    tickers: list
        list with stock tickers as in yahoo finance website
    start : str or datetime
        start date for downloading financial data of the stock of interest (YYYY-MM-DD)
    end : str or datetime
        end date for downloading financial data of the stock of interest (YYYY-MM-DD)
    period : str
        valid periods:
        1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        frequency of the data to be downloaded (daily, monthly etc.)

    Methods
    -------
    create_dict(self)
        downloads stocks' data from yahoo finance website and performs cleaning
        and wranling of the data and returns a dictionary containing dataframes
        with data of each stock
    plot_portfolio(self, col)
        plots the performance of each security that belongs to the portfolio
    plot_port_ret(self, weights)
        plot the performance of the portfolio for a given allocation (weights)
        and also plots the performance of a benchmark index
    volatility(self)
        calculates the annualized volatility of a portfolio
    sec_return(self)
        calculates the annualized return each security in the portfolio and
        return a dataframe
    correlation(self)
        returns the correlation matrix between the securities of the portfolio
    portfolio_return(self, weights)
        calculates the total return of the portfolio for a given time period
    expected_return(self, weights)
        calculates the annualized return of a portfolio
    portfolio_var(self, weights)
        calculates the annualized portfolio variance
    portfolio_vol(self, weights)
        calculates the annualized portfolio Volatility
    diversifiable_risk(self, weights)
        calculates the diversifiable_risk of the portfolio also known as the
        unsystematic risk
    get_beta(self)
        Calculates the beta of each stock that belongs to the portfolio. Beta is
        a measure of the relative volatility of a stock in relation to the overall
        market
    portfolio_beta(self, weights)
        calculates the beta of the portfolio as a whole
    weights(self)
        generates a random array with the share of each stock in the portfolio
    get_sharpe(self, weights)
        calculates the sharpe ratio of the portfolio using the brazilian risk
        free rate (selic)
    perform_optimization(self)
        performs 5000 simulations of portfolios with different allocations and picks
        the one with the highest sharpe ratio and returns the optimal allocation,
        optimal return and optimal Volatility
    plot_optimization(self, max_vol, max_ret, frontier_vol, frontier_y)
        plots the simulated portfolios with a scatter plot and highlights the
        optimal portfolio with a red dot. Also, it plots the efficient frontier
        curve
    calculate_frontier(self)
        calculates the x and y coordinates of the efficient frontier curve and
        returns them as arrays
    """

    def __init__(self, tickers, start, end, period = '1d'):
        """
        Constructs the portfolio object with all the necessary attributes.

        Parameters
        ----------
        tickers: list
            list with securities tickers as in yahoo finance website
        start : str or datetime
            start date for downloading financial data of the securities in the
            tickers list (YYYY-MM-DD)
        end : str or datetime
            end date for downloading financial data of the securities in the
            tickers list (YYYY-MM-DD)
        period : str
            valid periods:
            1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            frequency of the data to be downloaded (daily, monthly etc.)
        """

        self.tickers = tickers
        self.start = start
        self.end = end
        self.period = period
        self.data = dict()

    def create_dict(self):
        """
        downloads stocks' data from yahoo finance website and performs cleaning
        and wranling of the data and returns a dictionary containing dataframes
        with data of each stock.

        Parameters
        ----------
        portfolio object

        Returns
        -------
        None

        """
        ohlcv_data = {}

        for ticker in self.tickers:
            ohlcv_data[ticker] = yf.download(ticker, self.start, self.end)
            ohlcv_data[ticker]['daily_return'] = ohlcv_data[ticker]['Adj Close']/ohlcv_data[ticker]['Adj Close'].shift(1)-1
            ohlcv_data[ticker].iloc[0,6]=0
            ohlcv_data[ticker]['cum_return'] = (ohlcv_data[ticker]['Adj Close']/ohlcv_data[ticker]['Adj Close'][0])-1
            ohlcv_data[ticker]['log_return'] = np.log(ohlcv_data[ticker]['Adj Close']/ohlcv_data[ticker]['Adj Close'].shift(1))
            ohlcv_data[ticker].iloc[0,8]=0
            ohlcv_data[ticker]['cum_log_return'] = np.exp(np.log(ohlcv_data[ticker]['Adj Close']/ohlcv_data[ticker]['Adj Close'][0]))-1

        self.data = ohlcv_data

    def plot_portfolio(self, col='Close'):
        """
        plots the performance of each security that belongs to the portfolio.

        Parameters
        ----------
        col: str
            Date choosen to be ploted (Open, Close, Adj Close)

        Returns
        -------
        None
        """

        fig = go.Figure()
        traces = []
        for ticker in self.tickers:
            x = self.data[ticker].index
            y = (self.data[ticker][col]/self.data[ticker][col].iloc[0])*100

            trace = go.Scatter(
                x = x,
                y = y,
                mode = 'lines',
                line=dict(width=1.5),
                connectgaps = True,
                name = ticker
                )
            traces.append(trace)

        layout = go.Layout(title = 'Portfolio prices',
                            xaxis = dict(title = 'Date'),
                            yaxis = dict(title = 'stock price'),
                            showlegend = True
                            )


        fig = go.Figure(data = traces, layout = layout)
        fig.show()

    def plot_port_ret(self, weights):
        """
        plot the performance of the portfolio for a given allocation (weights)
        and also plots the performance of a benchmark index.

        Parameters
        ----------
        weights: list
            list with the allocation of each security of the porfolio object

        Returns
        -------
        None
        """

        start = self.start
        end = self.end

        sec_return = self.sec_return()
        sec_return['portfolio'] = np.dot(sec_return,weights)
        sec_return['port_return'] = (1 + sec_return['portfolio']).cumprod()-1
        ibov = yf.download('^BVSP', start, end)['Adj Close']
        sec_return['IBOV'] = (ibov/ibov[0]) - 1
        sec_return.iloc[0,1]=0


        fig = go.Figure()

        x = sec_return.index
        y = sec_return['port_return']

        trace1 = go.Scatter(
            x = x,
            y = y,
            mode = 'lines',
            line=dict(width=1.5),
            connectgaps = True,
            name = 'Portfolio'
            )

        trace2 = go.Scatter(
            x = x,
            y = sec_return['IBOV'],
            mode = 'lines',
            line=dict(width=1.5),
            connectgaps = True,
            name = 'IBOV'
            )

        trace3 = go.Scatter(
            x = ['2022-04-20'],
            y = [y.iloc[-1]],
            text = [str(round(y.iloc[-1]*100,2)) + '%'],
            mode = 'markers+text',
            marker=dict(color='blue', size=10),
            textfont=dict(color='black', size=20),
            textposition='top right',
            showlegend=False
            )

        trace4 = go.Scatter(
            x = ['2022-04-20'],
            y = [sec_return['IBOV'].iloc[-1]],
            text = [str(round(sec_return['IBOV'].iloc[-1]*100,2)) + '%'],
            mode = 'markers+text',
            marker=dict(color='red', size=10),
            textfont=dict(color='black', size=20),
            textposition='top right',
            showlegend=False
            )

        layout = go.Layout(title = 'portfolio performance',
                            xaxis = dict(title = 'Date'),
                            yaxis = dict(title = 'stock price'),
                            showlegend = True
                            )

        fig = go.Figure(data = [trace1, trace2, trace3, trace4], layout = layout)
        fig.show()


    def volatility(self):
        """
        calculates the annualized volatility of each security of the portfolio.

        Parameters
        ----------
        portfolio object

        Returns
        -------
        vol_series: Pandas Series
            Series with the annualized volatilities of each security of the portfolio
        """

        data = self.data
        vol_dict = {}
        for ticker in self.tickers:
            vol_dict[ticker] = data[ticker]['log_return'].std()*252**0.5*100

        vol_series = pd.Series(vol_dict)
        return vol_series

    def sec_return(self):
        """
        calculates the annualized return each security in the portfolio and
        returns a dataframe.

        Parameters
        ----------
        portfolio object

        Returns
        -------
        sec_return: Pandas DataFrame
            DataFrame with the daily log returns of each security of the portfolio
        """

        data = self.data
        tickers = self.tickers
        sec_dict = {}

        for ticker in tickers:
            sec_dict[ticker] = data[ticker]['log_return']

        sec_return = pd.DataFrame(sec_dict)
        return sec_return

    def correlation(self):
        """
        returns the correlation matrix between the securities of the portfolio.

        Parameters
        ----------
        portfolio object

        Returns
        -------
        DataFrame containing the correlation matrix
        """

        sec_return = self.sec_return()
        return sec_return.corr()

    def portfolio_return(self, weights):
        """
        calculates the total return of the portfolio for a given time period.

        Parameters
        ----------
        weights: list
            list with the allocation of each security of the porfolio object

        Returns
        -------
        float:
            total return of the portfolio for the given time interval
        """

        data = self.data
        rets = []
        for ticker in self.tickers:
            rets.append(data[ticker]['cum_log_return'].iloc[-1])

        return np.sum(weights*rets)

    def expected_return(self, weights):
        """
        calculates the annualized return of a portfolio.
        Parameters
        ----------
        weights: list
            list with the allocation of each security of the porfolio object

        Returns
        -------
        float:
            annualized return of the portfolio

        """

        data = self.data
        ret = []
        for ticker in self.tickers:
            ret.append(data[ticker]['log_return'].mean())

        return np.sum(weights*ret)*252

    def portfolio_var(self, weights):
        """
        calculates the annualized portfolio variance.

        Parameters
        ----------
        weights: list
            list with the allocation of each security of the porfolio object

        Returns
        -------
        float:
            annualized variance of the portfolio
        """

        sec_return = self.sec_return()
        pfolio_var = np.dot(weights.T,np.dot(sec_return.cov()*252, weights))
        return pfolio_var

    def portfolio_vol(self, weights):
        """
        calculates the annualized portfolio volatility.

        Parameters
        ----------
        weights: list
            list with the allocation of each security of the porfolio object

        Returns
        -------
        pfolio_vol : float
            annualized volatility of the portfolio
        """

        sec_return = self.sec_return()
        pfolio_vol = (np.dot(weights.T,np.dot(sec_return.cov()*252, weights)))**0.5
        return pfolio_vol

    def diversifiable_risk(self, weights):
        """
        calculates the diversifiable_risk of the portfolio also known as the
        unsystematic risk.

        Parameters
        ----------
        weights: list
            list with the allocation of each security of the porfolio object

        Returns
        -------
        dr : float
            measure of the unsystematic risk
        """

        sec_return = self.sec_return()
        pfolio_var = self.portfolio_var(weights)
        dr = pfolio_var - np.dot(np.square(weights), sec_return.var()*252)

        return dr

    def get_beta(self):
        """
        Calculates the beta of each stock that belongs to the portfolio. Beta is
        a measure of the relative volatility of a stock in relation to the overall
        market.

        Parameters
        ----------
        portfolio object

        Returns
        -------
        beta_df : DataFrame
            DataFrame with the beta value of each security of the portfolio

        """

        start = self.start
        end = self.end
        stocks = self.tickers
        beta_df = pd.Series()

        for ticker in stocks:
            st = stock(ticker,start,end)
            beta_df[ticker] = st.get_beta()

        return beta_df

    def portfolio_beta(self, weights):
        """
        Calculates the beta of the portfolio as a whole.

        Parameters
        ----------
        weights: list
            list with the allocation of each security of the porfolio object

        Returns
        -------
        beta : float
            beta value of the whole portfolio

        """

        start = self.start
        end = self.end

        sec_return = self.sec_return()
        sec_return['portfolio'] = np.dot(sec_return,weights)
        ibov = yf.download('^BVSP', start, end)['Adj Close']
        sec_return['IBOV'] = np.log(ibov/ibov.shift(1))
        sec_return.iloc[0,1]=0

        cov = sec_return.cov()*252
        cov_rx_rm = cov.iloc[0,1]
        market_var = sec_return['IBOV'].var()*252

        beta = cov_rx_rm/market_var
        return beta

    def weights(self):
        """
        generates a random array with the share of each stock in the portfolio.

        Returns
        -------
        w : array
            array with the share of each security in the portfolio
        """
        n = len(self.tickers)
        w = np.random.random(n)
        w = w/sum(w)

        return w

    def get_sharpe(self, weights):
        """
        Calculates the sharpe ratio of the portfolio using the brazilian risk
        free rate (selic). Sharpe ratio is a metric that measures the return of a investment compared to its risk.

        Parameters
        ----------
        weights: list
            list with the allocation of each security of the porfolio object

        Returns
        -------
        sharpe : float
            Sharpe ratio value

        """

        rf = 0.1175
        pt_er = self.expected_return(weights)
        pt_vol = self.portfolio_vol(weights)

        sharpe = (pt_er - rf)/pt_vol

        return sharpe

    def perform_optimization(self):
        """
        Performs 5000 simulations of portfolios with different allocations, picks
        the one with the highest sharpe ratio and returns the optimal allocation,
        optimal return and optimal Volatility.

        Returns
        -------
        max_vol: float
            volatility of the portfolio eith highest sharpe
        max_ret: float
            return of the portfolio eith highest sharpe
        optimal_sharpe: float
            highest sharpe ratio
        optimal_weight: float
            allocation of the optimal portfolio

        """

        pfolio_returns = []
        pfolio_vols = []
        pfolio_sharpe = []
        pfolio_weights = []

        for x in range(5000):

            weights = self.weights()
            pfolio_weights.append(weights)
            pfolio_returns.append(self.expected_return(weights))
            pfolio_vols.append(self.portfolio_vol(weights))
            pfolio_sharpe.append(self.get_sharpe(weights))

        all_weights = np.array(pfolio_weights)
        all_ret = np.array(pfolio_returns)
        all_vol = np.array(pfolio_vols)
        all_sharpe = np.array(pfolio_sharpe)

        optimal_sharpe = all_sharpe.max()
        optimal_weight = all_weights[all_sharpe.argmax()]

        max_vol = all_vol[all_sharpe.argmax()]
        max_ret = all_ret[all_sharpe.argmax()]

        return max_vol, max_ret, optimal_sharpe, optimal_weight



    def plot_optimization(self, max_vol, max_ret, frontier_vol, frontier_y):
        """
        Plots the simulated portfolios with a scatter plot and highlights the
        optimal portfolio with a red dot. Also, it plots the efficient frontier
        curve.

        Parameters
        ----------
        max_vol: float
            volatility of the portfolio eith highest sharpe
        max_ret: float
            return of the portfolio eith highest sharpe
        optimal_sharpe: float
            highest sharpe ratio
        optimal_weight: float
            allocation of the optimal portfolio
        frontier_y : array
            array with the y values of the efficient frontier curve calculated
            in calculate_frontier method
        frontier_vol : list
            list with the volatility values of the efficient frontier curve calculated
            in calculate_frontier method

        Returns
        -------
        None
        """


        pfolio_returns = []
        pfolio_vols = []


        for x in range(10000):
            weights = self.weights()
            pfolio_returns.append(self.expected_return(weights))
            pfolio_vols.append(self.portfolio_vol(weights))

        fig = go.Figure()

        x = pfolio_vols
        y = pfolio_returns

        trace1 = go.Scatter(
            x = x,
            y = y,
            mode = 'markers',
            name = 'Portfolio Optimization'
            )

        trace2 = go.Scatter(
            x = np.array(max_vol),
            y = np.array(max_ret),
            marker = dict(
                color = 'red',
                size = 10
                ),
            mode = 'markers',
            name = 'Optimal portfolio'

            )

        trace3 = go.Scatter(
            x = frontier_vol,
            y = frontier_y,
            marker = dict(
                color = 'green'
                ),
            mode = 'lines',
            name = 'Efficient Frontier'

            )

        layout = go.Layout(title = 'portfolio optimization',
                            xaxis = dict(title = 'Volatility'),
                            yaxis = dict(title = 'Return'),
                            showlegend = True
                            )

        fig = go.Figure(data = [trace1, trace2, trace3], layout = layout)
        fig.show()

    def neg_sharpe(self, weights):
        return self.get_sharpe(weights)*(-1)

    def check_sum(weights):
        return np.sum(weights)-1

    def get_vol(self, weights):
        return self.portfolio_vol(weights)

    def calculate_frontier(self):
        """
        Calculates the x and y coordinates of the efficient frontier curve and
        returns them as arrays using the scipy's minimize function to find the
        optimals portfolios given an array with a chosen number of possible returns.

        Returns
        -------
        frontier_y : array
            array with the y values of the efficient frontier curve
        frontier_vol : list
            list with the volatility values of the efficient frontier curve
        """

        bounds = ((0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1))
        init_guess = [0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10,0.10]

        frontier_y = np.linspace(0.03,0.35,100)
        frontier_vol = []

        for pos_ret in frontier_y:
            cons = ({'type':'eq', 'fun':portfolio.check_sum},
            {'type':'eq', 'fun': lambda w: self.expected_return(w)-pos_ret})

            result = minimize(self.get_vol, init_guess, method = 'SLSQP', bounds = bounds, constraints = cons)
            frontier_vol.append(result['fun'])

        return frontier_vol, frontier_y
