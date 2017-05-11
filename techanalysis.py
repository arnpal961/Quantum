import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as animation

class TechIndicator(object):
    '''Calculate Technical Indicators'''
    
    def __init__(self,data,resampling_interval=None):
        '''
        Valid Intervals be like: '1min','3min','5min','1h','1D','2M'
        Data must be in pandas dataframe format with columns = high,
        low,close,open,volume.
        '''
        if resampling_interval is not None:
            data = data.resample(how={'open':'first',
                                      'high':'max',
                                      'low':'min',
                                      'close': 'last',
                                      'volume': 'sum'},
                                      rule=resampling_interval)
            self.data = data.dropna()
        else:
            self.data = data
        
        self.h = self.data['high']
        self.l = self.data['low']
        self.o = self.data['open']
        self.c = self.data['close']
        self.v = self.data['volume']
        self.d = self.data.index
        
    def pivot(self):
        return (self.l+self.h+self.c)/3
    
    def vwap(self):
        vp = self.v * TechIndicator.pivot(self)
        return vp.cumsum()/self.v.cumsum()
    
    def sma(self,n):
        return self.c.rolling(window=n).mean()
    
    def ema(self,n):
        return self.c.ewm(span=n,min_periods=n).mean().dropna()
    
    def change(self):
        return self.c.diff(periods=1) 
    
    def pct_change(self):
        return self.c.pct_change(periods=1)
    
    def gain(self):
        return np.maximum(TechIndicator.change(self),0)
    
    def loss(self):
        return np.abs(np.minimum(TechIndicator.change(self),0))
    
    def sup_res(self):
        R1 = TechIndicator.pivot(self)*2 - self.l
        R2 = TechIndicator.pivot(self)- TechIndicator.tr(self)
        S1 = TechIndicator.pivot(self)*2 - self.h
        S2 = TechIndicator.pivot(self) + TechIndicator.tr(self)
        R1_fib = TechIndicator.pivot(self)+(0.382*TechIndicator.tr(self))
        R2_fib = TechIndicator.pivot(self)+(0.618*TechIndicator.tr(self))
        R3_fib = TechIndicator.pivot(self)+(1*TechIndicator.tr(self))
        S1_fib = TechIndicator.pivot(self)-(0.382*TechIndicator.tr(self))
        S2_fib = TechIndicator.pivot(self)-(0.618*TechIndicator.tr(self))
        S3_fib = TechIndicator.pivot(self)-(1*TechIndicator.tr(self))
        rs = pd.DataFrame([R1,S1,R2,S2,R1_fib,S1_fib,R2_fib,S2_fib,R3_fib,S3_fib],
                          index=['R1','S1','R2','S2','R1_fib','S1_fib','R2_fib','S2_fib','R3_fib','S3_fib']).T
        return np.round(rs,2)
    
    def tr(self):
        true_range = pd.DataFrame([self.h-self.l,
                       np.abs(self.h - self.c[1:]),
                       np.abs(self.l - self.c[1:])]).T
        return true_range.max(axis=1)
    
    def hhv(self,n):
        return self.h.rolling(window=n).max()
    
    def llv(self,n):
        return self.l.rolling(window=n).min()
    
    def atr(self,n):
        return TechIndicator.tr(self).rolling(window=n).mean()

    def roc(self,n):
        return self.c.pct_change(periods=n)*100


    def splk(self):
        roc_windows = [10,15,20,30,40,65,75,100,195,265,390,530]
        rcma_windows = [10,10,10,15,50,65,75,100,130,130,130,195]
        factor = [1,2,3,4,1,2,3,4,1,2,3,4]
        spl_k = 0
        for i,j,k in zip(roc_windows,rcma_windows,factor):
            spl_k +=(TechIndicator.roc(self,i).rolling(window=j).mean())*k
        return spl_k.dropna()
    
    def willr(self,n):
        return ((TechIndicator.hhv(self,n)-self.c)/(TechIndicator.hhv(self,n)-TechIndicator.llv(self,n)))*(-100)
    
    def rsi(self,n):
        rs = (TechIndicator.gain(self).rolling(window=n).mean())/(TechIndicator.loss(self).rolling(window=n).mean())
        rsi = (rs/(1+rs))*100
        return rsi.dropna()
    
    def rwi(self,n):
        rwih = (self.h- TechIndicator.llv(self,n))/(TechIndicator.atr(self,n)*math.sqrt(n))
        rwil = (TechIndicator.hhv(self,n) - self.l)/(TechIndicator.atr(self,n)*math.sqrt(n))
        return pd.DataFrame([rwih,rwil],index=['rwih','rwil']).T.dropna()
    
    def keltner_chnl(self,n):
        keltner_mid = TechIndicator.ema(self,20)
        keltner_up = keltner_mid + 2*TechIndicator.tr(self).rolling(window=10).mean()
        keltner_down = keltner_mid - 2*TechIndicator.tr(self).rolling(window=10).mean()
        return pd.DataFrame([keltner_down,keltner_mid,keltner_up],
                            index=['keltner_down','keltner_mid','keltner_up']).T.dropna()
    def prings_kst(self):
        rcma1 = TechIndicator.roc(self,10).rolling(window=10).mean()
        rcma2 = TechIndicator.roc(self,10).rolling(window=15).mean()
        rcma3 = TechIndicator.roc(self,10).rolling(window=20).mean()
        rcma4 = TechIndicator.roc(self,15).rolling(window=30).mean()
        kst = rcma1*1 + rcma2 * 2 + rcma3 * 3 + rcma4 * 4
        kst_signal = kst.rolling(window=9).mean()
        return pd.DataFrame([kst,kst_signal],index=['kst','kst_signal']).T.dropna()
    
    def stoch_rsi(self,n):
        max_rsi = TechIndicator.rsi(self,n).rolling(window=n).max()
        min_rsi = TechIndicator.rsi(self,n).rolling(window=n).min()
        return ((TechIndicator.rsi(self,n)- max_rsi)/(max_rsi - min_rsi)).dropna()
    
    def chandelier_exit(self):
        long = TechIndicator.hhv(self,22) - TechIndicator.atr(self,22)*3
        short = TechIndicator.hhv(self,22) + TechIndicator.atr(self,22)*3
        return np.round(pd.DataFrame([long,short],index=['long','short']).T.dropna(),2)
    
    def macd(self):
        macd_line = TechIndicator.ema(self,12) - TechIndicator.ema(self,26)
        signal_line = macd_line.ewm(span=9,min_periods=9).mean().dropna()
        macd_hist = macd_line - signal_line
        return pd.DataFrame([macd_line,signal_line,macd_hist],
                            index=['macd_line','signal line','macd_hist']).T
    def pvo(self):
        ema9 = self.v.ewm(span=9,min_periods=9).mean().dropna()
        ema12 = self.v.ewm(span=12,min_periods=12).mean().dropna()
        ema16 = self.v.ewm(span=26,min_periods=26).mean().dropna()
        calc = ((ema12 - ema26)/ema26)*100
        return pd.DataFrame([calc,ema9],index=['pvo','signal line']).T
    
    def mass_index(self,n=9,w=25):
        single_ema = (self.h - self.l).ewm(span=n,min_periods=n).mean()
        double_ema = single_ema.ewm(span=n,min_periods=n).mean()
        mass_index = (single_ema/double_ema).rolling(window=w).sum()
        return mass_index
    
    def adl(self):
        mfm = ((self.c-self.l)-(self.h - self.c))/(self.h-self.l)
        mfv = self.v*mfm
        return mfv.cumsum()
    
    def chaikin_osc(self):
        adl3 = TechIndicator.adl(self).ewm(span=3,min_periods=3).mean()
        adl9 = TechIndicator.adl(self).ewm(span=9,min_periods=9).mean()
        return adl3-adl9
    def cmf(self):
        mfm = ((self.c-self.l)-(self.h - self.c))/(self.h-self.l)
        mfv = self.v*mfm
        cmf = mfv.rolling(window=20).sum()/self.v.rolling(window=20).sum()
        return cmf
    
    def coppock_curve(self):
        return TechIndicator.roc(self,14).ewm(span=10,min_periods=10).mean() + TechIndicator.roc(self,10)

    def cci(self):
        xbar = TechIndicator.pivot(self).rolling(window=20).mean().dropna()
        x = TechIndicator.pivot(self) - xbar
        md = np.mean(np.abs(x-xbar))
        return x-xbar/(0.015*md)
        
    def eom(self,n=14):
        rng = (self.h.diff(periods=1)+self.h.diff(periods=1))/2
        br = (self.v/1e8)/(self.h-self.l)
        emv = rng/br
        return emv.rolling(window=n).mean()
    
    def force_index(self,n=13):
        fi = TechIndicator.change(self)*self.v
        return fi.rolling(window=n).mean()
    
    def ppo(self):
        ppo_ = (TechIndicator.ema(self,12)-TechIndicator.ema(self,26))/(TechIndicator.ema(self,26))*100
        sig_line = ppo_.rolling(window=9).mean()
        ppo_hist = ppo_ - sig_line
        return pd.DataFrame([ppo_,sig_line,ppo_hist],index=['ppo','signal_line','ppo_hist']).T.dropna()
    
    def ichimoku_cloud(self):
        #conversion_line
        tenkan_sen = (TechIndicator.hhv(self,9) + TechIndicator.llv(self,9))/2
        #base_line
        kijun_sen = (TechIndicator.hhv(self,26) + TechIndicator.llv(self,26))/2
        #Leading_spanA
        senoku_spanA = (tenkan_sen + kijun_sen)/2
        #Leading_spanB
        senoku_spanB = (TechIndicator.hhv(self,52) + TechIndicator.llv(self,52))/2
        #Lagging_span
        chikou_span=self.c
        return pd.DataFrame([tenkan_sen,kijun_sen,senoku_spanA,senoku_spanB],
                            index=['tenkan_sen','kijun_sen','senoku_spanA','senoku_spanB']).T
    
    def ulcer_index(self):
        pct_drawdown = ((self.c - self.c.rolling(window=14).max())/self.c.rolling(window=14).max())*100
        squared_drawdown = np.square(pct_drawdown)
        squared_avg = squared_drawdown.rolling(window=14).mean()
        ui = np.sqrt(squared_avg)
        return ui
    
    def ultimate_osc(self):
        #buying_prsr = self.c - 
        pass
    def tsi(self):
        pass
    def aroon_osc():
        pass
    

class CandlePattern(TechIndicator):

    def __init__(self):
        pass