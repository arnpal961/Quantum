'''
Quantum
A wrapper for kiteconnect
Algorithmic trading module for NSE-India via pykiteconnect & zerodha
'''
#Importing necessary libraries
import webbrowser as wb
import pandas as pd
from kiteconnect import KiteConnect,WebSocket

#api_key,api_secret,user_id given by zerodha
#store it in the working directory in a file called credential 
api_key,api_secret,user_id = open('credential','r+').read().strip('\n').split(',')
kite = KiteConnect(api_key=api_key)
login_url = kite.login_url()
wb.open(login_url) 




#Get the request token after login from webbrowser and put the token here
request_token = ''
auth_data = kite.request_access_token(request_token,secret=api_secret)
access_token = auth_data["access_token"]
public_token = auth_data["public_token"]
kite.set_access_token(access_token)
with open('auth_data.txt','r+') as f:
    f.write(access_token+','+public_token)    


#Retrieve instruments token
instrument = kite.instruments()
instrument = pd.DataFrame(instrument)


#Saving to local file
instrument.to_csv('instrument.csv')
