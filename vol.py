'''
plot the vol surface using free delayed quote data from CBOE
'''
# from pandas_datareader.data import Options
from pandas import read_csv
from dateutil.parser import parse
from datetime import datetime
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm
#from implied_vol import BlackScholes
from functools import partial
from scipy import optimize
import numpy as np
from scipy.interpolate import griddata
from scipy.stats import norm

def CND(X):
	return float(norm.cdf(X))
   
def BlackScholes(v,CallPutFlag = 'c',S = 100.,X = 100.,T = 1.,r = 0.01):
# try:
	# print(v,S,X,T,r)
	d1 = (log(S/X)+(r+v*v/2.)*T)/(v*sqrt(T))
	# print((d1))
	# print(CND(d1))
	d2 = d1-v*sqrt(T)
	# print(d2)
	if CallPutFlag=='c':
		return S*CND(d1)-X*exp(-r*T)*CND(d2)
	else:
		return X*exp(-r*T)*CND(-d2)-S*CND(-d1)
# except: 
	# print('exception thrown')
	# return 0
	
def calc_impl_vol(price, right, underlying, strike, time, rf = 0.01, inc = 0.001):
	f = lambda x: BlackScholes(x,CallPutFlag=right,S=underlying,X=strike,T=time,r=rf)-price
	return optimize.brentq(f,0.001,5.0)
	
	
def get_surf(ticker):
	# q = Options(ticker, 'yahoo').get_all_data()
	q=[]
	if ticker == 'SPY':
		q = read_csv('data.csv',sep=',')
	else: q = read_csv('data_tsla.csv',sep=',')
	
	vals = []
	print(q.head())
	for index, row in q.iterrows():
	# if row['Type'] == 'call': # the data are only calls
		underlying = float(row['Underlying'])
		price = (float(row['Ask'])+float(row['Bid']))/2.0
		# price = (float(row['Last Sale']))
		expd = (datetime.strptime(row['Expiry'], '%m/%d/%Y') - datetime.now()).days
		exps = (datetime.strptime(row['Expiry'], '%m/%d/%Y')  - datetime.now()).seconds
		exp = (expd*24.*3600. + exps) / (365.*24.*3600.)
		if exp>0 and price>0:
			try:
				# print(price, 'c', underlying, float(row['Strike']), exp)
				impl = calc_impl_vol(price, 'c', underlying, float(row['Strike']), exp)
				# impl = float(row['IV'])
				# print(exp)
				print(impl)
				vals.append([exp,float(row['Strike']),impl])
			except:
				pass
	vals = array(vals).T
	print(vals[2])
	mesh_plot2(vals[0],vals[1],vals[2])
	# plot3D(vals[0],vals[1],vals[2])
	#if you want to call both plots use this code instead:
	# combine_plots(vals[0],vals[1],vals[2])

	
def make_surf(X,Y,Z):
	XX,YY = meshgrid(linspace(min(X),max(X),230),linspace(min(Y),max(Y),230))
	ZZ = griddata(array([X,Y]).T,array(Z),(XX,YY), method='linear')
	return XX,YY,ZZ
	
# def mesh_plot2(X,Y,Z, fig , ax):
def mesh_plot2(X,Y,Z):
	fig = plt.figure()
	ax = Axes3D(fig, azim = 38, elev = 25)
	XX,YY,ZZ = make_surf(X,Y,Z)
	ax.plot_surface(XX,YY,ZZ)
	ax.contour(XX,YY,ZZ)
	plt.xlabel("expiry")
	plt.ylabel("strike")
	plt.title('Implied Volatility Surface')
	plt.show()
	
def plot3D(X,Y,Z):
	fig = plt.figure()
	ax = Axes3D(fig, azim = -29, elev = 50)
	ax.plot(X,Y,Z,'o')
	plt.xlabel("expiry")
	plt.ylabel("strike")
	plt.show()

def combine_plots(X,Y,Z):
	fig = plt.figure()
	ax = Axes3D(fig, azim = 38, elev = 25)
	mesh_plot2(X,Y,Z,fig,ax)
	plot3D(X,Y,Z,fig,ax)
	plt.title('Implied Volatility Surface')
	plt.show()
	
	
get_surf('SPY')
