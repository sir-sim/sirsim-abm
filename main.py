#!/usr/bin/python

import numpy as np
import pandas as pd

class Model:
	def __init__(self, time_steps, theta, model_name):
		# model_name should be 'scientists', 'lobbyists'
		
		if (model_name == 'scientists'):
			# print('scientists run')
			self.weights    = np.array([[0.0,0.4,0.0,0.0,0.0,0.4, 0.2,0.0,0.0,0.0], 
									    [0.4,0.0,0.4,0.0,0.0,0.0, 0.2,0.0,0.0,0.0],
									    [0.0,0.4,0.0,0.4,0.0,0.0, 0.2,0.0,0.0,0.0],
									    [0.0,0.0,0.4,0.0,0.4,0.0, 0.2,0.0,0.0,0.0],
									    [0.0,0.0,0.0,0.4,0.0,0.4, 0.2,0.0,0.0,0.0],
									    [0.4,0.0,0.0,0.0,0.4,0.0, 0.2,0.0,0.0,0.0],

									    [0.0,0.0,0.0,0.0,0.0,0.0,0.0, .33,.33,.33],
									    [0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.5,0.5],
									    [0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.5,0.0,0.5],
									    [0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.5,0.5,0.0]
									    ])
			self.opinions 	= np.array([[0.0,0.0,0.0,0.0,0.0,0.0, 0.0, 1.0,1.0,1.0], 
									    [0.0,0.0,0.0,0.0,0.0,0.0, 0.0, 0.5,0.5,0.0]]) # Val, Prob
			self.no_agents  = len(self.weights[1,:])
			if (theta < 0 or theta > 1.0):
				raise ValueError("Theta must be between 0 and 1 inclusive")	
			else:
				self.theta		= theta*np.ones((1,self.no_agents)) # Resistance. Weighting of my current opinion vs new information 

		elif (model_name == 'lobbyists'):
			# print('lobbyists run')
			self.weights    = np.array([[0.0,0.8, 0.0,0.0,0.0,0.4, 0.2,0.0,0.0,0.0], 
									    [0.8,0.0, 0.4,0.0,0.0,0.0, 0.2,0.0,0.0,0.0],
									    
									    [0.0,0.4,0.0,0.4,0.0,0.0, 0.2,0.0,0.0,0.0],
									    [0.0,0.0,0.4,0.0,0.4,0.0, 0.2,0.0,0.0,0.0],
									    [0.0,0.0,0.0,0.4,0.0,0.4, 0.2,0.0,0.0,0.0],
									    [0.4,0.0,0.0,0.0,0.4,0.0, 0.2,0.0,0.0,0.0],

									    [0.2,0.2,0.0,0.0,0.0,0.0,0.0, 0.2,0.2,0.2],
									    [0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.0,0.5,0.5],
									    [0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.5,0.0,0.5],
									    [0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.5,0.5,0.0]
									    ])
			self.opinions   = 	np.array([[-3.,-3., 0.0,0.0,0.0,0.0,   0.0,  1.0,1.0,1.0],
										  [0.2,0.2, 0.0,0.0,0.0,0.0,   0.0,  0.5,0.5,0.0]])
			self.no_agents  = len(self.weights[1,:])
			if (theta < 0 or theta > 1.0):
				raise ValueError("Theta must be between 0 and 1 inclusive")	
			else:
				self.theta		= theta*np.ones((1,self.no_agents)) # Resistance. Weighting of my current opinion vs new information 
				self.theta[0,:2] = 1.0
				# print(self.theta)
		else:
			raise ValueError("model_name must be either 'scientists', 'lobbyists'")	

		self.theta_var = theta
		# self.opinions   = np.random.rand(2,self.no_agents)
		# self.weights    = np.random.rand(self.no_agents,self.no_agents)
		
		self.time_steps = time_steps
		self.t 			= 0
		self.alpha		= np.ones((1,self.no_agents)) # Loudness
		self.messages  	= np.zeros((2,self.no_agents))



		self.data       = pd.DataFrame()
		self.agg_df		= pd.DataFrame()	 		
		self.run_data	= {'Time':[],'Agent':[], 'Alpha':[], 'Theta':[], 'Theta_Run':[],
						   'Value':[],'Prob':[], 'Value_Msg':[],'Prob_Msg':[], 'Expected_Value':[]}

		self.agg_data = {'Time':[], 'Theta_Run':[],
						 'Total_Value':[],'Total_Prob':[],'Total_Expected_Value':[]}

	def store_data(self):
		temp_val 		= 0.0
		temp_prob 		= 0.0
		temp_exp_val   	= 0.0
		# TODO: Theta is used to sperate plot runs, but also changed, for lobbiests. 
		for i in range(self.no_agents):
			self.run_data['Time'].append(self.t)
			self.run_data['Agent'].append(i)
			self.run_data['Alpha'].append(self.alpha[0,i])
			self.run_data['Theta'].append(self.theta[0,i])
			self.run_data['Theta_Run'].append(self.theta_var)
			self.run_data['Value'].append(self.opinions[0,i])
			self.run_data['Prob'].append(self.opinions[1,i])
			self.run_data['Value_Msg'].append(self.messages[0,i])
			self.run_data['Prob_Msg'].append(self.messages[1,i])
			self.run_data['Expected_Value'].append(self.opinions[0,i] * self.opinions[1,i])
			temp_val 		+= self.run_data['Value'][-1]
			temp_prob		+= self.run_data['Prob'][-1]
			temp_exp_val 	+= self.run_data['Expected_Value'][-1]
			# print("  Agent e Val %f, Val %f, Prob %f, %f" % (self.run_data['Expected_Value'][-1], self.run_data['Value'][-1], self.run_data['Prob'][-1], self.run_data['Value'][-1] * self.run_data['Prob'][-1]))
		self.agg_data['Time'].append(self.t)
		self.agg_data['Theta_Run'].append(self.theta_var)
		self.agg_data['Total_Value'].append(temp_val)
		self.agg_data['Total_Prob'].append(temp_prob)
		self.agg_data['Total_Expected_Value'].append(temp_exp_val)
		# print("Expected Val %f, Val %f, Prob %f, %f" % (temp_exp_val,temp_val,temp_prob,temp_val*temp_prob))
	

	def generate_messages(self):
		# G(alpha * opinion + bias) assume G, alpha 1, bias 0 initially
		self.messages = self.opinions

	def update_opinions(self):
		o = self.opinions
		w = self.weights
		theta = self.theta
		o = theta*o + (1-theta)*np.array((np.dot(w,o[0,:]),np.dot(w,o[1,:])))
		self.opinions = o

	def update_weights(self):	
		m_val 		= self.messages[0,:]
		m_prob 		= self.messages[1,:]
		o_val 		= self.opinions[0,:]
		o_prob 		= self.opinions[1,:]
		w 			= self.weights
		m_min_val   = np.min(self.messages[0,:])
		m_max_val   = np.max(self.messages[0,:])
		m_min_prob  = np.min(self.messages[1,:])
		m_max_prob  = np.max(self.messages[1,:])
		
		# Note if messages are more complicated, we will need to perform a 
		# 'decode' step first, so messages are comparable with opinions
		val_adjustment  = 0.5 * abs(o_val  - m_val)  / (m_max_val  - m_min_val)
		prob_adjustment = 0.5 * abs(o_prob - m_prob) / (m_max_prob - m_min_prob)
		w = w * (1.0 - val_adjustment - prob_adjustment)

		# Rescale so new weights are between 0 and 1
		for row in range(len(w[0,:])):
			row_sum  = np.sum(w[row,:])
			w[row,:] = w[row,:]/row_sum
		self.weights = w

	def step(self):
		self.store_data()
		self.generate_messages()
		self.update_opinions()
		self.update_weights()
		self.t += 1

	def run(self):
		for time in range(self.time_steps):
			self.step()
		self.data   = pd.DataFrame(self.run_data)
		self.agg_df = pd.DataFrame(self.agg_data)

if __name__ == '__main__':
	print('Running model')
	time_steps = 2
	agents     = 2
	datasets   = []

	for theta in ([0, .5, .8, .9, 1.0, 1]):
		m = Model(time_steps, theta, 'lobbyists')
		m.run()
		datasets.append(m.data)
		m2 = Model(time_steps, theta, 'scientists')

	merged_data = pd.concat(datasets)
	# print(merged_data)

	# run_data = pd.DataFrame(rows_list, columns=['C1', 'C2','C3']) 	

	# 'time' 'agent' 'val' 'prob' 'val_msg' 'prob_msg'

	# Quick sanity tests	
	# a = 1*np.ones((2,3,3))  #m
	# e = 3*np.ones((2,3,3))	#o
	# w = 2*np.ones((3,3))	#w
	# # Step 2 updating opinions
	# print(e-a)
	# # Step 3 updating weights 
	# print(w*e) 

	# e = np.random.rand(2,1,1)	#o
	# print(e)
	# print(np.min(e[0,:,:]))

	# w = np.random.rand(2,2)
	# print(w)
	# print('Check')

	# for row in range(len(w[0,:])):
	# 	#print(w[row,:])
	# 	row_sum  = np.sum(w[row,:])
	# 	w[row,:] = w[row,:]/2
	# 	print(w[row,:])


	# Q: sum on axis 0 or 1?
	# w = 1*np.ones((6,4))
	# sr 	= w.sum(1)
	# sc 	= w.sum(0)
	# print(sr)
	# print(sc)


	# Selection examples
	# #w[0,:]*2
	# w[:3,:2] +=1
	# w[5:,:2] +=1
	# w[1:2,:2] 
	# 1. matrix mult
	# 2. plot


	# d = {'x': [1, 1 , 2 , 2], 'y': [3, 4 , 3, 4], 'z':[34,2, 10, 4]}
	# df = pd.DataFrame(data=d)
	# print(df)


	# test = np.dot(m.weights,m.opinions[0,:])

	# w = np.array([[1, 2, 4],
	# 	 		  [3, 6, 9],
	# 	 		  [2, 2, 2]])

	# o = np.ones([2,3])
	# print(o)
	# print(w)
	# print(o[0,:])
	# vals  = np.dot(w,o[0,:])
	# probs = np.dot(w,o[1,:])
	# print(vals)
	

	# final = np.array((vals,probs))
	# #np.concatenate((vals.T,probs.T))
	# print(final)

	# ff = np.array((np.dot(w,o[0,:]),np.dot(w,o[1,:])))
	# print(ff)
    
