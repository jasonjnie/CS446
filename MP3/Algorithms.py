import sys
import numpy as np

R = 1000

class Perceptron:
### Simple Perceptron: LR=1, Margin=0
### Perceptron with Margin: LR=chosen, Margin = 1
	def __init__(self,LR,Margin,train_data,Num_Feat):
		self.LR = LR
		self.margin = Margin
		self.Num_Inst = len(train_data)
		self.Num_Feat = Num_Feat
		self.w = np.zeros(Num_Feat)
		self.theta = 0
		self.mistake = np.zeros(self.Num_Inst)
		self.num_R = 0		# num_R denotes the number of consecutive examples survived
		self.mistake_converge = 0

	def train(self,train_data,train_label):
		temp_mistake = 0
		for i in range(self.Num_Inst):
			x = train_data[i]
			y = train_label[i]

			temp = y * (np.dot(x,self.w) + self.theta)
			self.mistake[i] = temp_mistake
			if self.margin == 0:	# No margin
				if temp <= self.margin:
					self.mistake[i] += 1
					self.w += self.LR * y * x
					self.theta += self.LR * y
			else:			# Has margin
				if temp < self.margin:
					if temp <= 0:
						self.mistake[i] += 1
					self.w += self.LR * y * x
					self.theta += self.LR * y	
			temp_mistake = self.mistake[i]			

	def predict(self,test_data):
		N = len(test_data)
		est_label = np.zeros(N)
		for i in range(N):
			x = test_data[i]
			temp = np.dot(x,self.w) + self.theta
			est_label[i] = np.sign(temp)
		return est_label

	def Converge(self,train_data,train_label):
		num_iter = 0
		while num_iter < 10:
			num_iter += 1
			for i in range(self.Num_Inst):
				x = train_data[i]
				y = train_label[i]
				temp = y * (np.dot(x,self.w) + self.theta)
				if self.margin == 0:	# No margin
					if temp <= self.margin:
						self.mistake_converge += 1
						self.num_R = 0	
						self.w += self.LR * y * x
						self.theta += self.LR * y
					else:
						self.num_R += 1
				else:			# Has margin
					if temp < self.margin:
						if temp <= 0:	# make mistake
							self.mistake_converge += 1
							self.num_R = 0
						self.w += self.LR * y * x
						self.theta += self.LR * y
					else:
						self.num_R += 1

				if self.num_R >= R:
					return True
		return False	


class Winnow:
### Winnow: Alpha=chosen, Margin=0
### Winnow with Margin: Alpha=chosen, Margin = chosen
	def __init__(self,Alpha,Margin,train_data,Num_Feat):
		self.alpha = Alpha
		self.margin = Margin
		self.Num_Inst = len(train_data)
		self.Num_Feat = Num_Feat
		self.w = np.ones(Num_Feat)
		self.theta = -Num_Feat
		self.mistake = np.zeros(self.Num_Inst)
		self.num_R = 0
		self.mistake_converge = 0

	def train(self,train_data,train_label):
		temp_mistake = 0
		for i in range(self.Num_Inst):
			x = train_data[i]
			y = train_label[i]
			temp = y * (np.dot(x,self.w) + self.theta)
			self.mistake[i] = temp_mistake
			if self.margin == 0:	# No margin
				if temp <= self.margin:
					self.mistake[i] += 1
					self.w *= self.alpha ** (y*x)	# (y*x) to get negative value on ith component to update
			else:			# Has margin
				if temp < self.margin:
					if temp <= 0:
						self.mistake[i] += 1
					self.w *= self.alpha ** (y*x)
			temp_mistake = self.mistake[i]

	def predict(self,test_data):
		N = len(test_data)
		est_label = np.zeros(N)
		for i in range(N):
			x = test_data[i]
			temp = np.dot(x,self.w) + self.theta
			est_label[i] = np.sign(temp)
		return est_label

	def Converge(self,train_data,train_label):
		num_iter = 0
		while num_iter < 10:
			num_iter += 1
			for i in range(self.Num_Inst):
				x = train_data[i]
				y = train_label[i]
				temp = y * (np.dot(x,self.w) + self.theta)
				if self.margin == 0:	# No margin
					if temp <= self.margin:		# make mistake
						self.mistake_converge += 1
						self.num_R = 0	
						self.w *= self.alpha ** (y*x)
					else:
						self.num_R += 1
				else:			# Has margin
					if temp < self.margin:
						if temp <= 0:		# make mistake
							self.mistake_converge += 1
							self.num_R = 0
						self.w *= self.alpha ** (y*x)
					else:
						self.num_R += 1
						
				if self.num_R >= R:
					return True
		return False	


class AdaGrad:
### AdaGrad: LR=chosen
	def __init__(self,LR,train_data,Num_Feat):
		self.LR = LR
		self.Num_Inst = len(train_data)
		self.Num_Feat = Num_Feat
		self.w = np.zeros(Num_Feat)
		self.theta = 0
		self.G = np.zeros(Num_Feat+1)		# w = G[:self.N], theta = G[self.N]
		self.mistake = np.zeros(self.Num_Inst)
		self.num_R = 0
		self.mistake_converge = 0

	def train(self,train_data,train_label):
		temp_mistake = 0
		for i in range(self.Num_Inst):
			x = train_data[i]
			y = train_label[i]
			temp = y * (np.dot(x,self.w) + self.theta)
			self.mistake[i] = temp_mistake
			if temp <= 1:	# update
				if temp <= 0:	# made mistake
					self.mistake[i] += 1
				temp_gt_w = -y * x
				#print "gt_w", temp_gt_w.shape
				temp_gt_theta = -y
				#print "gt_thera", temp_gt_theta
				self.G[:self.Num_Feat] += temp_gt_w ** 2	# w
				self.G[self.Num_Feat] += temp_gt_theta ** 2	# theta	
				#print "G", self.G.shape
				temp_G = self._check_denominator(self.G)
				self.w += self.LR * y * x / np.sqrt(temp_G[:self.Num_Feat])	
				self.theta += self.LR * y / np.sqrt(temp_G[self.Num_Feat])
			temp_mistake = self.mistake[i]

	def predict(self,test_data):
		N = len(test_data)
		est_label = np.zeros(N)
		for i in range(N):
			x = test_data[i]
			temp = np.dot(x,self.w) + self.theta
			est_label[i] = np.sign(temp)
		return est_label

	def _check_denominator(self,temp_array):
		ret_array = np.zeros(len(temp_array))
		for i in range(len(temp_array)):
			if temp_array[i] != 0:
				ret_array[i] = temp_array[i]
			else:
				ret_array[i] = 1
		return ret_array

	def Converge(self,train_data,train_label):
		num_iter = 0
		while num_iter < 10:
			num_iter += 1
			for i in range(self.Num_Inst):
				x = train_data[i]
				y = train_label[i]
				temp = y * (np.dot(x,self.w) + self.theta)
				if temp <= 1:		# update
					if temp <= 0:		# made mistake
						self.mistake_converge += 1
						self.num_R = 0	
						temp_gt_w = -y * x
						temp_gt_theta = -y
						self.G[:self.Num_Feat] += temp_gt_w ** 2	# w
						self.G[self.Num_Feat] += temp_gt_theta ** 2	# theta	
						temp_G = self._check_denominator(self.G)
						self.w += self.LR * y * x / np.sqrt(temp_G[:self.Num_Feat])	
						self.theta += self.LR * y / np.sqrt(temp_G[self.Num_Feat])
				else:
					self.num_R += 1				
						
				if self.num_R >= R:
					return True
		return False	

	def misclf_error(self,train_data,train_label):
		temp_error = 0
		for i in range(self.Num_Inst):
			x = train_data[i]
			y = train_label[i]
			temp = y * (np.dot(x,self.w) + self.theta)
			if temp <= 0:		# made mistake
				temp_error += 1

		return temp_error		

	def hinge_loss(self,train_data,train_label):
		temp_loss = 0
		for i in range(self.Num_Inst):
			x = train_data[i]
			y = train_label[i]
			temp_val = 1 - y * (np.dot(x,self.w)+self.theta)
			temp = np.amax(np.asarray([0,temp_val]))
			#print temp
			temp_loss += temp

		return temp_loss














	

