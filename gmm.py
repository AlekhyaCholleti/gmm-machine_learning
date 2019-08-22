from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

X,Y = make_blobs(n_samples = 500,centers = 3, cluster_std=1.5 ,random_state=20)
plt.scatter(X[:,0],X[:,1],marker = 'o')
plt.show()
#print(X)
#X = np.dot(X,np.random.randn(2,2))
#print(X)
#plt.scatter(X[:,0],X[:,1],marker = 'o',c=Y)
#plt.show()

class GMM:
	def __init__(self,X,no_of_sources,iterations):
		self.X = X
		self.no_of_sources = no_of_sources  #noof clusters
		self.iterations = iterations
		self.mu = None
		self.pi = None
		self.cov = None
		self.XY = None

	def run(self):

		self.reg_cov = 1e-6*np.identity(len(self.X[0])) 
		x,y = np.meshgrid(np.sort(self.X[:,0]),np.sort(self.X[:,1]))
		#print(np.sort(self.X[:,0]))
		#print("x",x) 
		#print(y)

		self.XY = np.array([x.flatten(),y.flatten()]).T 
	
		self.mu = np.random.randint(min(X[:,0]),max(X[:,0]),size=(self.no_of_sources,len(X[0])))

		self.cov = np.zeros((self.no_of_sources,len(self.X[0]), len(self.X[0])))
		for i in range(self.no_of_sources):
			np.fill_diagonal(self.cov[i],5)

		self.pi = np.ones((self.no_of_sources,))/self.no_of_sources
		#print(self.pi)
		log_likelihoods = []

		fig = plt.figure(figsize=(10,10))
		ax0 = fig.add_subplot(111)
		ax0.scatter(self.X[:,0],self.X[:,1])
		ax0.set_title('Initial state')
		for m,c in zip(self.mu,self.cov):
			c += self.reg_cov
			multi_normal = multivariate_normal(mean=m,cov=c)
			ax0.contour(np.sort(self.X[:,0]),np.sort(self.X[:,1]),multi_normal.pdf(self.XY).reshape(len(self.X),len(self.X)),colors='black',alpha=0.3)
			ax0.scatter(m[0],m[1],c='grey',zorder=10,s=100)	

		  
		for i in range(self.iterations):
			#E-step  
			r_ic = np.zeros((len(self.X),self.no_of_sources))
			for m,co,p,r in zip(self.mu,self.cov,self.pi,range(self.no_of_sources)):
				co += self.reg_cov
				r_ic[:,r] = (p*multivariate_normal.pdf(self.X,mean=m,cov=co))/(np.sum([pi_c*multivariate_normal.pdf(self.X,mean=mu_c,cov=cov_c) for pi_c,mu_c,cov_c in zip(self.pi,self.mu,self.cov+self.reg_cov)],axis = 0))
			#print(r_ic.shape)
				
			#M-step
			self.mu = []
			self.cov = []
			self.pi = []
			for c in range(self.no_of_sources):
				m_c = np.sum(r_ic[:,c],axis=0)
				mu_c = (np.sum(self.X*r_ic[:,c].reshape(len(self.X),1),axis=0))/m_c
				#mu_c2 = (1/m_c)*np.sum(self.X*r_ic[:,c].reshape(len(self.X),1),axis=0)
				#print(np.array_equal(mu_c, mu_c2))
				self.mu.append(mu_c)

				cov_c = ((np.dot((r_ic[:,c].reshape(1,len(self.X))*(self.X- mu_c.reshape(1,len(self.X[0]))).transpose()),self.X- mu_c.reshape(1,len(self.X[0]))))/m_c)+self.reg_cov
				#cov_c2 = ((1/m_c)*np.dot((np.array(r_ic[:,c]).reshape(len(self.X),1)*(self.X-mu_c)).T,(self.X-mu_c))) + self.reg_cov
				#print(np.array_equal(cov_c2, cov_c))
				self.cov.append(cov_c)
				#print((self.cov[c]).shape)
	
				pi_c = m_c/(np.sum(r_ic))
				self.pi.append(pi_c)
				
			#log likelihood
			log_likelihoods.append(np.log(np.sum([k*multivariate_normal(self.mu[i],self.cov[j]).pdf(X) for k,i,j in zip(self.pi,range(len(self.mu)),range(len(self.cov)))])))

		fig2 = plt.figure(figsize=(10,10))
		ax1 = fig2.add_subplot(111)
		ax1.set_title('Log-Likelihood')
		ax1.plot(range(0,self.iterations,1), log_likelihoods)
		plt.show()

	#predict the membership of new datapoint
	def predict(self,Y):
		#plot the point onto the fittet gaussians
		fig3 = plt.figure(figsize=(10,10))
		ax2 = fig3.add_subplot(111)
		ax2.scatter(self.X[:,0],self.X[:,1])
		for m,c in zip(self.mu,self.cov):
			multi_normal = multivariate_normal(mean=m,cov=c)
			ax2.contour(np.sort(self.X[:,0]),np.sort(self.X[:,1]),multi_normal.pdf(self.XY).reshape(len(self.X),len(self.X)),colors='black',alpha=0.3)
			ax2.scatter(m[0],m[1],c='grey',zorder=10,s=100)
			ax2.set_title('Final state')
			for y in Y:
				ax2.scatter(y[0],y[1],c='orange',zorder=10,s=100)
		prediction = []        
		for m,c in zip(self.mu,self.cov):  
			#print(c)
			prediction.append(multivariate_normal(mean=m,cov=c).pdf(Y)/np.sum([multivariate_normal(mean=mean,cov=cov).pdf(Y) for mean,cov in zip(self.mu,self.cov)]))
		plt.show()
		print(prediction)	

GMM = GMM(X,3,50)     
GMM.run()
GMM.predict([[0.5,0.5]])
				

					
