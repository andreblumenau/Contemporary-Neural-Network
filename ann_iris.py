import numpy
import pandas
import scipy
import os
from scipy import special
from numpy import random

class ModernArtificialNeuralNetworkModel:
	def __init__(self,neurons_per_layer):

		#	NEURAL NETWORK STRUCTURE		
		self.neurons_per_layer = neurons_per_layer
		self.number_of_layers  = len(neurons_per_layer)

		self.bias 	 	 = [None] * self.number_of_layers
		self.weights 	 = [None] * self.number_of_layers
		self.summations  = [None] * self.number_of_layers		
		self.activations = [None] * self.number_of_layers

		#	HE INITIALIZATION
		for i in range(self.number_of_layers):
			#	INPUT LAYER		
			if i == 0:

				self.bias[i] = numpy.random.randn(1,self.neurons_per_layer[i])* numpy.square(12/(1+self.neurons_per_layer[i])) #0.1
				self.weights[i] = numpy.random.randn(1,self.neurons_per_layer[i])* numpy.square(12/(1+self.neurons_per_layer[i])) #0.1
				continue

			#	HIDDEN AND OUTPUT LAYERS
			self.bias[i] = numpy.random.randn(1,self.neurons_per_layer[i])* numpy.square(12/(self.neurons_per_layer[i-1]+self.neurons_per_layer[i]))#0.1
			self.weights[i] = numpy.random.randn(self.neurons_per_layer[i-1],self.neurons_per_layer[i])* numpy.square(12/(self.neurons_per_layer[i-1]+self.neurons_per_layer[i]))#0.1

class LearningAlgorithm:
	def __init__(self,ann_model,learning_rate,epochs):

		self.ann_model = ann_model	
		self.alpha = learning_rate	
		self.epochs = epochs

		self.beta1 = 0.9
		self.beta2 = 0.999
		self.err   = 10e-8

		self.input_vector 		    = None
		self.weights_first_moment   = [0.0] * self.ann_model.number_of_layers
		self.bias_first_moment	    = [0.0] * self.ann_model.number_of_layers

		self.weights_second_moment  = [0.0] * self.ann_model.number_of_layers
		self.bias_second_moment		= [0.0] * self.ann_model.number_of_layers

		self.corrected_weights_average_gradient = 0
		self.corrected_bias_average_gradient 	= 0

		self.iteration_count = 0
		self.lambda_ 		 = 0.001
		self.elu_alpha    	 = 1
		self.dropout_rate 	 = 0

	#	ACTIVATIONS
	#	SOFTMAX
	def softmax(self,x):
		try:
			x = x - [[numpy.max(z,axis=1)] for z in x]
			numerator = scipy.special.expit(x)
			denominator = [numpy.sum(z,axis=1,keepdims=True)[0]+0.00000001 for z in numerator]
			numerator = numerator.reshape((numerator.shape[0],numerator.shape[2]))
			denominator = numpy.sum(numerator,axis=1,keepdims=True)
			denominator = numpy.array(denominator)
			result = numerator / (denominator)
			result = numpy.array(numpy.split(result,result.shape[0]))
			return result
		except Exception as e:
			raise e

	#	EXPONENTIAL LINEAR UNIT
	def elu(self,x):
		x[x < 0] = self.elu_alpha*(numpy.exp(x[x < 0])-1)
		return x

	def elu_derivative(self,x):
		x[x >= 0] = 1
		x[x < 0] = self.elu_alpha*numpy.exp(x[x < 0])
		return x

	#	LOSSES
	#	CROSS ENTROPY
	def cross_entropy(self,predicted_vector,target_vector):
		batch_size = target_vector.shape[0]
		cost = -1 / batch_size * (target_vector * numpy.log(numpy.clip(predicted_vector, 1e-20, 1.0))).sum()
		return cost

	def get_matthews_coefficient(self,perf_table):
		tp = perf_table[0][0] #true positive
		tn = perf_table[1][1] #true negative
		fp = perf_table[0][1] #false positive
		fn = perf_table[1][0] #false negative    
		# A soma de um valor infinitesimal(0.00000001) no denominador evita a divisão por zero
		M = (tp*tn - (fp*fn))/(numpy.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))+0.00000001))
		return M

	def get_classification_matrix(self,prediction,actual_labels):
		table_t = [[0 for x in range(2)] for y in range(2)]
		len_actual_labels = len(actual_labels)
		for i in range(len_actual_labels):
			a = prediction[i]
			b = actual_labels[i]
			table_t[a][b] = table_t[a][b] + 1
		return table_t

	#	PREDICTION
	def prediction_comprehension(self,output):
		temp = numpy.abs(output - numpy.amax(output,axis=1,keepdims=True))
		temp[temp != 0] = 1
		temp = 1-temp        
		return temp

	def feedforward(self,model_input):

		self.ann_model.model_input = model_input
		activation = None

		model_input  = numpy.split(model_input,model_input.shape[0])		
		
		for i in range(self.ann_model.number_of_layers):
			if i == 0:
				mult = numpy.multiply(model_input,self.ann_model.weights[i])
			else:
				mult = numpy.matmul(model_input, self.ann_model.weights[i])
				
			sum_of_elements = self.ann_model.bias[i] + mult

			if i < self.ann_model.number_of_layers-1:
				activation = self.elu(sum_of_elements)
			else:
				activation = self.softmax(sum_of_elements)

			self.ann_model.summations[i] = sum_of_elements
			self.ann_model.activations[i] = activation

			model_input = activation			

		#FROM TENSOR BACK TO MATRIX
		model_output = self.ann_model.activations[-1]
		model_output = model_output.reshape(model_output.shape[0],model_output.shape[2])			
			
		return model_output		

		#	LEARNING
	def backpropagation(self,predicted_vector,target_vector):

		y_observed_n_dimensions = numpy.split(target_vector,target_vector.shape[0])
		error_derivative	    = self.ann_model.activations[-1]-y_observed_n_dimensions		
		
		b_gradient = None
		w_gradient = None

		first_moment_correction  = (1-numpy.power(self.beta1,self.iteration_count))
		second_moment_correction = (1-numpy.power(self.beta2,self.iteration_count))

		#GRADIENTS
		last_b_gradient = None 
		last_w_gradient	= None
		
		for i in reversed(range(self.ann_model.number_of_layers)):

			#GRADIENTS
			if i == 0:
				#   INPUT LAYER
				last_b_gradient = [numpy.sum(x,axis=1,keepdims=True) for x in last_b_gradient]						
				last_b_gradient = last_b_gradient + numpy.multiply(last_b_gradient,self.elu_derivative(self.ann_model.summations[i]))
				b_gradient = numpy.mean(last_b_gradient,axis=0,keepdims=True)[0]
				
				input_proxy = [numpy.reshape(x,(x.shape[0],1)) for x in self.ann_model.model_input]
				input_proxy = [numpy.transpose(x) for x in input_proxy]

				last_w_gradient = [numpy.sum(numpy.transpose(x),axis=0,keepdims=True) for x in last_w_gradient]
				last_w_gradient = numpy.multiply(input_proxy,last_w_gradient*self.elu_derivative(self.ann_model.summations[i]))
				w_gradient = numpy.mean(last_w_gradient,axis=0,keepdims=True)[0]			
			elif i < self.ann_model.number_of_layers-1:
				last_b_gradient = [numpy.sum(x,axis=1,keepdims=True) for x in last_b_gradient]			
				last_b_gradient = last_b_gradient + numpy.multiply(last_b_gradient,self.elu_derivative(self.ann_model.summations[i]))
				b_gradient = numpy.mean(last_b_gradient,axis=0,keepdims=True)[0]
				
				activation_proxy = [numpy.transpose(x) for x in self.ann_model.activations[i - 1]]
				
				last_w_gradient = [numpy.sum(numpy.transpose(x),axis=0,keepdims=True) for x in last_w_gradient]
				
				last_w_gradient = numpy.matmul(activation_proxy,last_w_gradient*self.elu_derivative(self.ann_model.summations[i]))
				w_gradient = numpy.mean(last_w_gradient,axis=0,keepdims=True)[0]
				
			else:
				last_b_gradient = error_derivative + numpy.multiply(error_derivative,self.ann_model.summations[i])
				b_gradient = numpy.mean(last_b_gradient,axis=0,keepdims=True)[0]			
			
				error_derivative = [ numpy.transpose(x) for x  in error_derivative]				
			
				last_w_gradient = numpy.matmul(error_derivative,self.ann_model.activations[i - 1])
				last_w_gradient = [numpy.transpose(x) for x in last_w_gradient]						
				w_gradient = numpy.mean(last_w_gradient,axis=0,keepdims=True)[0]

			
			if self.iteration_count == 0:
				
				first_moment_correction  	  = self.err
				second_moment_correction 	  = self.err
				self.weights_first_moment[i]  = numpy.zeros(w_gradient.shape)#self.err#w_gradient
				self.bias_first_moment[i]	  = numpy.zeros(b_gradient.shape)#self.err#b_gradient

				self.weights_second_moment[i] = self.err
				self.bias_second_moment[i]	  = self.err
			else:
				self.weights_first_moment[i] = self.beta1*self.weights_first_moment[i] + (1-self.beta1)*w_gradient
				self.bias_first_moment[i]	= self.beta1*self.bias_first_moment[i] + (1-self.beta1)*b_gradient
            
				self.weights_second_moment[i] = self.beta2*self.weights_second_moment[i] + (1-self.beta2)*numpy.square(w_gradient)
				self.bias_second_moment[i]	= self.beta2*self.bias_second_moment[i] + (1-self.beta2)*numpy.square(b_gradient)
			
			w_first_moment_corrected = self.weights_first_moment[i]/(first_moment_correction)
			b_first_moment_corrected = self.bias_first_moment[i]/(first_moment_correction)
            
			w_second_moment_corrected = self.weights_second_moment[i]/(second_moment_correction)
			b_second_moment_corrected = self.bias_second_moment[i]/(second_moment_correction)
			
			self.ann_model.weights[i] = self.ann_model.weights[i] - self.alpha*w_first_moment_corrected/(numpy.sqrt(w_second_moment_corrected)+self.err)
			self.ann_model.bias[i]	  = self.ann_model.bias[i]	  - self.alpha*b_first_moment_corrected/(numpy.sqrt(b_second_moment_corrected)+self.err)	
			
		self.iteration_count +=1						
		return

def main():
	model = ModernArtificialNeuralNetworkModel(
		neurons_per_layer  = [4,10,3]
	)
	
	__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))	
	csv_path = os.path.join(__location__,"iris.csv")	
	print("csv_path = ",csv_path)
	
	dataframe = pandas.read_csv(csv_path,sep=',',encoding="utf-8")	
	target 	  = dataframe.pop("variety")

	virginica  = numpy.zeros((target.shape[0],1),dtype=float)
	setosa 	   = numpy.zeros((target.shape[0],1),dtype=float)
	versicolor = numpy.zeros((target.shape[0],1),dtype=float)

	virginica [target=="Virginica" ] = 1
	setosa    [target=="Setosa"    ] = 1   
	versicolor[target=="Versicolor"] = 1

	target = numpy.stack([virginica,setosa,versicolor],axis=1)
	target = target.reshape(150,3)

	dataframe = (dataframe-dataframe.mean())/dataframe.std()		
	dataframe = dataframe.values
	
	##	MULTICLASS INDEXING
	#   TRAINING	
	learning_rate 	   = 0.001
	epochs 		  	   = 20000
	training_algorithm = LearningAlgorithm(model,learning_rate,epochs)

	output 			 = None
	predicted_vector = None
	perf_matrix 	 = None
	mcc 			 = 0
	hst_mcc 		 = 0
	acc				 = 0
	highest_accuracy = 0

	for i in range(epochs):
		output = training_algorithm.feedforward(dataframe)   
		training_algorithm.backpropagation(output,target)
		predicted_vector = training_algorithm.prediction_comprehension(output)
		compare_matrix = (predicted_vector==target)
		acc= numpy.sum(compare_matrix)/compare_matrix.size
		ce = training_algorithm.cross_entropy(predicted_vector,target)
		prediction = [ int(x) for x in predicted_vector[:,0].tolist()]		
		reality = [ int(x) for x in target[:,0].tolist()]	
		perf_matrix = training_algorithm.get_classification_matrix(prediction,reality)
		mcc = training_algorithm.get_matthews_coefficient(perf_matrix)
		
		if	mcc > hst_mcc:
			highest_accuracy = acc
			hst_mcc = mcc
		
		print("ACC = ",acc,"HST MCC = ",hst_mcc," MCC (Virginica) = ",mcc," CE = ",ce," EPOCH = ",i)
		
	print("(Virginica) HIGHEST MCC = ",hst_mcc)
	print("(Virginica) MCC = ",hst_mcc)	
	print("(Virginica) PERF MATRIX = ",perf_matrix)	

	return model

if __name__ == "__main__":	
	main()