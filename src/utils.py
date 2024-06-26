import numpy as np
from data.font import _font_1
import random


def check_type(type, array, str):
   if type not in array:
      raise ValueError(f"Valor de '{str}' invalido")
   
   return type

def check_positivity(num, str):
   if not type(num) == int or num < 0:
      raise ValueError(f"Valor de '{str}' invalido")
   
   return num

def check_arr(arr, str):
   for i in range(len(arr)):
      check_positivity(arr[i], str)
   
   return arr

def check_prob(num, str):
   if not type(num) == float or num < 0 or num > 1:
      raise ValueError(f"Valor de '{str}' invalido")
   
   return num

def check_num(num, str):
   if not type(num) == float:
      raise ValueError(f"Valor de '{str}' invalido")
   
   return num

class DataConfig:

    def __init__(self, data, fonts=_font_1):
      self.input_data = extract_patterns(fonts)
      self.bias = check_positivity(data['bias'], "bias")

     
      # Layer params
      self.output_activation = check_type(data['output_activation'], data['activation_options'], "funcion de activacion de capa de salida")
      self.hidden_activation = check_type(data['hidden_activation'], data['activation_options'], "funcion de activacion de capas ocultas")
      self.beta = check_num(data['beta'], "beta")

      self.qty_hidden_layers = check_positivity(data['qty_hidden_layers'], "cantidad de capas ocultas")
      self.qty_nodes_in_hidden_layers = check_arr(data['qty_nodes_in_hidden_layers'], "cantidad de nodos en capas ocultas")

      if(self.qty_hidden_layers != len(self.qty_nodes_in_hidden_layers)):
         raise ValueError("qty_hidden_layers y qty_nodes_in_hidden_layers no se corresponden entre si")
      
      self.latent_space_size = check_positivity(data['latent_space_size'], "tamaño de espacio latente")

      # Training params
      self.learning_rate = check_prob(data['learning_rate'], "tasa de aprendizaje")
      self.epochs = check_positivity(data['epochs'], "epocas")
      self.training_percentage = check_prob(data['training_percentage'], "porcentaje de entrenamiento")
      self.min_error = check_prob(data['min_error'], "cota de error")


      # Optimizer values
      self.optimizer_method = check_type(data['optimizer_method'], data['optimizer_options'], "metodo de optimizacion")
      self.alpha = check_prob(data['alpha'], "alpha")
      self.beta1 = check_prob(data['beta1'], "beta 1")
      self.beta2 = check_prob(data['beta2'], "beta 2")
      self.epsilon = check_prob(data['epsilon'], "epsilon")

def extract_patterns(font):
    patterns = []
    for pattern in font:
        matrix = []
        for byte in pattern:
            binary = format(byte, '08b')  # Pasa a byte a una cadena binaria de 8 bits
            row = [int(bit) for bit in binary[-5:]]  # últimos 5 bits de la cadena binaria
            matrix.extend(row)
        patterns.append(matrix)
    return np.array(patterns)


def alter_data(data, prob, n):
    rng = np.random.default_rng()
    mutated_data = []
    mutated_indices = random.sample(range(len(data)), n)  # Indices aleatorios para mutacion
    for i, elem in enumerate(data):
        mutated_elem = []
        if i in mutated_indices:
            for value in elem:
                if rng.random() < prob:
                    noise = rng.uniform(0, 0.5)
                    if value == 0:
                        mutated_elem.append(float(value) + noise)
                    else:
                        mutated_elem.append(float(value) - noise)
                else:
                    mutated_elem.append(float(value))
        else:
            mutated_elem = elem
        mutated_data.append(mutated_elem)
    return mutated_data