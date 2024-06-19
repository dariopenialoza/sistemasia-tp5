import numpy as np
from src.layer import Layer
import matplotlib.pyplot as plt
import copy

class Autoencoder:
    def __init__(self, input_data, expected_data, latent_space_size, learning_rate, bias, epochs, training_percentage, min_error,
                 qty_hidden_layers, qty_nodes_in_hidden_layers, output_activation, hidden_activation, beta,
                 optimization_method, alpha, beta1, beta2, epsilon):

        # Info del set de entrenamiento 
        self.input_data_len = len(input_data[0])
        self.bias = bias
        self.input_data = input_data
        self.expected_data = expected_data
        self.min, self.max = self.__calculate_min_and_max(self.expected_data)
        self.train_MSE = -1

        # Global para la red neuronal
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.training_percentage = training_percentage

        self.train_input_data = self.train_expected_data = self.test_input_data = self.test_expected_data = None
        if training_percentage < 1:
            self.train_input_data, self.train_expected_data, self.test_input_data, self.test_expected_data = self.__divide_data_by_percentage(self.input_data, self.training_percentage)

        self.min_error = min_error

        # Metodos de activacion
        self.output_activation = output_activation
        self.hidden_activation = hidden_activation
        self.beta = beta

        # Metodos de optimización
        self.optimization_method = optimization_method
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Arquitectura de la red neuronal
        self.qty_hidden_layers = qty_hidden_layers
        self.qty_nodes_in_hidden_layer = qty_nodes_in_hidden_layers
        self.latent_space_size = latent_space_size
        self.layers, self.latent_space_idx = self.__init_layers()  # Inicialización de las capas
    