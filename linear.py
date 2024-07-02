import json
import copy
from src.utils import DataConfig, alter_data, extract_patterns
from src.autoencoder import Autoencoder
from src.plots import *
from data.font import _font_2,_font_3, symbols2, symbols3, _font_num, symbols_num    


def main(): 
    print("Error para distintas optimizaciones")

    with open('./config_linear.json', 'r') as f:
        data_config = json.load(f)

    c = DataConfig(data_config, _font_2)

    OPTIMIZACIONES = ["ADAM", "MOMENTUM", "NONE"]

    arr_of_errors = []
    arr_of_epochs = []
    for i in range(len(OPTIMIZACIONES)):
        autoencoder = Autoencoder(c.input_data, c.input_data, c.latent_space_size,
                                c.learning_rate, c.bias, c.epochs, 1,
                                c.min_error, c.qty_hidden_layers, c.qty_nodes_in_hidden_layers,
                                c.output_activation, c.hidden_activation, c.beta,
                                OPTIMIZACIONES[i], c.alpha, c.beta1, c.beta2,
                                c.epsilon)
        mse_errors, total_epochs = autoencoder.train()
        arr_of_errors.append(mse_errors)
        arr_of_epochs.append(total_epochs)

    colors = ['blue', 'green', 'red', 'orange']

    fig, ax = plt.subplots()  
    for i, arquitectura in enumerate(OPTIMIZACIONES):
        ax.plot(range(arr_of_epochs[i]), arr_of_errors[i],  label=f"{OPTIMIZACIONES[i]}") 
    ax.set_title('Error para distintas optimizaciones') 
    ax.set_xlabel('Epocas') 
    ax.set_ylabel('Error (MSE)')  
    ax.legend(loc='best' )   
    plt.show()

    print("Error para distintas arquitecturas")    

    with open('./config_linear.json', 'r') as f:
        data_config = json.load(f)

    c = DataConfig(data_config, _font_3)

    ARQUITECTURAS = [[20], [15], [10], [5]]
    ARQUITECTURAS_COMPLETA = ["35x20x2x20x35","35x15x2x15x35", "35x10x2x10x35", "35x5x2x5x35"]

    arr_of_errors = []
    arr_of_epochs = []
    for i in range(len(ARQUITECTURAS)):
        autoencoder = Autoencoder(c.input_data, c.input_data, c.latent_space_size,
                                c.learning_rate, c.bias, c.epochs, 1,
                                c.min_error, 1, ARQUITECTURAS[i],
                                c.output_activation, c.hidden_activation, c.beta,
                                c.optimizer_method, c.alpha, c.beta1, c.beta2,
                                c.epsilon)
        mse_errors, total_epochs = autoencoder.train()
        arr_of_errors.append(mse_errors)
        arr_of_epochs.append(total_epochs)

    fig, ax = plt.subplots()  
    for i, arquitectura in enumerate(ARQUITECTURAS):
        ax.plot(range(arr_of_epochs[i]), arr_of_errors[i],  label=f"{ARQUITECTURAS_COMPLETA[i]}")
    ax.set_title('Error para distintas arquitecturas') 
    ax.set_xlabel('Epocas') 
    ax.set_ylabel('Error (MSE)')  
    ax.legend(loc='best') 
    plt.show() 

    print("1.1 Tomando todo el conjunto de entrenamiento")

    with open('./config_linear.json', 'r') as f:
        data_config = json.load(f)

    c = DataConfig(data_config, _font_3)

    print(c.input_data)

    plot_letters(c.input_data, "Conjunto de entrenamiento")

    autoencoder = Autoencoder(c.input_data, c.input_data, c.latent_space_size,
                                c.learning_rate, c.bias, c.epochs, c.training_percentage,
                                c.min_error, c.qty_hidden_layers, c.qty_nodes_in_hidden_layers, 
                                c.output_activation, c.hidden_activation, c.beta,
                                c.optimizer_method, c.alpha, c.beta1, c.beta2,
                                c.epsilon)
    autoencoder.train()

    predicted = []
    for x in c.input_data:
            p = autoencoder.predict(x)
            predicted.append(p)
    plot_letters(predicted, "Predicted")
        
    list = []
    for i in range(len(c.input_data)):
            value = autoencoder.latent_space(c.input_data[i])
            list.append(value)
            print("Latent space value: ", value, " for letter in index ", i)
    plot_latent_space(np.array(list), symbols3)

    print("1.2 Tomando un subconjunto del conjunto de entrenamiento")

    with open('./config_linear.json', 'r') as f:
        data_config = json.load(f)

    c = DataConfig(data_config, _font_3)

    autoencoder = Autoencoder(c.input_data, c.input_data, c.latent_space_size,
                            c.learning_rate, c.bias, 15000, 0.5,
                            c.min_error, 1, [10],
                            c.output_activation, c.hidden_activation, c.beta,
                            c.optimizer_method, c.alpha, c.beta1, c.beta2,
                            c.epsilon)
    mse_errors, total_epochs = autoencoder.train()

    plot_letters(autoencoder.train_input_data, "Conjunto de entrenamiento")

    # Análisis de dataset original
    predicted = []
    for x in autoencoder.train_input_data:
        p = autoencoder.predict(x)
        predicted.append(p)
    plot_letters(predicted, "Predicted")

    print("Capacidad de generar nuevos caracteres")

    with open('./config_linear.json', 'r') as f:
        data_config = json.load(f)

    c = DataConfig(data_config, _font_3)

    print(c.input_data)

    autoencoder = Autoencoder(c.input_data, c.input_data, c.latent_space_size,
                                c.learning_rate, c.bias, 15000, 0.5,
                                c.min_error, 1, [10], 
                                c.output_activation, c.hidden_activation, c.beta,
                                c.optimizer_method, c.alpha, c.beta1, c.beta2,
                                c.epsilon)
    mse_errors, total_epochs = autoencoder.train()

    plot_letters(autoencoder.train_input_data, "Conjunto de entrenamiento")

    char_to_generate = []
    char_to_print = []
    idxs = np.random.choice(len(c.input_data), size=2, replace=False)

    for idx in idxs:
        char_to_generate.append(c.input_data[idx])

    value1 = autoencoder.latent_space(char_to_generate[0])
    value2 = autoencoder.latent_space(char_to_generate[1])
    punto_1_4 = value1 + 0.25 * (value2 - value1)
    punto_1_2 = value1 + 0.5 * (value2 - value1)
    punto_3_4 = value1 + 0.75 * (value2 - value1)

    list = []

    for i, num in enumerate(c.input_data):
        list.append(autoencoder.latent_space(num))
        char_to_print.append(symbols3[i])

    list.append(punto_1_4)
    list.append(punto_1_2)
    list.append(punto_3_4)

    char_to_print.append("1/4")
    char_to_print.append("1/2")
    char_to_print.append("3/4")

    plot_latent_space(np.array(list), char_to_print)

    p1 = autoencoder.predict_latent_space(punto_1_4)
    p2 = autoencoder.predict_latent_space(punto_1_2)
    p3 = autoencoder.predict_latent_space(punto_3_4)

    char_to_generate.append(p1)
    char_to_generate.append(p2)
    char_to_generate.append(p3)

    plot_letters(char_to_generate, "Espacio latente medio")

if __name__ == "__main__":
    main()