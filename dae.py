import json
from src.utils import DataConfig, alter_data
from src.autoencoder import Autoencoder
from src.plots import *
from data.font import _font_2, symbols2 

def main():

    with open('./config_dae.json', 'r') as f:
        data_config = json.load(f)

    c = DataConfig(data_config, _font_2)

    MUTATE_PROB = 0.1
    CANT_TO_MUTATE = 15
    arr_of_errors = []
    arr_of_epochs = []
    MUTATION=[0.2,0.4,0.6,0.8,1]
    for i, prob in enumerate(MUTATION):
        mutated_data = alter_data(c.input_data, prob, CANT_TO_MUTATE)
        mutated_data = np.array(list(mutated_data))
        print(c.input_data.shape)
        print(mutated_data.shape)

        autoencoder = Autoencoder(mutated_data, c.input_data, c.latent_space_size,
                                0.001, c.bias, 5000, 1,
                                c.min_error, 2, [20,10], 
                                c.output_activation, c.hidden_activation, c.beta,
                                c.optimizer_method, c.alpha, c.beta1, c.beta2,
                                c.epsilon)
        mse_errors, total_epochs = autoencoder.train()
        arr_of_errors.append(mse_errors)
        arr_of_epochs.append(total_epochs)

        plot_letters(mutated_data, f"Conjunto de entrenamiento con {CANT_TO_MUTATE} caracteres mutados")

        predicted = []
        predicted = []
        for x in mutated_data:
            p = autoencoder.predict(x)
            predicted.append(p)
        plot_letters(predicted, "Eliminacion del ruido")


        arr = []
        for j in range(len(mutated_data)):
            value = autoencoder.latent_space(mutated_data[j])
            arr.append(value)
            print("Latent space value: ", value, " for letter in index ", j)

        plot_latent_space(np.array(arr), symbols2)

    colors = ['blue', 'green', 'red', 'orange']

    fig, ax = plt.subplots()  
    for i, arquitectura in enumerate(MUTATION):
        ax.plot(range(arr_of_epochs[i]), arr_of_errors[i],  label=f"Prob {MUTATION[i]}")
    ax.set_title('Evaluacion del error de DAE con distintas probabilidades') 
    ax.set_xlabel('Epocas') 
    ax.set_ylabel('Error (MSE)')  
    ax.legend(loc='best')  
    plt.show() 

if __name__ == "__main__":
    main()