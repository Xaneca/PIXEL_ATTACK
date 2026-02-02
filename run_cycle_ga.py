# Imports
import pickle
import numpy as np
import pandas as pd
import matplotlib
from keras.datasets import cifar10
from keras import backend as K
import os
import time
import csv
import gc
import matplotlib.pyplot as plt
import json
import argparse

# Custom Networks
from networks.lenet import LeNet
from networks.resnet import ResNet
#from genetic_algorithm import genetic_algorithm
from genetic_algorithm_multiple import genetic_algorithm
from setup_cifar import CIFAR, CIFARModel

# Helper functions
import helper

matplotlib.style.use('ggplot')

# ==========================================
# CONFIGURAÇÃO DE ARGUMENTOS (O QUE FALTAVA)
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument("--run_id", type=int, default=None, help="Se definido, corre apenas esta run. Se vazio, corre todas.")
parser.add_argument("--gpu_id", type=str, default="0", help="ID da GPU")
args = parser.parse_args()

# ==========================================
# CONFIGURAÇÃO DE GPU
# ==========================================
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# IMPORT PYTORCH
import torch        # CUDA_VISIBLE_DEVICES tem de ser antes do import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Usando device:", device)
torch.set_grad_enabled(False)           # poupar memoria + velocidade (nao precisamos de gradientes
torch.backends.cudnn.benchmark = True   # devido ao CIFARModel ser CNN

# ==========================================
# LER JSON E CONFIGURAÇÕES
# ==========================================

# ----- 1. Ler JSON -----
with open("./results/config.json", "r") as f:
    config = json.load(f)

# ----- 2. Carregar parâmetros -----
nruns = config["nruns"]
seeds = config["seeds"]
n_samples = config["n_samples"]
pop_size = config["pop_size"]
generations = config["generations"]
pixels = config["pixels"]
mut = config["mut"]
cr = config["cr"]
elitism = config["elitism"]
tournament = config["tournament"]
LOCAL_SEARCH_IT = config["local_search_it"]

if args.run_id is not None:
    # O Python começa a contar do 0, mas as runs começam do 1
    indice_necessario = args.run_id - 1
    
    if indice_necessario >= len(seeds):
        print(f"ERRO CRÍTICO: Estás a tentar correr a Run {args.run_id}, mas só tens {len(seeds)} seeds no config.json.")
        print("Adiciona mais seeds à lista no JSON.")
        sys.exit(1)

# ----- 3. Carregar modelos reais -----
cifar_ = CIFARModel('models/cifar.keras')
cifar_100 = CIFARModel('models/cifar-distilled-100.keras')

# Mapa entre nome no JSON → objecto Python
model_registry = {
    "cifar_": cifar_,
    "cifar_100": cifar_100
}

# ----- 4. Verificação do tamanho das listas -----
if len(config["models"]) != len(config["model_Names"]):
    print("ERRO: As listas 'models' e 'model_Names' no JSON têm tamanhos diferentes.")
    print(f"models: {len(config['models'])}, model_Names: {len(config['model_Names'])}")
    print("Corrige o 'config.json' antes de correr o programa.")
    sys.exit(1)  # Termina o programa

# ----- 5. Montar a lista de modelos -----
models = [model_registry[name] for name in config["models"]]
modelNames = config["model_Names"]

# ----- 6. Load dataset -----
data = CIFAR()
x_train = data.train_data
x_test = data.test_data
x_train = (x_train + 0.5) * 255
x_test= (x_test + 0.5) * 255

y_train = data.train_labels
y_test = data.test_labels
y_train = np.argmax(y_train, axis=1)
y_train = y_train.reshape(-1, 1)
y_test = np.argmax(y_test, axis=1)
y_test = y_test.reshape(-1, 1)

(h, w, d) = x_test[0].shape
bounds = [[0, w - 1], [0, h - 1], [0, 255], [0, 255], [0, 255]]
bounds = np.array(bounds)

# ----- DEFINIR LISTA DE RUNS A EXECUTAR -----
if args.run_id is not None:
    # MODO PARALELO: A lista é apenas o número passado no argumento
    runs_to_execute = [args.run_id]
    print(f"Modo Paralelo: A executar apenas a Run {args.run_id}")
else:
    # MODO SEQUENCIAL: A lista é de 1 até nruns
    runs_to_execute = range(1, nruns + 1)
    print(f"Modo Sequencial: A executar runs de 1 a {nruns}")

for model, modelName in zip(models, modelNames):

    # Create GA folder
    ga_folder = f'./results/{modelName}/ga'
    if not os.path.exists(ga_folder):
        try:
            os.makedirs(ga_folder)
        except FileExistsError:
            pass # Ignorar erro se outra thread criou a pasta ao mesmo tempo

    # 1. Time File
    time_file = f'{ga_folder}/time.csv'
    if not os.path.exists(time_file):
        with open(time_file, 'w', newline='') as f:
            csv.writer(f).writerow(['model', 'run', 'time'])
    
    # 2. Metrics File
    file_metrics = f'{ga_folder}/metrics.csv'
    if not os.path.exists(file_metrics):
        with open(file_metrics, 'w', newline='') as f:
            csv.writer(f).writerow(['run', 'success rate dataset', 'time', 'success rate (per img)', 'adv prob label (per img)'])

    # 3. Covered Pixels File
    file_cover = f'{ga_folder}/covered_pixels.csv'
    if not os.path.exists(file_cover):
        with open(file_cover, 'w', newline='') as f:
            csv.writer(f).writerow(['run', 'img_counter', 'img_id', 'number of covered pixels'])
    
    # Select images to attack
    images_df = pd.read_csv(f'./results/{modelName}/images_to_attack_idx.csv')
    images_idx = images_df['image id']
    images_labels = images_df['true label']

    # Mean (across runs) success rate (per img)
    dict_success_rate = {}
    dict_adv_prob1 = {}
    dict_adv_prob2 = {}

    success_rate_dataset_store = []
    for run in runs_to_execute:
        suc_samples = 0 # from n samples, how many were successfully attacked at least once?
        samples = []

        success_rate_per_img = []
        adv_prob_per_img = []

        # Create run folder
        print("---------------------------")
        print("Starting run ", run)
        run_folder = f'{ga_folder}/run_{run}'
        if not os.path.exists(run_folder):
            try:
                os.makedirs(run_folder)
            except FileExistsError:
                pass
        
        # File to store best individual per image
        file_bests = f'{run_folder}/best_individuals.csv'
        header_best = ['img_counter','img_id', 'best pixel', 'fitness', 'true label', 'predicted label', 'prior confidence in true label', 'post confidence true label', 'confidence wrong label']
        
        with open(file_bests, 'w', newline='') as f_bests:
            csv.writer(f_bests).writerow(header_best)

        # Start timer
        start_time = time.time()

        # Attack
        for i in range(n_samples):
            print("\n--- Image ", i)
            img_idx = int(images_idx[i])
            samples.append(img_idx)
            # img = x_test[img_idx]
            img = torch.from_numpy(x_test[img_idx]).float().to(device)
            label = images_labels[i]
            
            if img_idx not in dict_success_rate:
                dict_success_rate[img_idx] = []
                dict_adv_prob1[img_idx] = []
                dict_adv_prob2[img_idx] = []

            img_folder = f'{run_folder}/img_{i}'
            if not os.path.exists(img_folder):
                os.makedirs(img_folder)

            best_fit, avg_fit, best_ind, suc, suc_act_total, n_covered_pixels, first_success_it = genetic_algorithm(img, label, model, pop_size, generations, mut, cr, tournament, elitism, pixels, LOCAL_SEARCH_IT, bounds, img_folder, seeds[run-1], device=device)
            fig = plt.figure(num=1, clear=True)
            plt.plot(list(range(generations)), best_fit)
            plt.plot(list(range(generations)), avg_fit)
            plt.xlabel('Geração')
            plt.ylabel('Fitness')
            plt.title('Fitness overtime')
            plt.legend(['best', 'average'])
            fig.savefig(f"{img_folder}/fitness_evolution_{i}")
            fig.clear()
            plt.close(fig)
            
            # Save how many pixels were successful
            file_npixels = f'./results/{modelName}/number_pixels.csv'
            # Usa 'with open' e append ('a') para segurança em paralelo
            with open(file_npixels, 'a', newline='') as f_npixels:
                csv.writer(f_npixels).writerow(['ga', run, img_idx, i, suc])

            # Save number of covered pixels
            file_cover = f'{ga_folder}/covered_pixels.csv'
            with open(file_cover, 'a', newline='') as f_cover:
                csv.writer(f_cover).writerow([run, i, img_idx, n_covered_pixels])

            # Save best individual
            predicted_label = np.argmax(best_ind['confidence'])
            activation = np.max(best_ind['confidence'])
            prior_confidence_true_label = images_df['confidence'].values[label]
            post_confidence_true_label = best_ind['confidence'][label]
            post_confidence_wrong_label = activation if best_ind['success'] else 0

            with open(file_bests, 'a', newline='') as f_bests:
                csv.writer(f_bests).writerow([i, img_idx, best_ind['genotype'], best_ind['fitness'], label, predicted_label, prior_confidence_true_label, post_confidence_true_label, post_confidence_wrong_label])

            # Metrics for img
            # For success rate dataset
            if suc > 0: 
                suc_samples += 1
                adv_prob2 = (suc_act_total / suc)
            else:
                adv_prob2 = 0
            success_rate = suc / (pop_size * generations) # number of successful attacks / number of evaluations
            adv_prob1 = suc_act_total / (pop_size * generations)

            success_rate_per_img.append(success_rate)
            adv_prob_per_img.append([adv_prob1, adv_prob2])

            # Save img metrics
            folder_metrics_img = f'{ga_folder}/metrics_img'
            if not os.path.exists(folder_metrics_img):
                try:
                    os.mkdir(folder_metrics_img)
                except FileExistsError:
                    pass
            
            file_metrics_img = f'{folder_metrics_img}/img_{i}.csv'
            
            # Verifica se existe para meter header
            if not os.path.exists(file_metrics_img):
                with open(file_metrics_img, 'w', newline='') as f_img:
                    csv.writer(f_img).writerow(['run', 'success rate', 'adv probability'])

            with open(file_metrics_img, 'a', newline='') as f_img:
                csv.writer(f_img).writerow([run, success_rate, [adv_prob1, adv_prob2]])

            dict_success_rate[img_idx].append(success_rate)
            dict_adv_prob1[img_idx].append(adv_prob1)
            dict_adv_prob2[img_idx].append(adv_prob2)

        # f_bests.close()

        # Run metrics
        success_rate_dataset = suc_samples / n_samples
        success_rate_dataset_store.append(success_rate_dataset)

        # End timer and write elapsed time into csv
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        with open(time_file, 'a', newline='') as f_time:
            writer_time = csv.writer(f_time)
            writer_time.writerow([modelName, run, elapsed_time])
        
        with open(file_metrics, 'a', newline='') as f_metrics:
            writer_metrics = csv.writer(f_metrics)
            # success_rate_per_img e adv_prob_per_img sao listas
            writer_metrics.writerow([run, success_rate_dataset, elapsed_time, success_rate_per_img, adv_prob_per_img])
        gc.collect()

    if args.run_id is None:
        print("Modo Sequencial detetado: A calcular médias...")
        # Write mean metrics
        file_means = f'./results/{modelName}/metrics_mean.csv'
        with open(file_means, 'a', newline='') as f_means: # <--- WITH OPEN
             # Nota: Se correres o script sequencial várias vezes, isto vai fazer append infinitamente.
             # Talvez fosse melhor 'w' se quiseres resetar, mas mantive 'a' como tinhas.
            mean_suc_rate_dataset = sum(success_rate_dataset_store) / nruns
            csv.writer(f_means).writerow(['ga', mean_suc_rate_dataset])

        # Write parameters
        # Specify the file path for the CSV
        csv_file = f'{ga_folder}/parameters.csv'
        parameter_names = ["nruns", "seeds", "n_samples", "samples", "pop_size", "generations", "pixels", "mut", "cr", "tournament"]
        parameter_values = [nruns, seeds, n_samples, samples, pop_size, generations, pixels, mut, cr, tournament]

        with open(csv_file, mode='w', newline='') as file: # <--- WITH OPEN
            writer = csv.writer(file)
            writer.writerow(parameter_names)
            writer.writerow(parameter_values)
        
        # Write mean metrics for each image
        # dict = {img_idx: [success rate run 1, ..., success rate nruns]}
        file_metrics_mean_img = f'./results/{modelName}/metrics_mean_img.csv'
        with open(file_metrics_mean_img, 'a', newline='') as f_metrics_img: # <--- WITH OPEN
            writer_metrics_img = csv.writer(f_metrics_img)
            counter = 0
            for img_idx, success_rate in dict_success_rate.items():
                mean_success_rate = sum(success_rate) / nruns
                mean_adv_prob1 = sum(dict_adv_prob1[img_idx]) / nruns
                mean_adv_prob2 = sum(dict_adv_prob2[img_idx]) / nruns
                writer_metrics_img.writerow([counter, img_idx, 'ga', mean_success_rate, [mean_adv_prob1, mean_adv_prob2]])
                counter += 1
    else:
        print(f"Run {args.run_id} concluída. O cálculo das médias globais foi ignorado no modo paralelo.")