# Computação Evolucionária

import random
import copy
import numpy as np
import matplotlib.pyplot as plt
import csv
#import cv2
import math
from skimage import io
import helper
import os
import gc
import keras
from PIL import Image
from differential_evolution_multiple import evaluate
from helper import perturb_image_mult_pixel
from local_search import local_search
import builtins
import torch


def predict_classes(xs, img, target_class, model, minimize=True):
    # Perturb the image with the given pixel(s) x and get the prediction of the model
    imgs_perturbed = perturb_image_mult_pixel(xs, img)
    predictions = model.predict(imgs_perturbed)[:,target_class]
    # This function should always be minimized, so return its complement if needed
    return predictions if minimize else 1 - predictions

def generate_random_individual(w, h, number_pixels): # um pixel modificado
  #image = img
  #print(i + 1)
  genotype = []
  for i in range(number_pixels):
    x = random.randint(0, w - 1)
    y = random.randint(0, h - 1)
    red = random.randint(0, 255)
    green = random.randint(0, 255)
    blue = random.randint(0, 255)
    pixel = np.array([x, y, red, green, blue]) # pixel = x,y,r,g,b
    genotype.append(pixel)
  return {'genotype': genotype, 'fitness': None, 'confidence': None, 'success': None}

def generate_initial_population(POPULATION_SIZE, w, h, number_pixels):
    for i in range(POPULATION_SIZE):
        yield generate_random_individual(w, h, number_pixels)

def mapping(genotype, image): # genotype = [np.array([x,y,r,g,b]), ...]
  image = perturb_image_mult_pixel(np.array(genotype), image)
  helper.plot_image(image)


def choose_indiv(population, TOURNAMENT):     # cópia do 1º projeto - inalterado
    pool = random.sample(population, TOURNAMENT)  # escolher aleatoriamente TOURNAMENT pixeis
    pool.sort(key=lambda i: i['fitness'])         # organizar de acordo com o fitness
    return copy.deepcopy(pool[-1])

########################################### CROSSOVER FUNCTIONS ######################################

# CROSSOVER ONE POINT
def crossover_one_point(p1, p2, number_pixels):
  genotype = []
  max_point = (5 * number_pixels) - 1

  p1_flat = np.array(p1['genotype']).flatten()
  p2_flat = np.array(p2['genotype']).flatten()

  cut_point = random.randint(1, max_point)  # IN CASE OF number_pixels = 5:
                                              # | x | y | r | g | b \\ x | y | r | g | b \\ x | y | r | g | b \\ x | y | r | g | b \\ x | y | r | g | b \\
                                              # 0   1   2   3   4   5    6   7   8   9   10  11  12  13  14   15  16  17  18  19   20  21  22  23  24   25
                                              #         1            |         2          |        3           |         4          |           5       
  for i in range(0, cut_point):
    genotype.append(p1_flat[i])
  for i in range(cut_point, max_point + 1):
    genotype.append(p2_flat[i])

  genotype = np.split(np.array(genotype), len(genotype) // 5)   # Split into 'number_pixels' again instead of flatten

  return {'genotype': genotype, 'fitness': None, 'confidence': None, 'success': None}

# CROSSOVER TWO POINT
def crossover_two_point(p1, p2, number_pixels):
    genotype = []
    max_point = (5 * number_pixels) - 1

    p1_flat = np.array(p1['genotype']).flatten()
    p2_flat = np.array(p2['genotype']).flatten()

    # TWO CUT POINTS
    cut_point1 = random.randint(1, max_point - 1)
    cut_point2 = random.randint(cut_point1 + 1, max_point)

    # TWO POINT CROSSOVER: chooses 1 segment at a time
    for i in range(0, cut_point1):
        genotype.append(p1_flat[i])
    for i in range(cut_point1, cut_point2):
        genotype.append(p2_flat[i])
    for i in range(cut_point2, max_point + 1):
        genotype.append(p1_flat[i])

    genotype = np.split(np.array(genotype), len(genotype) // 5) # Split into 'number_pixels' again instead of flatten

    return {'genotype': genotype, 'fitness': None, 'confidence': None, 'success': None}

# UNIFORME
# def crossover(PROB_CROSSOVER, TOURNAMENT, PIXELS):
#   # Parent Selection
#   p1 = choose_indiv(population, TOURNAMENT)
#   p2 = choose_indiv(population, TOURNAMENT)
#   # nao fazer crossover quando p1 e p2 sao iguais
#   while(np.array_equal(p2['genotype'][i], p1['genotype'][i])):
#     p2 = choose_indiv(population, TOURNAMENT)
  
#   for i in range(PIXELS):
#     if random.random() < PROB_CROSSOVER:    # random.random() -> probability:[0.0 ; 1.0[
#         # Recombination
#         ni = crossover(p1[i], p2[i])
#         #evaluate(ni, image, true_class, model)
#     else:
#         ni = choose_indiv(population, TOURNAMENT)[i]
  
#   return ni

def crossover_uniform(p1, p2, PIXELS):
  ni = []
  for i in range(PIXELS):
    pixel = []
    for j in range(5):
        if random.random() < 0.5:
            pixel.append(p1['genotype'][i][j])
        else:
            pixel.append(p2['genotype'][i][j])
    ni.append(pixel)

  return {'genotype': ni, 'fitness': None, 'confidence': None, 'success': None}

def crossover_block(p1, p2, PIXELS):
    ni = []
    for i in range(PIXELS):
        # CHOOSE WHICH (x,y) - p1 or p2
        if random.random() < 0.5:
            x_y = p1['genotype'][i][:2]
        else:
            x_y = p2['genotype'][i][:2]

        # CHOOSE HICH COLOR (r,g,b) - p1 or p2
        if random.random() < 0.5:
            r_g_b = p1['genotype'][i][2:]
        else:
            r_g_b = p2['genotype'][i][2:]

        # CONCATENATE
        ni.append(list(x_y) + list(r_g_b))

    return {'genotype': ni, 'fitness': None, 'confidence': None, 'success': None}

#####################################################################################################

# Funções de mutação para cada gene
def mutate_por_gene(p, w, h, PROB_MUTATION):
  p = copy.deepcopy(p)
  p['fitness'] = None

  for i in range(5):
    if random.random() > PROB_MUTATION:
      if i == 0:        # posiçao 0 -> x
        p['genotype'][0] = random.randint(0, w - 1)
      elif i == 1:      # posiçao 1 -> y
        p['genotype'][1] = random.randint(0, h - 1)
      else:                 # valores de red, green e blue
        p['genotype'][i] = random.randint(0, 255)

  return p

def mutate_por_gene_gauss(p, desvio, w, h, PROB_MUTATION, number_pixels):
  p = copy.deepcopy(p)
  p['fitness'] = None

  genotype = np.array(p['genotype']).flatten()

  for i in range(number_pixels * 5):    # 5 because 5 elements in [x, y, r, g, b]
    x1 = random.random()
    x2 = random.random()
    y1 = math.sqrt(-2.0 * math.log(x1)) * math.cos(2.0 * math.pi * x2)
    gene = int(y1 * desvio + genotype[i])

    if random.random() < PROB_MUTATION:
        j = i % 5
        if j == 0:  # x
            gene = min(max(gene, 0), w - 1)
        elif j == 1:  # y
            gene = min(max(gene, 0), h - 1)
        else:  # r,g,b
            gene = min(max(gene, 0), 255)
        genotype[i] = gene

  p['genotype'] = np.split(genotype, len(genotype) // 5)

  return p

# u ficheiro por populaçao
def infos_populacao_fich_v2(populacao, it):
    with open("/content/drive/MyDrive/UNI/Bolsa_dados/Populaçoes/populacoes_individuos_" + str(it), "w") as f:
      writer = csv.writer(f)
      writer.writerow(['index', 'fitness', 'confidence', 'success', 'genotype:'])
      for i in range(len(populacao)):
        writer.writerow([i, populacao[i]['fitness'], populacao[i]['confidence'], populacao[i]['success'], populacao[i]['genotype']])
      f.close()

def dicio_trues_add(dicio, gene, soma_dif, image_orig, suc, suc_act, number_pixels):
  x = gene['genotype']
  novo = 0 
  
  dif = 0
  for i in range(number_pixels):
    x1 = x[i]
    dif += (abs(x1[2] - image_orig[x1[0]][x1[1]][0]) + abs(x1[3] - image_orig[x1[0]][x1[1]][1]) + abs(x1[4] - image_orig[x1[0]][x1[1]][2])) / 3
  
  string = ''
  for i in x:
      string += (str(i) + '_')

  r = dicio.get(string)
  if r == None:
    dicio[string] = dif
    suc += 1
    suc_act += np.max(gene['confidence'])
    novo = 1

  return dicio, soma_dif, suc, suc_act, novo

def genetic_algorithm(image, true_class, model, POPULATION_SIZE, NUMBER_OF_ITERATIONS, PROB_MUTATION, PROB_CROSSOVER, TOURNAMENT, ELITISM, PIXELS, LOCAL_SEARCH_IT, bounds, folder_path, SEED, crossover_func=None, verbose=True, device=None):
    log = print if verbose else (lambda *a, **k: None)
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    random.seed(SEED)
    dicio_total_pixels = {}
    # Boundaries
    w = bounds[0][1]
    h = bounds[1][1]

    # Crossover Default Function
    if crossover_func is None or crossover_func == "flatten_one_point":
        # Se não passar, escolher um padrão, ex: crossover_one_point
        crossover_func = crossover_one_point
    elif crossover_func == "flatten_two_point":
        crossover_func = crossover_two_point
    elif crossover_func == "uniform":
        crossover_func = crossover_uniform
    elif crossover_func == "block":
        crossover_func = crossover_block

    # Count success
    suc = 0
    suc_act = 0
    
    if folder_path != None:
      # File to storage success
      header = ['gen', 'genotype', 'true label', 'predicted label',' confidence in wrong label']
      file_suc = f'{folder_path}/success_file.csv'
      f_suc = open(file_suc, 'w')
      writer_suc = csv.writer(f_suc)
      writer_suc.writerow(header)

      # File for evolution overview
      header = ['gen', 'best fitness', 'best individual', 'best confidence', 'best success', 'true label', 'predicted label', 'average fitness', 'std fitness', 'prediction']
      file_gen = f'{folder_path}/evolution_overview.csv'
      f_gen_info = open(file_gen, 'w')
      writer_gen_info = csv.writer(f_gen_info)
      writer_gen_info.writerow(header)

      # File for Local Search
      file_local = f"{folder_path}/new_best_by_local_search.csv"
      f_ls = open(file_local, 'w')
      writer_ls = csv.writer(f_ls)
      writer_ls.writerow([
          "img_id", "true_label",
          "genotype_before", "genotype_after",
          "fitness_before", "fitness_after",
          "success_before", "success_after",
          "confidence_before", "confidence_after",
          "dif", "it_final_best", "it_first_best"
      ])

      # This folder holds generation files that have all individuals 
      gen_folder = f'{folder_path}/generations_files'
      if not os.path.exists(gen_folder):
          os.makedirs(gen_folder)

    # Storage
    best_fit = []
    avg_fit = []
    x = []
    x.extend(range(0, NUMBER_OF_ITERATIONS)) # array com os primeiros N inteiros
    
    # Create a initial population randomly
    population = list(generate_initial_population(POPULATION_SIZE, w, h, PIXELS))
    dicio_trues = dict()
    it = 0
    soma_dif = 0
    first_success_it = None

    # Evaluate how good the individuals are (problem dependent)
    for it in range(it, NUMBER_OF_ITERATIONS):    
        # o filtra_por novos deve ser um funcao que filtra os elementos da populacao que nao foram ainda vistos.
        # evaluate(filtra_por_novos(population), image, true_class, model)
        evaluate(population, image, true_class, model, dicio_total_pixels, PIXELS, device=device)

        # population.sort(key=lambda x: x['fitness'])
        # best = population[-ELITISM:]
        # best_fit.append(best['fitness'])

        # 1) população atual já avaliada
        population.sort(key=lambda x: x['fitness'])

        # 2) melhor indivíduo da geração
        best = population[-1]
        best_fit.append(best['fitness'])

        # 3) elitismo
        elite_individuals = population[-ELITISM:]
        new_population = copy.deepcopy(elite_individuals)

        ## avaliar se é adversarial depois da avaliaçao
        for ni in population:
          if ni['success'] == True:
            if first_success_it is None:
              first_success_it = it
            dicio_trues, soma_dif, suc, suc_act, novo = dicio_trues_add(dicio_trues, ni, soma_dif, image, suc, suc_act, PIXELS)
            if novo:
              predicted_label = np.argmax(ni['confidence'])
              activation = np.max(ni['confidence'])
              if folder_path != None:
                writer_suc.writerow([it, ni['genotype'], true_class, predicted_label, activation])

        # Colocar o best e a média nesta iteração

        #bests.append(best)
        log("Best at", it, best)

        # Write for overview
        predicted_label = np.argmax(best['confidence'])
        activation = np.max(best['confidence'])

        avg = sum([ind['fitness'] for ind in population])/POPULATION_SIZE
        avg_fit.append(avg)

        # informaçao desta geraçao
        if folder_path != None:
          writer_gen_info.writerow([it, best['fitness'], best['genotype'], activation, best['success'], true_class, predicted_label, avg_fit[it], np.std([ind['fitness'] for ind in population]), list(best['confidence'])])
        #writer_gen_info.writerow([it, best['fitness'], best['genotype'], activation, best['success'], true_class, predicted_label, avg_fit[it], np.std([ind['fitness'] for ind in population]), best['confidence']])

        # Write entire population 
        # header_pergen = ['genotype', 'fitness', 'success', 'confidence']
        # file_pergen = f'{gen_folder}/gen{it}.csv'
        # f_pergen = open(file_pergen, 'w')
        # writer_pergen = csv.writer(f_pergen)
        # writer_pergen.writerow(header_pergen)
        # for m in range(len(population)):
        #     ind = population[m]
        #     writer_pergen.writerow([ind['genotype'], ind['fitness'], ind['success'], list(ind['confidence'])])
        #     #writer_pergen.writerow([ind['genotype'], ind['fitness'], ind['success'], ind['confidence']])
        # f_pergen.close()

        # elitismo + Local Search
        new_population = [best]
        if LOCAL_SEARCH_IT > 0:
          temp = local_search(image, best, true_class, n_trials=LOCAL_SEARCH_IT, SEED=SEED + it)  # SEED + it -> because if the best ind is the same as last population, with only "SEED" will explore the same pixels as previously
          (best_genotype_ls,
          best_fitness_ls,
          succ_after,
          conf_after,
          dif_value,
          it_final_best,
          it_first_best,
          new_best) = temp

          genotype_before = best['genotype']
          fitness_before = best['fitness']
          success_before = best['success']
          confidence_before = best['confidence']

          if new_best['genotype'] == best['genotype']:
              new_population = [best]
          else:
              # ESCREVER NO FICHEIRO → local search encontrou algo melhor
              if folder_path != None:
                writer_ls.writerow([
                    img_id,
                    true_class,
                    genotype_before,
                    new_best['genotype'],
                    fitness_before,
                    best_fitness_ls,
                    success_before,
                    succ_after,
                    list(confidence_before),
                    list(conf_after),
                    dif_value,
                    it_final_best,
                    it_first_best
                ])

              new_population = [best, new_best]

        # print("New population", new_population)

        #print("Populaçao inicial", population)
        ###### Operadores de variaçao e seleçao de descendentes 
        while len(new_population) < POPULATION_SIZE:
            if random.random() < PROB_CROSSOVER:    # random.random() -> probability:[0.0 ; 1.0[
                # Parent Selection
                p1 = choose_indiv(population, TOURNAMENT)
                p2 = choose_indiv(population, TOURNAMENT)
                # nao fazer crossover quando p1 e p2 sao iguais
                while(np.array_equal(p2['genotype'], p1['genotype'])):
                  p2 = choose_indiv(population, TOURNAMENT)
                # Recombination
                ni = crossover_func(p1, p2, PIXELS)

                #evaluate(ni, image, true_class, model)

            else:
                ni = choose_indiv(population, TOURNAMENT)
            # Mutation
                # mutacao por genes - funçoes mutate_por_gene() e mutate_por_gene_gauss()
            #mutate_por_gene(ni, w, h, PROB_MUTATION)
            ni = mutate_por_gene_gauss(ni, 3, w, h, PROB_MUTATION, PIXELS)
            #evaluate([ni], image, true_class, model)
            
            new_population.append(copy.deepcopy(ni)) # para garantir
        population = new_population
        
    log("Final: ", best)
    # bestie = perturb_image_mult_pixel(np.array(best['genotype']), image)
    # helper.plot_image(bestie)
    lista_trues = list(dicio_trues.keys())
    # print("Trues: ", lista_trues)
    log("Pixeis encontrados: ", len(lista_trues))    # convem q este valor seja igual ao 'suc' (success - numero de bem sucedidos)

    # diferença media dos pixeis encontrados e o pixel original
    for i in dicio_trues.values():
      soma_dif += i
    if len(lista_trues) != 0:
      media_difs = soma_dif  / len(lista_trues)

    # Close overview and sucess
    if folder_path != None:
      f_gen_info.close()
      f_suc.close()
      f_ls.close()

      perturbed_image = perturb_image_mult_pixel(np.array(best['genotype']), image)
      # Save perturbed image
      perturbed_image = perturb_image_mult_pixel(best['genotype'], image, device=device)
      perturbed_image = (
          perturbed_image
              .squeeze(0)        # remove batch se for 1
              .detach()
              .cpu()
              .numpy()
      )
      perturbed_image = np.clip(perturbed_image, 0, 255).astype(np.uint8)
      perturbed_pil_image = Image.fromarray(perturbed_image)
      scaled_perturbed_pil_image = perturbed_pil_image.resize((320, 320))
      scaled_perturbed_pil_image.save(f'{folder_path}/best_perturbed.png')

      # Save original image
      image_np = (
          image.detach().cpu().numpy()
          if torch.is_tensor(image)
          else image
      )
      image_np = np.clip(image_np, 0, 255).astype(np.uint8)
      original_pil_image = Image.fromarray(image_np)
      scaled_original_pil_image = original_pil_image.resize((320, 320))
      scaled_original_pil_image.save(f'{folder_path}/original_image.png')

    del population
    if folder_path != None:
      del f_gen_info
      del f_suc
    # del f_pergen
    gc.collect()

    return best_fit, avg_fit, best, suc, suc_act, len(dicio_total_pixels), first_success_it