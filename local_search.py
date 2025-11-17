import sys
import os
from setup_cifar import CIFAR, CIFARModel
from helper import perturb_image_mult_pixel
from differential_evolution_multiple import evaluate, attack_success
import numpy as np
import pandas as pd
import ast
import re

### CHANGE PARAMETERS HERE ### 
nruns = 5
n_samples = 500
n_pixels = 3
results_file = "./results"

# # temporary - img = 0
# img_id = 5027
# genotype = [[ 10,   1, 234, 227, 214], [14, 14, 30, 32, 45], [16, 14, 28, 27, 40]]
# fitness = 0.7142857142857143
# true_label = 0
# predicted_label = 0

# # temporary - img = 1
# img_id = 2757
# genotype = [[27, 18, 48,  0, 87], [ 19,  16,  94, 192, 146], [  8,   4, 118,  48,  72]]
# fitness = 2.006928287257223
#         # 2.007936507936508

# Load dataset
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
# Load models 
cifar_100 = CIFARModel('models/cifar-distilled-100.keras')
models = [cifar_100]
modelNames = ['distilled']
abordagens = ["ga"]

# ind = {'genotype': genotype, 'fitness': 0.7142857142857143, 'confidence': None, 'success': None}

def str_to_array_list(s):
    """
    Converte string do CSV do tipo 
    "[array([10,1,234,...]), array([..]), ...]" 
    para lista de np.array.
    """
    # Encontra todos os conteúdos dentro de array([...])
    matches = re.findall(r'array\(\[([^\]]+)\]\)', s)
    arrays = []
    for m in matches:
        # Converte os números da string em lista de ints
        nums = [int(x) for x in m.split(',')]
        arrays.append(np.array(nums))
    return arrays

def read_file(file_path):
    # Lê o CSV
    df = pd.read_csv(file_path)

    
    df['best pixel'] = df['best pixel'].apply(str_to_array_list)
    
    # Converte colunas numéricas
    df['fitness'] = df['fitness'].astype(float)
    df['img_id'] = df['img_id'].astype(int)
    df['true label'] = df['true label'].astype(int)
    df['predicted label'] = df['predicted label'].astype(int)
    
    return df

def pixel_test(ind, img_id, true_label):
    # for p in range(len(genotype)):
    #     population =

    # x, img, true_class, model, verbose=False
    print(ind)

    genotype = ind['genotype']
    
    evaluate([ind], x_test[img_id], true_label, models[0], {}, n_pixels)
    
    print(ind)
    
    l_suc, l_conf = attack_success([genotype], x_test[img_id], true_label, models[0])

    print(l_suc)
    print(l_conf)

def local_search(img_id, ind, true_label, n_trials=100, pixel_radius=2, rgb_radius=30):
    best_fitness = ind['fitness']
    best_genotype = [p.copy() for p in ind['genotype']]

    (h, w, d) = x_test[0].shape
    
    for _ in range(n_trials):
        new_genotype = []
        for pixel in ind['genotype']:
            x0, y0, r0, g0, b0 = pixel
            original_rgb = np.array([r0, g0, b0])
            original_rgb = np.array([x_test[img_id][pixel[0]][pixel[1]][0], x_test[img_id][pixel[0]][pixel[1]][1], x_test[img_id][pixel[0]][pixel[1]][2]])
            
            # Deslocamento pequeno no pixel
            dx = np.random.randint(-pixel_radius, pixel_radius + 1)
            dy = np.random.randint(-pixel_radius, pixel_radius + 1)
            new_x = np.clip(x0 + dx, 0, w - 1)
            new_y = np.clip(y0 + dy, 0, h - 1)
            
            # Pequena variação RGB
            new_rgb = original_rgb + np.random.randint(-rgb_radius, rgb_radius + 1, size=3)
            new_rgb = np.clip(new_rgb, 0, 255)
            
            new_pixel = [new_x, new_y] + new_rgb.tolist()
            # new_pixel = [x0, y0] + new_rgb.tolist()
            new_genotype.append(new_pixel)
        
        candidate = {'genotype': new_genotype, 'fitness': None, 'confidence': None, 'success': None}
        
        # Avalia fitness
        evaluate([candidate], x_test[img_id], true_label, models[0], {}, n_pixels)
        
        # Atualiza se melhorar
        if candidate['fitness'] > best_fitness:
            best_fitness = candidate['fitness']
            best_genotype = [p.copy() for p in new_genotype]
            # print(f"Found better fitness: {best_fitness}")
    
    # Atualiza o indivíduo
    ind['genotype'] = best_genotype
    ind['fitness'] = best_fitness
    
    # Avalia sucesso final
    l_suc, l_conf = attack_success([ind['genotype']], x_test[img_id], true_label, models[0])
    ind['success'] = l_suc
    ind['confidence'] = l_conf
    
    # print("Final individual:", ind)
    return best_genotype, best_fitness


def local_search(img_id, ind, true_label, n_trials=100, pixel_radius=2, rgb_radius=20):

    best_fitness = ind['fitness']
    best_genotype = [p.copy() for p in ind['genotype']]

    (h, w, d) = x_test[0].shape
    
    # número de tentativas por pixel
    trials_per_pixel = n_trials // len(ind['genotype'])
    remainder = n_trials % len(ind['genotype'])  # para dar +1 ao primeiro pixel

    for i, pixel in enumerate(ind['genotype']):
        extra = 1 if i < remainder else 0
        total_trials = trials_per_pixel + extra
        
        for _ in range(total_trials):

            # começamos SEMPRE no melhor genótipo atual
            new_genotype = [p.copy() for p in best_genotype]

            # pixel a alterar
            x0, y0, r0, g0, b0 = best_genotype[i]

            # obter RGB original da imagem
            orig_r, orig_g, orig_b = x_test[img_id][x0][y0]

            # pequeno deslocamento opcional (podes remover isto)
            dx = np.random.randint(-pixel_radius, pixel_radius+1)
            dy = np.random.randint(-pixel_radius, pixel_radius+1)
            new_x = int(np.clip(x0 + dx, 0, w-1))
            new_y = int(np.clip(y0 + dy, 0, h-1))

            # gerar RGB próximo do original (NÃO do genotype)
            new_r = np.clip(orig_r + np.random.randint(-rgb_radius, rgb_radius+1), 0, 255)
            new_g = np.clip(orig_g + np.random.randint(-rgb_radius, rgb_radius+1), 0, 255)
            new_b = np.clip(orig_b + np.random.randint(-rgb_radius, rgb_radius+1), 0, 255)

            # substituir apenas ESTE pixel
            new_genotype[i] = [new_x, new_y, int(new_r), int(new_g), int(new_b)]

            # avaliar
            candidate = {'genotype': new_genotype, 'fitness': None,
                         'confidence': None, 'success': None}

            evaluate([candidate], x_test[img_id], true_label, models[0], {}, len(ind['genotype']))

            if candidate['fitness'] > best_fitness:
                best_fitness = candidate['fitness']
                best_genotype = [p.copy() for p in new_genotype]

    # atualizar o indivíduo final
    ind['genotype'] = best_genotype
    ind['fitness'] = best_fitness

    # avaliar sucesso
    l_suc, l_conf = attack_success([best_genotype], x_test[img_id], true_label, models[0])
    ind['success'] = l_suc
    ind['confidence'] = l_conf

    return best_genotype, best_fitness

def local_search(img_id, ind, true_label, n_trials=100, pixel_radius=2, rgb_radius=20):

    best_fitness = ind['fitness']
    best_genotype = [p.copy() for p in ind['genotype']]

    (h, w, d) = x_test[0].shape
    
    # número de tentativas por pixel
    trials_per_pixel = n_trials // len(ind['genotype'])
    remainder = n_trials % len(ind['genotype'])  # para dar +1 ao primeiro pixel

    for i, pixel in enumerate(ind['genotype']):
        extra = 1 if i < remainder else 0
        total_trials = trials_per_pixel + extra
        
        for _ in range(total_trials):

            # começamos sempre no melhor genótipo atual
            new_genotype = [p.copy() for p in best_genotype]

            # pixel a alterar
            x0, y0, r0, g0, b0 = best_genotype[i]

            # obter RGB original da imagem
            orig_r, orig_g, orig_b = x_test[img_id][x0][y0]

            # pequeno deslocamento opcional
            dx = np.random.randint(-pixel_radius, pixel_radius+1)
            dy = np.random.randint(-pixel_radius, pixel_radius+1)
            new_x = int(np.clip(x0 + dx, 0, w-1))
            new_y = int(np.clip(y0 + dy, 0, h-1))

            # gerar RGB próximo do valor da imagem original
            new_r = np.clip(orig_r + np.random.randint(-rgb_radius, rgb_radius+1), 0, 255)
            new_g = np.clip(orig_g + np.random.randint(-rgb_radius, rgb_radius+1), 0, 255)
            new_b = np.clip(orig_b + np.random.randint(-rgb_radius, rgb_radius+1), 0, 255)

            # substituir apenas ESTE pixel
            new_genotype[i] = [new_x, new_y, int(new_r), int(new_g), int(new_b)]

            # avaliar
            candidate = {'genotype': new_genotype, 'fitness': None,
                         'confidence': None, 'success': None}

            evaluate([candidate], x_test[img_id], true_label, models[0], {}, len(ind['genotype']))

            if candidate['fitness'] > best_fitness:
                best_fitness = candidate['fitness']
                best_genotype = [p.copy() for p in new_genotype]

    # atualizar o indivíduo final
    ind['genotype'] = best_genotype
    ind['fitness'] = best_fitness

    # avaliar sucesso
    l_suc, l_conf = attack_success([best_genotype], x_test[img_id], true_label, models[0])
    ind['success'] = l_suc
    ind['confidence'] = l_conf

    return best_genotype, best_fitness


def main():
    # VERIFICATION
    # file_name, extention = os.path.splitext(os.path.basename(__file__))
    # if len(sys.argv[1:]) != 2:
    #     print(f"3 arguments required: {file_name + extention} <number of runs> <number of images>")

    # # RETRIEVE ARGUMENTS
    # for arg in sys.argv[1:]:
    #     n_runs = sys.argv[1]
    #     n_images = sys.argv[2]
    
    # DEFINE BOUNDS
    # (h, w, d) = x_test[0].shape
    # bounds = [[0, w - 1], [0, h - 1], [0, 255], [0, 255], [0, 255]]
    # bounds = np.array(bounds)

    # READ FILE
    for model in modelNames:
        for approach in abordagens:
            print(f"\n=== MODEL: {model} | APPROACH: {approach} ===")
            
            for run in range(1, nruns + 1):
                print(f"\n=== RUN {run} ===")
                input_file = os.path.join(results_file, model, approach, f"run_{run}", "best_individuals.csv")
                if not os.path.exists(input_file):
                    print(f"File not found: {input_file}, skipping.")
                    continue

                df = read_file(input_file)
                print("File extracted!")

                # EXTRACT ARRAYS FROM DF:
                best_individuals_before = df['best pixel'].to_numpy()
                best_fitness_before = df['fitness'].to_numpy()
                img_ids = df['img_id']
                true_labels = df['true label']

                # METRIRCS FOR COMPARISON
                best_individuals_after = []
                best_fitness_after = []
                img_changed = []

                for i in range(len(best_individuals_before)):
                    genotype = best_individuals_before[i]
                    fitness = best_fitness_before[i]
                    ind = {'genotype': genotype, 'fitness': fitness, 'confidence': None, 'success': None}
                    gen, fit = local_search(img_ids[i], ind, true_labels[i])

                    best_individuals_after.append(gen)
                    best_fitness_after.append(fit)

                    if fit != best_fitness_before[i]:
                        print(i, "|", best_fitness_before[i], "|", best_fitness_after[i])
                        img_changed.append(i)
                    else:
                        print(i, "|", fit)
                
                np.array(best_individuals_after)
                np.array(best_fitness_after)

                # print("MEDIA FITNESS ANTES: ", np.average(best_fitness_before[:5]))
                # print("MEDIA FITNESS DEPOIS: ", np.average(best_fitness_after))
                # print("BESTS | ANTES | DEPOIS:")
                # for i in img_changed:
                #     print(f"{i} | {best_individuals_before[i]} | {best_individuals_after[i]}")
                
                if img_changed:
                    results = []
                    for i in img_changed:
                        results.append({
                            'img_id': img_ids[i],
                            'true_label': true_labels[i],
                            'genotype_before': best_individuals_before[i],
                            'genotype_after': best_individuals_after[i],
                            'fitness_before': best_fitness_before[i],
                            'fitness_after': best_fitness_after[i]
                        })

                    # mesmo diretório do ficheiro lido
                    output_file = os.path.join(os.path.dirname(input_file), "local_search_new_best_pixels.csv")
                    pd.DataFrame(results).to_csv(output_file, index=False)
                    print(f"Results saved to {output_file}")
                else:
                    print("No individuals changed in this run.")

# def main():
#     img_id = 2757
#     genotype = [[27, 18, 40,  0, 87], [ 19,  16,  94, 192, 146], [  8,   4, 118,  48,  72]]
#     fitness = 2.006928287257223
#     true_label = 1
#     ind = {'genotype': genotype, 'fitness': fitness, 'confidence': None, 'success': None}

#     original_rgb_1 = np.array([x_test[img_id][27][18][0], x_test[img_id][27][18][1], x_test[img_id][27][18][2]])
#     original_rgb_2 = np.array([x_test[img_id][19][16][0], x_test[img_id][19][16][1], x_test[img_id][19][16][2]])
#     original_rgb_3 = np.array([x_test[img_id][8][4][0], x_test[img_id][8][4][1], x_test[img_id][8][4][2]])
#     print(original_rgb_1, original_rgb_2, original_rgb_3)

#     pixel_test(ind, img_id, true_label)

if __name__ == "__main__":
    main()