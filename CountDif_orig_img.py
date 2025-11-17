# Calculate distortion of each pixel attack


import pandas as pd
from keras.datasets import cifar10
import matplotlib.pyplot as plt
from setup_cifar import CIFAR

def clean_genotype(x):
    # se já é lista de ints, retorna direto
    if isinstance(x, list):
        return [int(v) for v in x]

    # remove aspas externas
    x = str(x).strip().strip('"').strip("'")

    # remove np.int64(...)
    x = x.replace("np.int64(", "").replace(")", "")

    # remove colchetes
    x = x.strip("[]")

    # divide por vírgula ou espaço
    if "," in x:
        parts = x.split(",")
    else:
        parts = x.split()

    # converte para int e remove strings vazias
    return [int(p) for p in parts if p.strip()]

import re
import numpy as np
import ast

# version of clean_genotype that is automatic to multiple pixels
def clean_genotype(x):
    # --- 1️⃣ Caso já seja um array numpy ---
    if isinstance(x, np.ndarray):
        return x.astype(int).tolist()

    # --- 2️⃣ Caso já seja uma lista (de ints ou listas) ---
    if isinstance(x, list):
        def convert_all(v):
            if isinstance(v, list):
                return [convert_all(i) for i in v]
            try:
                return int(v)
            except:
                return v
        return convert_all(x)

    # --- 3️⃣ Converte para string e limpa ---
    x = str(x).strip().strip('"').strip("'")

    # Remove prefixos de numpy
    x = re.sub(r'np\.array\(|array\(', '', x)

    # Remove parênteses finais, se existirem
    x = re.sub(r'\)+$', '', x)

    # Garante colchetes em volta (para ast.literal_eval)
    if not x.startswith('['):
        x = '[' + x + ']'

    # --- 4️⃣ Tenta interpretar como lista Python ---
    try:
        parsed = ast.literal_eval(x)
    except Exception:
        # fallback: se falhar, extrai todos os números manualmente
        nums = re.findall(r'-?\d+', x)
        return [int(n) for n in nums]

    # --- 5️⃣ Converte tudo para int, recursivamente ---
    def to_int_list(y):
        if isinstance(y, list):
            return [to_int_list(i) for i in y]
        try:
            return int(y)
        except:
            return y

    return to_int_list(parsed)


def main():
    # modelNames = ['regular', 'distilled']
    modelNames = ['distilled']
    # abordagens = ['de', 'ga', 'ra', 'cmaes']
    # abordagens = ['de', 'ga']
    abordagens = ['ga']
    #nruns = 10
    #n_imgs = 500
    nruns = 5
    n_imgs = 500

    # Load dataset
    data_cifar = CIFAR()
    x_test = data_cifar.test_data
    x_test = (x_test + 0.5) * 255

    for modelName in modelNames:
        path_model = f'./results/{modelName}'
        print(f"\n\n\t\t\tMODELO: {modelName}\n\n")

        for abordagem in abordagens:
            print(f"\n\t\tABORDAGEM: {abordagem}")
            for i_run in range(1, nruns+1):
                print(f"\n\tRUN: {i_run}\n")
                for i_img in range(n_imgs):
                    print(f"IMAGEM: {i_img}")
                        
                    # path the metrics_mean_img.csv - ficheiro q usamos para buscar o id
                    path_id = f'{path_model}/metrics_mean_img.csv'
                    ids = pd.read_csv(path_id)["img_idx"].tolist()
                    
                    # buscar o id da imagem atual
                    id_atual = ids[i_img + n_imgs * abordagens.index(abordagem)]

                    # imagem atual
                    img = x_test[id_atual]

                    # BUSCAR DADOS AO FICHEIRO
                    # aqui n sei pq quando fiz download os modelos nao apareceram, com os modelos é o primeiro path
                    file_success = f"{path_model}/{abordagem}/run_{i_run}/img_{i_img}/success_file.csv"
                    # this_path = path + f"/{modelName}/{abordagem}/run_{i_run + 1}/img_{i_img}/success_file.csv"
                    # this_path = path + f"//{abordagem}/run_{i_run + 1}/img_{i_img}/success_file.csv"
                    data = pd.read_csv(file_success)
                    data["genotype"] = data["genotype"].apply(clean_genotype)   # clean lists
                    print(data)
                    # listas das colunas
                    gen = data["gen"].tolist()
                    genotype = data["genotype"].tolist()
                    true_label = data["true label"].tolist()
                    predicted_label = data["predicted label"].tolist()
                    confidence = data.iloc[:, 3].tolist()

                        # CALCULAR A DIFERENÇA
                    dif = list()

                    for gt in genotype:
                        # # gt é uma string - transformar em lista -> its not anymore after applying clean_genotype
                        # gt = gt[1:-1]   # retirar os parentesis retos '[' e ']'
                        # if ',' in gt:
                        #     gt = gt.split(',')
                        # else:
                        #     gt = gt.split(" ")
                        # while '' in gt:
                        #     gt.remove('')
                        # for i in range(len(gt)):
                        #     gt[i] = int(gt[i])



                        # dif = (abs(x[2] - image_orig[x[0]][x[1]][0]) + abs(x[3] - image_orig[x[0]][x[1]][1]) + abs(x[4] - image_orig[x[0]][x[1]][2]))
                        d = (abs(gt[2] - img[gt[0]][gt[1]][0]) + abs(gt[3] - img[gt[0]][gt[1]][1]) + abs(gt[4] - img[gt[0]][gt[1]][2]))
                        dif.append(d)   # colocar na lista de diferentes

                    # Testar q os comprimentos estao todos iguais
                    #print(f"gen: {len(gen)}\ngenotype: {len(genotype)}\ntrue label: {len(true_label)}\npredicted label: {len(predicted_label)}\nconfidence: {len(confidence)}\ndif: {len(dif)}")

                    d = {"gen": gen, "genotype": genotype, "true label": true_label, "predicted label": predicted_label, "confidence in wrong label": confidence, "dif": dif}
                    df = pd.DataFrame(d)
                    #print(df)
                    new_success_file = f"{path_model}/{abordagem}/run_{i_run}/img_{i_img}/new_success_file.csv"
                    df.to_csv(new_success_file, index = False) 
                    

if __name__ == "__main__":
    main()