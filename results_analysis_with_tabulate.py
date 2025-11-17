import pandas as pd
import numpy as np
from tabulate import tabulate

# modelNames = ["regular"]
modelNames = ["distilled"]
# abordagens = ["ra"]
abordagens = ["ga"]
results_path = './results_multipixel_first'
# nruns = 10
nruns = 5
n_samples = 500
pop_size = 400

# Create a text file for output
output_file_path = "./results_multipixel_first/results_full.txt"
output_file_path_simp = "./results_multipixel_first/results_simp.txt"
headers = ["Approach", "Mean", "Std Dev"]

with open(output_file_path, "w") as output_file:
    with open(output_file_path_simp, "w") as output_file_simp:
        # Analyse covered pixels
        print("---------- Mean number of covered pixels ----------", file=output_file)
        print("---------- Mean number of covered pixels ----------", file=output_file_simp) # SIMPLIFIED TXT
        
        # mean das runs por imagem e depois mean das imagens
        for modelName in modelNames:
            print(modelName.upper(), file=output_file_simp) # SIMPLIFIED TXT
            table = []

            for abordagem in abordagens:
                print(modelName, " - ", abordagem, file=output_file)
                file = f"{results_path}/{modelName}/{abordagem}/covered_pixels.csv"
                data = pd.read_csv(file)
                result = data.groupby(['run'])['number of covered pixels'].mean().reset_index()

                mean_val = np.mean(result['number of covered pixels'])
                std_val = np.std(result['number of covered pixels'])

                print(result['number of covered pixels'], file=output_file)
                print(abordagem, mean_val, std_val, file=output_file)
                
                # SIMPLIFIED TXT:
                table.append([abordagem, f"{mean_val:.3f}", f"{std_val:.3f}"])
            # SIMPLIFIED TXT:
            print(tabulate(table, headers=headers, tablefmt="github"), file=output_file_simp)

        # Mean adv per img
        print("\n---------- Mean quantity of adversarial images found for an original image ----------", file=output_file)
        print("\n---------- Mean quantity of adversarial images found for an original image ----------", file=output_file_simp) # SIMPLIFIED TXT
        # mean das imagens por run e depois mean das runs
        for modelName in modelNames:
            model_path = f'{results_path}/{modelName}'
            print(modelName.upper(), file=output_file_simp) # SIMPLIFIED TXT
            table = []

            for abordagem in abordagens:
                print(modelName, " - ", abordagem, file=output_file)
                mean_runs = []
                for run in range(1, 1 + nruns):    
                    adv_per_image = []

                    for img in range(n_samples):
                        success_data = pd.read_csv(f'{model_path}/{abordagem}/run_{run}/img_{img}/success_file.csv')
                        n_advs = len(success_data)
                        adv_per_image.append(n_advs)

                    mean = np.mean(adv_per_image)
                    print(mean, file=output_file)
                    mean_runs.append(mean)
                mean_val = np.mean(mean_runs)
                std_val = np.std(mean_runs)
                print("mean runs ", mean_val, std_val, file=output_file)
                table.append([abordagem, f"{mean_val:.3f}", f"{std_val:.3f}"])  # SIMPLIFIED TXT
            # SIMPLIFIED TXT:
            print(tabulate(table, headers=headers, tablefmt="github"), file=output_file_simp)

        # Nevals for first success
        print("\n---------- Mean number of evaluations before finding an adversarial image ----------", file=output_file)
        print("skiping images with no success\n", file=output_file)
        print("\n---------- Mean number of evaluations before finding an adversarial image ----------", file=output_file_simp) # SIMPLIFIED TXT
        print("skiping images with no success\n", file=output_file_simp) # SIMPLIFIED TXT

        # mean das imagens por run e depois mean das runs
        for modelName in modelNames:
            model_path = f'{results_path}/{modelName}'
            print(modelName.upper(), file=output_file_simp) # SIMPLIFIED TXT

            table = []

            for abordagem in abordagens:
                print(modelName, " - ", abordagem, file=output_file)
                mean_runs = []
                for run in range(1, 1 + nruns):    
                    nevals_for_success = []

                    for img in range(n_samples):
                        success_data = pd.read_csv(f'{model_path}/{abordagem}/run_{run}/img_{img}/success_file.csv')
                        has_rows = len(success_data) > 0
                        if not has_rows:
                            continue
                        gen_first_success = success_data['gen'][0]
                        if abordagem == 'ra':
                            nevals_for_success.append(gen_first_success)
                        else:
                            nevals_for_success.append(gen_first_success*pop_size)

                    mean = np.mean(nevals_for_success)
                    mean_runs.append(mean)
                    print(mean, file=output_file)

                mean_val = np.mean(mean_runs)
                std_val = np.std(mean_runs)

                print("mean runs: ", mean_val, std_val, file=output_file)
                table.append([abordagem, f"{mean_val:.3f}", f"{std_val:.3f}"])  # SIMPLIFIED TXT
            print(tabulate(table, headers=headers, tablefmt="github"), file=output_file_simp)

        print("\n---------- Mean number of evaluations before finding an adversarial image ----------", file=output_file)
        print("counting as 40 000 evals when no success\n", file=output_file)
        print("\n---------- Mean number of evaluations before finding an adversarial image ----------", file=output_file_simp)
        print("counting as 40 000 evals when no success\n", file=output_file_simp)

        for modelName in modelNames:
            model_path = f'{results_path}/{modelName}'
            print(modelName.upper(), file=output_file_simp) # SIMPLIFIED TXT

            table = []

            for abv in abordagens:
                print(modelName, " - ", abv, file=output_file)
                mean_runs = []
                for run in range(1, 1 + nruns):    
                    nevals_for_success = []

                    for img in range(n_samples):
                        success_data = pd.read_csv(f'{model_path}/{abv}/run_{run}/img_{img}/success_file.csv')
                        has_rows = len(success_data) > 0
                        if not has_rows:
                            nevals_for_success.append(40000)
                        else:
                            gen_first_success = success_data['gen'][0]
                            if abordagem == 'ra':
                                nevals_for_success.append(gen_first_success)
                            else:
                                nevals_for_success.append(gen_first_success*pop_size)

                    mean = np.mean(nevals_for_success)
                    mean_runs.append(mean)
                    print(mean, file=output_file)

                mean_val = np.mean(mean_runs)
                std_val = np.std(mean_runs)

                print("mean runs: ", mean_val, std_val, file=output_file)
                table.append([abv, f"{mean_val:.3f}", f"{std_val:.3f}"])  # SIMPLIFIED TXT
            print(tabulate(table, headers=headers, tablefmt="github"), file=output_file_simp)

        # Success rate per dataset (das n_samples, quantas conseguiu achar advs?)
        print("\n---------- Success rate per dataset ----------", file=output_file)
        print("\n---------- Success rate per dataset ----------", file=output_file_simp) # SIMPLIFIED TXT
        #print("success rate", file=output_file)
        for modelName in modelNames:
            model_path = f'{results_path}/{modelName}'
            print(modelName.upper(), file=output_file_simp) # SIMPLIFIED TXT

            table = []

            for abv in abordagens:
                print(modelName, " - ", abv, file=output_file)
                data = pd.read_csv(f'{model_path}/{abv}/metrics.csv')
                # suc_rate = data.iloc[:, 0]  # estava a ir buscar a primeira coluna que é o nº da run
                suc_rate = data["success rate dataset"]  # <-- coluna correta aqui
                print(suc_rate, file=output_file)

                mean_val = np.mean(suc_rate)
                std_val = np.std(suc_rate)

                print(mean_val, std_val, file=output_file)
                table.append([abv, f"{mean_val:.3f}", f"{std_val:.3f}"])  # SIMPLIFIED TXT  
            print(tabulate(table, headers=headers, tablefmt="github"), file=output_file_simp)

        print("\n---------- DISTORTION ----------", file=output_file)
        print("\n---------- DISTORTION ----------", file=output_file_simp) # SIMPLIFIED TXT
        for modelName in modelNames:
            model_path = f'{results_path}/{modelName}'
            print(modelName.upper(), file=output_file_simp) # SIMPLIFIED TXT

            table = []

            for abv in abordagens:
                print(modelName, " - ", abv, file=output_file)
                data = pd.read_csv(f'{model_path}/difMeans.csv')
                data_abv = data[data["abordagem"] == abv]
                print(data_abv, file=output_file)

                mean_means = data_abv["mean das means"].values
                std_means = data_abv["std das means"].values
                mean_mins = data_abv["mean do minimo"].values
                std_mins = data_abv["std do min"].values

                print("mean das means: ", mean_means, std_means, file=output_file)
                print("mean do min: ", mean_mins, std_mins, file=output_file)
                table.append([abv, f"{mean_means[0]:.3f}", f"{std_means[0]:.3f}"])  # SIMPLIFIED TXT  
            print(tabulate(table, headers=headers, tablefmt="github"), file=output_file_simp)

        print("\n---------- TIME ----------", file=output_file)
        print("\n---------- TIME ----------", file=output_file_simp) # SIMPLIFIED TXT
        for modelName in modelNames:
            model_path = f'{results_path}/{modelName}'
            print(modelName.upper(), file=output_file_simp) # SIMPLIFIED TXT

            table = []

            for abv in abordagens:
                print(modelName, " - ", abv, file=output_file)
                data = pd.read_csv(f'{model_path}/{abv}/time.csv')
                times = data["time"][-nruns:]  # só nas últimas runs
                print(times, file=output_file)

                mean_val = np.mean(times)
                std_val = np.std(times)

                print(mean_val, std_val, file=output_file)
                table.append([abv, f"{mean_val:.3f}", f"{std_val:.3f}"])  # SIMPLIFIED TXT  
            print(tabulate(table, headers=headers, tablefmt="github"), file=output_file_simp)