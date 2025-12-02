import pandas as pd
import numpy as np

# modelNames = ["regular"]
modelNames = ["distilled"]
# abordagens = ["ra"]
abordagens = ["ga"]
results_path = './results'
# nruns = 10
nruns = 5
n_samples = 500
pop_size = 400

# Create a text file for output
output_file_path = "./results/results_full.txt"
output_file_path_simp = "./results/results_simp.txt"

with open(output_file_path, "w") as output_file:
    with open(output_file_path_simp, "w") as output_file_simp:
        # Analyse covered pixels
        print("---------- Mean number of covered pixels ----------", file=output_file)
        print("---------- Mean number of covered pixels ----------", file=output_file_simp) # SIMPLIFIED TXT
        # mean das runs por imagem e depois mean das imagens
        for modelName in modelNames:
            print(modelName, file=output_file_simp) # SIMPLIFIED TXT

            for abordagem in abordagens:
                print(modelName, " - ", abordagem, file=output_file)
                file = f"{results_path}/{modelName}/{abordagem}/covered_pixels.csv"
                data = pd.read_csv(file)
                result = data.groupby(['run'])['number of covered pixels'].mean().reset_index()
                print(result['number of covered pixels'], file=output_file)
                print(abordagem, np.mean(result['number of covered pixels']), np.std(result['number of covered pixels']), file=output_file)
                
                # SIMPLIFIED TXT:
                print(abordagem, np.mean(result['number of covered pixels']), np.std(result['number of covered pixels']), file=output_file_simp)

        # Mean adv per img
        print("---------- Mean quantity of adversarial images found for an original image ----------", file=output_file)
        print("---------- Mean quantity of adversarial images found for an original image ----------", file=output_file_simp) # SIMPLIFIED TXT
        # mean das imagens por run e depois mean das runs
        for modelName in modelNames:
            model_path = f'{results_path}/{modelName}'
            print(modelName, file=output_file_simp) # SIMPLIFIED TXT

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
                print("mean runs ", np.mean(mean_runs), np.std(mean_runs), file=output_file)
                print(abordagem, np.mean(mean_runs), np.std(mean_runs), file=output_file_simp) # SIMPLIFIED TXT  

        # Nevals for first success
        print("---------- Mean number of evaluations before finding an adversarial image ----------", file=output_file)
        print("skiping images with no success", file=output_file)
        print("---------- Mean number of evaluations before finding an adversarial image ----------", file=output_file_simp) # SIMPLIFIED TXT
        print("skiping images with no success", file=output_file_simp) # SIMPLIFIED TXT

        # mean das imagens por run e depois mean das runs
        for modelName in modelNames:
            model_path = f'{results_path}/{modelName}'
            print(modelName, file=output_file_simp) # SIMPLIFIED TXT

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
                print("mean runs: ", np.mean(mean_runs), np.std(mean_runs), file=output_file)
                print(abordagem, np.mean(mean_runs), np.std(mean_runs), file=output_file_simp)

        print("---------- Mean number of evaluations before finding an adversarial image ----------", file=output_file)
        print("counting as 40 000 evals when no success", file=output_file)
        print("---------- Mean number of evaluations before finding an adversarial image ----------", file=output_file_simp)
        print("counting as 40 000 evals when no success", file=output_file_simp)

        for modelName in modelNames:
            model_path = f'{results_path}/{modelName}'
            print(modelName, file=output_file_simp) # SIMPLIFIED TXT

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
                print("mean runs: ", np.mean(mean_runs), np.std(mean_runs), file=output_file)
                print(abv, np.mean(mean_runs), np.std(mean_runs), file=output_file_simp) # SIMPLIFIED TXT

        # Success rate per dataset (das n_samples, quantas conseguiu achar advs?)
        print("---------- Success rate per dataset ----------", file=output_file)
        print("---------- Success rate per dataset ----------", file=output_file_simp) # SIMPLIFIED TXT
        #print("success rate", file=output_file)
        for modelName in modelNames:
            model_path = f'{results_path}/{modelName}'
            print(modelName, file=output_file_simp) # SIMPLIFIED TXT

            for abv in abordagens:
                print(modelName, " - ", abv, file=output_file)
                data = pd.read_csv(f'{model_path}/{abv}/metrics.csv')
                # suc_rate = data.iloc[:, 0]  # estava a ir buscar a primeira coluna que é o nº da run
                suc_rate = data["success rate dataset"]  # <-- coluna correta aqui
                print(suc_rate, file=output_file)
                print(np.mean(suc_rate), np.std(suc_rate), file=output_file)
                print(abv, np.mean(suc_rate), np.std(suc_rate), file=output_file_simp) # SIMPLIFIED TXT

        ##########################
        #   LOCAL SEARCH         #
        ##########################
        print("---------- Local Search Metrics ----------", file=output_file)
        print("---------- Local Search Metrics ----------", file=output_file_simp)

        for modelName in modelNames:
            print(modelName, file=output_file_simp)

            for abordagem in abordagens:

                print(modelName, " - ", abordagem, file=output_file)

                fitness_per_run = []
                it_first_per_run = []
                it_final_per_run = []
                succ_trans_per_run = []

                # Cabeçalho da tabela
                print("Run,Fitness change %,It first best,It final best,Success F→T %", file=output_file)

                for i in range(1, nruns + 1):

                    file_path = f"{results_path}/{modelName}/{abordagem}/run_{i}/local_search_new_best_pixels_temp.csv"

                    try:
                        df = pd.read_csv(file_path)
                    except FileNotFoundError:
                        print(f"Missing file: {file_path}", file=output_file)
                        continue

                    # Limpar espaços nos nomes das colunas
                    df.columns = df.columns.str.strip()

                    # Normalizar booleanos
                    df["success_before_norm"] = df["success_before"].astype(str).str.contains("True")
                    df["success_after_norm"] = df["success_after"].astype(str).str.contains("True")

                    # FITNESS IMPROVEMENT %
                    df["fitness_change_pct"] = ((df["fitness_after"] - df["fitness_before"]) / df["fitness_before"]) * 100
                    fitness_mean = df["fitness_change_pct"].mean()
                    fitness_per_run.append(fitness_mean)

                    # ITERAÇÕES
                    it_first = df["it_first_best"].mean()
                    it_final = df["it_final_best"].mean()
                    it_first_per_run.append(it_first)
                    it_final_per_run.append(it_final)

                    # TRANSIÇÕES DE SUCCESS (False → True)
                    transitions = ((df["success_before_norm"] == False) & (df["success_after_norm"] == True)).sum()
                    succ_rate = (transitions / len(df)) * 100
                    succ_trans_per_run.append(succ_rate)

                    # Print por run numa linha da tabela
                    print(f"{i},{fitness_mean:.2f},{it_first:.1f},{it_final:.1f},{succ_rate:.1f}", file=output_file)

                # ==== PRINT MÉDIAS GERAIS ====
                print("\n--- Summary ---", file=output_file)
                print(f"Average Fitness change %: {np.mean(fitness_per_run):.2f} ± {np.std(fitness_per_run):.2f}", file=output_file)
                print(f"Average It first best: {np.mean(it_first_per_run):.1f} ± {np.std(it_first_per_run):.1f}", file=output_file)
                print(f"Average It final best: {np.mean(it_final_per_run):.1f} ± {np.std(it_final_per_run):.1f}", file=output_file)
                print(f"Average Success F→T %: {np.mean(succ_trans_per_run):.1f} ± {np.std(succ_trans_per_run):.1f}", file=output_file)

                # ---- SIMPLIFIED TEXT ----
                print(abordagem,
                    f"{np.mean(fitness_per_run):.2f}",
                    f"{np.mean(it_first_per_run):.1f}",
                    f"{np.mean(it_final_per_run):.1f}",
                    f"{np.mean(succ_trans_per_run):.1f}",
                    file=output_file_simp)
