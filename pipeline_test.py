import os
import sys

import json
import threading


# Get the name of the config folder from the user

def config_generate():
    loss_group = ["approxndcg", "neuralndcg", "ranknet", "mse"]
    # Loop through the config files
    for loss in loss_group:
        config_folder = "allrank\configs"
        config_files = [f for f in os.listdir(config_folder) if
                        f.startswith('config_{}'.format(loss)) and f.endswith('.json')]
        for config_file in config_files:
            task = config_file.split('_')[2]
            if task == "mq2008":
                basic_value = "C:\\Users\\tyfann\\Documents\\TU Delft\\IN4325 Information Retrieval\\IN4325-IR\\data\\MQ2008\\{}"

            else:
                basic_value = "C:\\Users\\tyfann\\Documents\\TU Delft\\IN4325 Information Retrieval\\IN4325-IR\\data\\MSLR-WEB10K\\{}"

            config_file_path = os.path.join(config_folder, config_file)
            # 读取原始JSON文件
            with open(config_file_path, "r") as f:
                config = json.load(f)

            fold_common_name = "Fold"
            fold_names = ["{}{}_normalized".format(fold_common_name, i) for i in range(1, 6, 1)]
            print(fold_names)
            for fold_name in fold_names:
                value = basic_value.format(
                    fold_name)
                config["data"]["path"] = value
                config["training"]["epochs"] = 30
                config["training"]["early_stopping_patience"] = 30
                if "epoch" in config["training"]:
                    del config["training"]["epoch"]
                config["val_metric"] = "ndcg_5"
                config["metrics"] = ["ndcg_5","ndcg_10","ndcg_30","ndcg_60","dcg_5","dcg_10","dcg_30","dcg_60", "mrr_5","mrr_10","mrr_30","mrr_60"]
                if "mrr_5" in config["expected_metrics"]["val"]:
                    val = config["expected_metrics"]["val"].pop("mrr_5")
                    config["expected_metrics"]["val"]["ndcg_5"] = val

                new_config_file = config_file.replace(config_file.split("_")[3].split(".")[0], fold_name.split("_")[0].lower())

                new_filename = os.path.join(config_folder, new_config_file)
                with open(new_filename, "w") as f:
                    json.dump(config, f, indent=4)


def dcg_config_generate():
    loss_group = ["approxndcg", "neuralndcg", "ranknet", "mse"]
    # Loop through the config files
    for loss in loss_group:
        config_folder = "allrank\configs"
        config_files = [f for f in os.listdir(config_folder) if
                        f.startswith('config_{}'.format(loss)) and f.endswith('.json')]
        for config_file in config_files:
            task = config_file.split('_')[2]
            if task == "mq2008":
                basic_value = "C:\\Users\\tyfann\\Documents\\TU Delft\\IN4325 Information Retrieval\\IN4325-IR\\data\\MQ2008\\{}"

            else:
                basic_value = "C:\\Users\\tyfann\\Documents\\TU Delft\\IN4325 Information Retrieval\\IN4325-IR\\data\\MSLR-WEB10K\\{}"

            config_file_path = os.path.join(config_folder, config_file)
            # 读取原始JSON文件
            with open(config_file_path, "r") as f:
                config = json.load(f)

            fold_common_name = "Fold"
            fold_names = ["{}{}_normalized".format(fold_common_name, i) for i in range(1, 6, 1)]
            print(fold_names)
            for fold_name in fold_names:
                value = basic_value.format(
                    fold_name)
                config["data"]["path"] = value
                config["training"]["epochs"] = 30
                config["training"]["early_stopping_patience"] = 30
                if "epoch" in config["training"]:
                    del config["training"]["epoch"]
                config["val_metric"] = "dcg_5"
                config["metrics"] = ["dcg_5","dcg_10","dcg_30","dcg_60"]
                if "mrr_5" in config["expected_metrics"]["val"]:
                    val = config["expected_metrics"]["val"].pop("mrr_5")
                    config["expected_metrics"]["val"]["dcg_5"] = val

                new_config_file = config_file.replace(config_file.split("_")[3].split(".")[0], fold_name.split("_")[0].lower())

                new_filename = os.path.join(config_folder, new_config_file)
                with open(new_filename, "w") as f:
                    json.dump(config, f, indent=4)

if __name__ == '__main__':
    # config_generate()
    # exit(0)

    threads = []
    config_folder = "allrank\configs"
    # loss_group = ["approxndcg", "neuralndcg", "ranknet", "mse"]
    loss_group = ["mse"]
    # Loop through the config files
    for loss in loss_group:
        # Build the path to the config file
        config_files = [f for f in os.listdir(config_folder) if
                        f.startswith('config_{}'.format(loss)) and f.endswith('.json')]
        # config_file = os.path.join(config_folder, "config_{}.json".format(i))
        # print(config_files)
        for config_file_name in config_files:
            task_name = config_file_name.split("_")[2]
            fold_name = config_file_name.split("_")[3].split(".")[0]

            if not task_name == "web10":
                continue
            else:
                run_id = "test_run_{}".format(loss)
                job_dir = os.path.join(r"allrank\task-{}".format(task_name), "{}-lr0001-{}-ndcg".format(loss, fold_name))

                # if
                config_file_name = os.path.join(config_folder, config_file_name)
                # print(
                #     "python allrank\main.py --config-file-name {} --run-id {} --job-dir {}".format(config_file_name, run_id,
                #                                                                                    job_dir))
                t = threading.Thread(target=os.system("python allrank\main.py --config-file-name {} --run-id {} --job-dir {}".format(config_file_name, run_id, job_dir)))
                threads.append(t)

            # os.system("python allrank\main.py --config-file-name {} --run-id {} --job-dir {}".format(config_file_name, run_id, job_dir))

    for t in threads:
        t.start()

    for t in threads:
        t.join()


    # # Extract the file number
    # file_number = str(i)
    # file_number = file_number.replace("config", "")
    #
    # # Call the main script with the config file and file number as arguments
    # os.system("python main.py {} {}".format(config_file, file_number))
