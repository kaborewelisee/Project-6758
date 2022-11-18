import nhl_dataset
import nhl_tidy_data

raw_data_folder_path = './data'
csv_path = './data/df.csv'
seasons = [2015, 2016, 2017, 2018, 2019]

if __name__ == "__main__":
    dataset = nhl_dataset.NhlDataset(raw_data_folder_path)
    dataset.load_all(seasons)
    nhl_tidy_data.convert_raw_data_to_panda_csv(dataset, csv_path, seasons)