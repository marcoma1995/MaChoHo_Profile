from merge import data_male

from bld.project_paths import project_paths_join as ppj

# Output the DataFrame.


def save_data(merge_fale):
    merge_male.to_csv(ppj("OUT_DATA", "merge_male.csv"),
                      sep=",")


if __name__ == "__main__":
    merge_male = data_male
    save_data(merge_male)
