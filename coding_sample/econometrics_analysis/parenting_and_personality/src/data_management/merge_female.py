from merge import data_female

from bld.project_paths import project_paths_join as ppj

# Output the DataFrame.


def save_data(merge_female):
    merge_female.to_csv(ppj("OUT_DATA", "merge_female.csv"),
                        sep=",")


if __name__ == "__main__":
    merge_female = data_female
    save_data(merge_female)
