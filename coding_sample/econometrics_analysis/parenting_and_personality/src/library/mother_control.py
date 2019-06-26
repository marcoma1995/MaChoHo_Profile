from father_control import mother_controls

from bld.project_paths import project_paths_join as ppj

# Output the DataFrame.


def save_data(mother_control):
    mother_control.to_csv(ppj("OUT_DATA", "mother_control.csv"),
                          sep=",")


if __name__ == "__main__":
    mother_control = mother_controls
    save_data(mother_control)
