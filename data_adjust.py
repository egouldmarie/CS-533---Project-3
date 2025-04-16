import pandas as pd
import numpy as np



def get_subset(df, n, filename):

    new_df = df.sample(n)
    new_df.to_csv(filename + '-' + str(n) + '.csv', index = False)


if __name__=="__main__":
    filename1 = "DS-1/dataset-1-1"
    filename2 = "DS-2/dataset-2-1"
    filename3 = "DS-3/dataset-3-1"
    filename4 = "DS-4/dataset-4-1"

    # CHANGE FILE USED FOR TESTING HERE 
    FILE_IN_USE = filename1

    df = pd.read_csv(FILE_IN_USE + ".csv")

    get_subset(df, 2000, FILE_IN_USE)

    for i in range(2000, 5001, 300):

        get_subset(df, i, FILE_IN_USE)

    



    