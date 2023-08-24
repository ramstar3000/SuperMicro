import os
import pandas as pd



def Sorter(folderPath, path_to_plates):

    '''
    This function sorts the data from the clusterProfile files into a single file. It also adds the location of the well and the FOV number.

    Parameters
    ----------
    folderPath : str
        The path to the folder containing the clusterProfile files.
    path_to_plates : str
        The path to the folder containing the plate.xlsx file

    Returns
    -------
    None

    Files
    -----
    all_data.txt : The file containing all the data from the clusterProfile files.
    
    
    '''

    directory = path_to_plates + "//" + "plate.xlsx"

    # Read the different sheets within the directory

    xl = pd.ExcelFile(directory)

    titles = xl.sheet_names

    sheets = {sheet_name: xl.parse(sheet_name, header=None)
              for sheet_name in titles}


    locations = sheets['location']
    all_dfs = []

    for x in range(locations.shape[0]):
        for y in range(locations.shape[1]):
            well = locations.iloc[x, y]
            other_data = [well]

            for sheet in xl.sheet_names:
                if sheet != 'location':
                    other_data.append(sheets[sheet].iloc[x, y])

            fileList = os.listdir(folderPath)
            fileList = [file for file in fileList if (
                "clusterprofile" in file) or ("clusterProfile" in file)]
            fileList = [file for file in fileList if well.lower()
                        in file.lower()]

            s = 0

            flag = False

            for n, file in enumerate(fileList):
                df = pd.read_csv(folderPath + "//" + file, index_col=0)

                if not flag:
                    titles2 = list(titles) + list(df.columns)
                    flag = True

                df['FOV'] = n + 1

                for title in range(len(titles)):
                    df[titles[title]] = other_data[title]

                all_dfs.append(df)

                s += len(df)

    df = pd.concat(all_dfs)

    titles2 = ["FOV"] + titles2

    df.to_string(folderPath + "//" + "all_data.txt", index=False, columns = titles2 )