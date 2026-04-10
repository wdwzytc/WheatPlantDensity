import pandas as pd
import datetime


def load_dataset(path):
    return pd.read_excel(path, sheet_name='Data')


def str_is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def get_dataset_for_dynamic(df):
    """
    get target data from dataset_index.xlsx
    Data of Avignon 2023-2024 with full dynamics
        (remove some scattered images for image annotations)
    Data of Fourques 2022-2023 with full dynamics
    Data of Salin-de-Giraud 2022-2023 with full dynamics

    """

    # rows to keep
    flag1 = [site in ['Avignon', 'Fourques', 'Salin-de-Giraud'] for site in df['Site']]
    date_lst = [datetime.datetime(2023, 11, 29, 0, 0),
                datetime.datetime(2022, 10, 28, 0, 0),
                datetime.datetime(2022, 11, 18, 0, 0)]
    flag2 = [date in date_lst for date in df['Date of sowing']]
    flag3 = (df['CVAT task id'] != 60)  # small avignon dataset, cropped
    flag4 = df['view zenith angle (°)'] == 45
    flag5 = [str_is_float(i) for i in df['note14 Air temperature_Thermal time #4']]
    df_c = df[pd.Series(flag1)
              & pd.Series(flag2)
              & pd.Series(flag3)
              & pd.Series(flag4)
              & pd.Series(flag5)]

    pass  # keep all columns

    return df_c
