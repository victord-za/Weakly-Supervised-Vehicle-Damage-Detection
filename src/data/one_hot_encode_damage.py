import pandas as pd

def process_damage_data(filepath):
    df = pd.read_csv(filepath)
    df.drop(['img_url'], axis=1, inplace=True)

    one_hot_encoded = df['damage_detail'].str.get_dummies(sep=',')
    one_hot_encoded.columns = ['DAMAGE_' + col for col in one_hot_encoded.columns]

    df = pd.concat([df, one_hot_encoded], axis=1)
    df.drop(['damage_detail'], axis=1, inplace=True)

    return df