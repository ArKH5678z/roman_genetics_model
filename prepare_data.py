import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def load_ind(filepath):
    df = pd.read_csv(
        filepath,
        sep=r'\s+',
        header=None,
        names=["sample_id", "sex", "population"]
    )
    return df

def load_anno(filepath):
    df = pd.read_csv(
        filepath,
        sep='\t',
        header=0,
        low_memory=False
    )
    return df

def filter_roman_samples(anno_df):
    date_col = 'Date mean in BP in years before 1950 CE [OxCal mu for a direct radiocarbon date, and average of range for a contextual date]'
    entity_col = 'Political Entity'

    date_filtered = anno_df[
        (pd.to_numeric(anno_df[date_col], errors='coerce') >= 1250) &
        (pd.to_numeric(anno_df[date_col], errors='coerce') <= 2450)
    ]

    roman_regions = [
        'Italy', 'Greece', 'Turkey', 'Tunisia', 'Libya', 'Egypt',
        'Spain', 'France', 'Croatia', 'Romania', 'Bulgaria',
        'Syria', 'Lebanon', 'Israel', 'Jordan', 'Algeria', 'Morocco'
    ]

    geo_filtered = date_filtered[
        date_filtered[entity_col].str.contains(
            '|'.join(roman_regions), case=False, na=False
        )
    ]

    return geo_filtered

if __name__ == "__main__":
    ind = load_ind(os.path.join(DATA_DIR, "v66.1240K.aadr.PUB.ind"))
    anno = load_anno(os.path.join(DATA_DIR, "v66.1240K.aadr.PUB.anno"))

    roman = filter_roman_samples(anno)
    print(f"Total samples: {len(anno)}")
    print(f"Roman-period Mediterranean samples: {len(roman)}")
    print(roman.iloc[:, [0, 15, 14]].head(10))