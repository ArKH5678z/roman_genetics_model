import pandas as pd
import os

DATA_DIR = "/home/grace-matiba/projects/roman_genetics_model/data"

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

def assign_subpopulation(group_id):
    group = str(group_id)
    if any(x in group for x in ['Italy', 'Sicily', 'Sardinia', 'Croatia']):
        return 'Italian_Central_Med'
    elif any(x in group for x in ['Turkey', 'Byzantine', 'Greece', 'Lebanon',
                                   'Syria', 'Israel', 'Jordan', 'Egypt']):
        return 'Eastern_Med'
    elif any(x in group for x in ['England', 'Spain', 'France', 'Germany',
                                   'Britain', 'Scotland', 'Romania', 'Bulgaria']):
        return 'Western_European'
    else:
        return None
def aggregate_subpopulations(roman_df):
    group_col = 'Group ID'
    lat_col = 'Latitude'
    lon_col = 'Longitude'
    date_col = 'Date mean in BP in years before 1950 CE [OxCal mu for a direct radiocarbon date, and average of range for a contextual date]'

    roman_df = roman_df.copy()
    roman_df['subpopulation'] = roman_df[group_col].apply(assign_subpopulation)
    roman_df = roman_df[roman_df['subpopulation'].notna()]

    roman_df['date_CE'] = 1950 - pd.to_numeric(roman_df[date_col], errors='coerce')
    roman_df[lat_col] = pd.to_numeric(roman_df[lat_col], errors='coerce')
    roman_df[lon_col] = pd.to_numeric(roman_df[lon_col], errors='coerce')

    summary = roman_df.groupby('subpopulation').agg(
        sample_count=(group_col, 'count'),
        mean_date_CE=('date_CE', 'mean'),
        mean_lat=(lat_col, 'mean'),
        mean_lon=(lon_col, 'mean')
    ).reset_index()

    return roman_df, summary

if __name__ == "__main__":
    ind = load_ind(os.path.join(DATA_DIR, "v66.1240K.aadr.PUB.ind"))
    anno = load_anno(os.path.join(DATA_DIR, "v66.1240K.aadr.PUB.anno"))

    roman = filter_roman_samples(anno)
    print(f"Total samples: {len(anno)}")
    print(f"Roman-period Mediterranean samples: {len(roman)}")

    roman_labelled, summary = aggregate_subpopulations(roman)

    print("\nSubpopulation summary:")
    print(summary)

    roman.to_csv(os.path.join(DATA_DIR, "roman_filtered.csv"), index=False)
    roman_labelled.to_csv(os.path.join(DATA_DIR, "roman_labelled.csv"), index=False)
    summary.to_csv(os.path.join(DATA_DIR, "subpopulation_summary.csv"), index=False)
    print("\nSaved all files")