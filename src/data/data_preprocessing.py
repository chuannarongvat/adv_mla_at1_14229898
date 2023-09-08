import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class DataPreprocessing:
    def __init__(self, df):
        self.df = df

    def missing_value(df):
        missing_values = df.isna().sum()
        missing_percentage = (missing_values / len(df)) * 100

        missing_data = pd.DataFrame({
        'Column': missing_values.index,
        'Missing Values': missing_values.values,
        'Missing Percentage': missing_percentage.values
        })

        missing_data = missing_data[missing_data['Missing Values'] > 0]
        missing_data = missing_data.sort_values(by='Missing Percentage', ascending=False)

        print("Columns with missing values:")
        print(missing_data)
        return list(missing_data['Column'])

    def unique_value(df, df_name):
        cat_cols = [feature for feature in df.columns if df[feature].dtypes=='O']
    
        print(df_name)
        print(f'Categorical Columns: {cat_cols}')
    
        for i in cat_cols:
            print(f"{i}: {df[i].nunique()}")

    def target_cols(df, target_col):
        palette = 'ch:.25'

        sns.countplot(data=df, x=target_col, palette=palette)
        plt.show()

    def fill_missing_with_value(df, col, target_col ,fill_value):
        df.loc[~df[col].isna(), target_col] = fill_value

    def drop_column(df, col):
        df.drop(col, axis=1, inplace=True)

    def extract_num_cat_cols(df):
        cat_cols = [feature for feature in df if df[feature].dtypes=='O']
        num_cols = list(set(df.columns) - set(cat_cols))
        return num_cols, cat_cols
    
    def convert_height(ht):
        if pd.notna(ht):
            if ht in ['-', '0']:
                return ht
            
            if 'Apr' in ht:
                feet = '4'
            elif 'May' in ht:
                feet = '5'   
            elif 'Jun' in ht:
                feet = '6'
            elif 'Jul' in ht:
                feet = '7'
            else:
                feet = '0'
            inches = ''.join(filter(str.isdigit, ht))
            
            return f"{feet}'{inches}"
        return ht
    
    def classify_height(height):
        if pd.isna(height) or height == '0' or height == '-' or height == 'nan':
            return 'Short'
        
        feet, inches = height.split("'")
        total_height = int(feet) * 12 + int(inches)
        if total_height >= 84:
            return 'Very Tall'
        elif total_height >= 78:
            return 'Tall'
        elif total_height >= 72:
            return 'Medium'
        else:
            return 'Short'