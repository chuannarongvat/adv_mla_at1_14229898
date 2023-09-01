import pandas as pd

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

def unique_value(df, df_name):
    cat_cols = [feature for feature in df.columns if df[feature].dtypes=='O']
    
    print(df_name)
    print(f'Categorical Columns: {cat_cols}')
    
    for i in cat_cols:
        print(f"{i}: {df[i].nunique()}")