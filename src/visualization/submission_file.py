import pandas as pd

def submission_file(y_test_prob):
    player_id_df = pd.read_csv('../data/processed/player_id.csv')
    
    submission_df = pd.DataFrame({
        'player_id': player_id_df['player_id'],
        'drafted': y_test_prob
    })
    
    submission_df['drafted'] = submission_df['drafted'].round(1)
    return submission_df