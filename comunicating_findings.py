import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load and prepare data
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path, delimiter=';')
    df.rename(columns={
        'Div': 'Division',
        'Date': 'Date',
        'Time': 'Time',
        'HomeTeam': 'Home Team',
        'AwayTeam': 'Away Team',
        'FTHG': 'Full-time Home Goals',
        'FTAG': 'Full-time Away Goals',
        'FTR': 'Full-time Result',
        'HTHG': 'Half-time Home Goals',
        'HTAG': 'Half-time Away Goals',
        'HTR': 'Half-time Result',
        'HS': 'Home Shots',
        'AS': 'Away Shots',
        'HST': 'Home Shots on Target',
        'AST': 'Away Shots on Target',
        'HF': 'Home Fouls',
        'AF': 'Away Fouls',
        'HC': 'Home Corners',
        'AC': 'Away Corners',
        'HY': 'Home Yellow Cards',
        'AY': 'Away Yellow Cards',
        'HR': 'Home Red Cards',
        'AR': 'Away Red Cards'
    }, inplace=True)
    df['HT_Total_Goals'] = df['Half-time Home Goals'] + df['Half-time Away Goals']
    df['FT_Total_Goals'] = df['Full-time Home Goals'] + df['Full-time Away Goals']
    return df

# Perform EDA
def perform_eda(df):
    sns.histplot(df['HT_Total_Goals'], kde=True)
    plt.title('Distribution of First Half Goals')
    plt.show()

    sns.histplot(df['FT_Total_Goals'], kde=True)
    plt.title('Distribution of Total Goals')
    plt.show()

# Visualize relationship
def visualize_relationship(df):
    sns.regplot(x='HT_Total_Goals', y='FT_Total_Goals', data=df)
    plt.title('First Half Goals vs. Total Goals')
    plt.xlabel('First Half Goals')
    plt.ylabel('Total Goals')
    plt.show()

# Main function
def main():
    base_path = 'C:/Users/34633/OneDrive/Desktop/soccer match prediction system/matches_system_overview/season_fixed_shorted/'
    seasons = ['2019_20_fix', '2020_21_fix', '2021_22_fix', '2022_23_fix', '2023_24']
    csv_files = [f'{base_path}S_{season}.csv' for season in seasons]

    combined_df = pd.DataFrame()
    for file_path in csv_files:
        df = load_and_prepare_data(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)

    perform_eda(combined_df)
    visualize_relationship(combined_df)

if __name__ == "__main__":
    main()
