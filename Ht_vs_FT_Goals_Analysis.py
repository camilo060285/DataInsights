import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# List of CSV files representing each season
base_path = 'C:/Users/34633/OneDrive/Desktop/soccer match prediction system/matches_system_overview/season_fixed_shorted/'
seasons = ['2019_20_fix', '2020_21_fix', '2021_22_fix', '2022_23_fix', '2023_24']
csv_files = [f'{base_path}S_{season}.csv' for season in seasons]

def load_and_prepare_data(file_path):
    """
    Load the data from a CSV file and prepare it by renaming columns and calculating total goals.
    """
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

def perform_eda(df):
    """
    Perform exploratory data analysis by plotting the distribution of first half and total goals.
    """
    sns.histplot(df['HT_Total_Goals'], kde=True)
    plt.title('Distribution of First Half Goals')
    plt.show()

    sns.histplot(df['FT_Total_Goals'], kde=True)
    plt.title('Distribution of Total Goals')
    plt.show()

def correlation_analysis(df):
    """
    Calculate and print the correlation between first half goals and total goals.
    """
    correlation = df['HT_Total_Goals'].corr(df['FT_Total_Goals'])
    print(f"Correlation between first half goals and total goals: {correlation}")

def regression_analysis(df):
    """
    Perform a linear regression analysis to model the relationship between first half goals and total goals.
    """
    X = df['HT_Total_Goals']
    y = df['FT_Total_Goals']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    print(model.summary())

def visualize_relationship(df):
    """
    Visualize the relationship between first half goals and total goals using a scatter plot with a regression line.
    """
    sns.regplot(x='HT_Total_Goals', y='FT_Total_Goals', data=df)
    plt.title('First Half Goals vs. Total Goals')
    plt.xlabel('First Half Goals')
    plt.ylabel('Total Goals')
    plt.show()

def ab_testing(df):
    """
    Set up and perform A/B/n testing to compare different predictive models.
    """
    # Scenario A: Using only first half goals
    X_A = df[['HT_Total_Goals']]
    y_A = df['FT_Total_Goals']
    X_A = sm.add_constant(X_A)
    model_A = sm.OLS(y_A, X_A).fit()
    print("Scenario A:")
    print(model_A.summary())

    # Scenario B: Using first half goals and full-time home goals
    X_B = df[['HT_Total_Goals', 'Full-time Home Goals']]
    y_B = df['FT_Total_Goals']
    X_B = sm.add_constant(X_B)
    model_B = sm.OLS(y_B, X_B).fit()
    print("Scenario B:")
    print(model_B.summary())

    # Scenario C: Using first half goals, full-time home goals, and full-time away goals
    X_C = df[['HT_Total_Goals', 'Full-time Home Goals', 'Full-time Away Goals']]
    y_C = df['FT_Total_Goals']
    X_C = sm.add_constant(X_C)
    model_C = sm.OLS(y_C, X_C).fit()
    print("Scenario C:")
    print(model_C.summary())

def main():
    """
    Main function to load data, perform EDA, correlation analysis, regression analysis, visualization, and A/B/n testing.
    """
    combined_df = pd.DataFrame()
    
    for file_path in csv_files:
        print(f"Processing file: {file_path}")
        df = load_and_prepare_data(file_path)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    perform_eda(combined_df)  # Perform EDA on combined data
    correlation_analysis(combined_df)
    regression_analysis(combined_df)
    visualize_relationship(combined_df)  # Visualize relationship on combined data
    ab_testing(combined_df)

if __name__ == "__main__":
    main()
