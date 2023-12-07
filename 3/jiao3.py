import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from matplotlib import MatplotlibDeprecationWarning
import warnings
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

# Read data
file_path = '15-20.xlsx'
df = pd.read_excel(file_path)
df.dropna(inplace=True)

# Data exploration and visualization
plt.figure(figsize=(12, 6))
# Plot enrollment trends for males and females by different education stages
for stage in ['Elementary', 'Secondary', 'Total']:
    male_column = f'{stage} Male Enrolment'
    female_column = f'{stage} Female Enrolment'

    plt.plot(df.groupby('year')[male_column].sum(), label=f'{stage} Male')
    plt.plot(df.groupby('year')[female_column].sum(), label=f'{stage} Female')

plt.title('Enrolment Trends Over Years by Gender and Stage')
plt.xlabel('Year')
plt.ylabel('Enrolment Count')
plt.xticks(df['year'].unique())   # Set the x-axis ticks to unique year values
plt.legend()
plt.show()

# Calculate t-statistics and p-values for Elementary, Secondary, and Total stages
for stage in ['Elementary', 'Secondary', 'Total']:
    male_column = f'{stage} Male Enrolment'
    female_column = f'{stage} Female Enrolment'

    t_stat, p_value = ttest_ind(df[male_column], df[female_column])

    if p_value < 0.05:
        print(f'The difference in enrollment between males and females in {stage} is statistically significant (p-value: {p_value:.4f}).')
    else:
        print(f'The difference in enrollment between males and females in {stage} is not statistically significant (p-value: {p_value:.4f}).')
