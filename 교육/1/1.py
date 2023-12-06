import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
import statsmodels.api as sm
from matplotlib import MatplotlibDeprecationWarning
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

# Import data
df = pd.read_csv("2.csv", encoding='GBK')
df = df.interpolate()
df = df.dropna()

# 2. Data type conversion
df = df.astype({'2000': float, '2001': float, '2002': float, '2003': float, '2004': float,
                '2005': float, '2006': float, '2007': float, '2008': float, '2009': float,
                '2010': float, '2011': float, '2012': float, '2013': float, '2014': float,
                '2015': float, '2016': float, '2017': float, '2018': float, '2019': float,
                '2020': float, '2021': float, '2022': float})

columns_to_clean = df.columns[2:]
# 3. Handling outliers
for column in columns_to_clean:
    # Check for outliers (e.g., using a boxplot)
    # sns.boxplot(x=df[column])
    # plt.title(f'Boxplot of {column}')
    # plt.show()

    # Define a function to remove outliers
    def remove_outliers(data):
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        data_no_outliers = data[(data >= lower_bound) & (data <= upper_bound)]
        return data_no_outliers

    # Remove outliers
    df[column] = remove_outliers(df[column])

# 4. Remove whitespace from column names
df.columns = df.columns.str.strip()
df = df.interpolate()
# 5. Create a 'Year' column
df = pd.melt(df, id_vars=['CountryCode', 'IndicatorCode'], var_name='Year', value_name='Value')
df['Year'] = df['Year'].astype(int)

# Randomly select ten countries
random_countries = random.sample(df['CountryCode'].unique().tolist(), 10)

# Select data for these countries
df_selected_countries = df[(df['CountryCode'].isin(random_countries)) & (df['IndicatorCode'] == 'SE.XPD.TOTL.GD.ZS')]

# 6. Data visualization
plt.figure(figsize=(12, 6))
sns.lineplot(x='Year', y='Value', hue='CountryCode', style='IndicatorCode', data=df_selected_countries)
plt.title('Education Indicators Over Time')
plt.show()

# Set Chinese font
plt.rcParams['axes.unicode_minus'] = False  # Resolve negative sign display issue

# Set plot style
sns.set(style="whitegrid")

# Draw a scatter plot
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Year', y='Value', hue='CountryCode', data=df_selected_countries)
plt.xlabel('Year')
plt.ylabel('Value')
plt.show()

# Select enrollment rate data for these countries
df_selected_countries_enrollment = df[(df['CountryCode'].isin(random_countries)) & (df['IndicatorCode'] == 'SE.TER.ENRR')]

# Set plot style
sns.set(style="whitegrid")

# Draw a scatter plot
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Year', y='Value', hue='CountryCode', data=df_selected_countries_enrollment)
plt.xlabel('Year')
plt.ylabel('Value')
plt.show()

# Select data containing 'SE.TER.ENRR' and 'SE.XPD.TOTL.GD.ZS'
subset_df = df[df['IndicatorCode'].isin(['SE.TER.ENRR', 'SE.XPD.TOTL.GD.ZS'])]

# Create a pivot table, grouping the data by 'CountryCode' and 'IndicatorCode'
pivot_table = subset_df.pivot_table(index='CountryCode', columns='IndicatorCode', values='Value', aggfunc='first')

# Remove rows containing NaN
cleaned_data = pivot_table.dropna()

# Calculate correlation coefficient
correlation = cleaned_data.corr().loc['SE.TER.ENRR', 'SE.XPD.TOTL.GD.ZS']

print("Correlation between government education expenditure and education level:", correlation)

# Prepare data
X = sm.add_constant(cleaned_data['SE.XPD.TOTL.GD.ZS'])  # Add intercept
y = cleaned_data['SE.TER.ENRR']

# Build a linear regression model
model = sm.OLS(y, X)

# Fit the model
results = model.fit()

# Print regression coefficients and statistics
print(results.summary())
