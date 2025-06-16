"""
Utility functions for A/B testing analysis of the Fast Food Marketing Campaign
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.stats.api as sms
from statsmodels.stats.proportion import proportions_ztest
from typing import Tuple, List, Dict, Optional, Union
import warnings
warnings.filterwarnings('ignore')




def check_zeros_and_outliers(df: pd.DataFrame, numeric_cols: List[str]) -> None:
    """
    Check for zeros and potential outliers in numeric columns
    """
    for col in numeric_cols:
        zeros_count = (df[col] == 0).sum()
        print(f"\nColumn {col}:")
        print(f"  - Zeros count: {zeros_count} ({zeros_count/len(df):.2%} of data)")

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        print(f"  - Potential outliers: {len(outliers)} ({len(outliers)/len(df):.2%} of data)")



        

def plot_distributions(df: pd.DataFrame, target_col: str, group_col: str) -> None:
    """
    Plot distributions of the target metric across different groups
    
    Args:
        df: DataFrame with data
        target_col: Column name for the target metric
        group_col: Column name for the grouping variable
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    for group in sorted(df[group_col].unique()):
        subset = df[df[group_col] == group]
        sns.histplot(subset[target_col], label=f'{group_col}={group}', 
                     kde=True, ax=axes[0], alpha=0.5)
    
    axes[0].set_title(f'Distribution of {target_col} by {group_col}')
    axes[0].legend()
    
    sns.boxplot(x=group_col, y=target_col, data=df, ax=axes[1])
    axes[1].set_title(f'Boxplot of {target_col} by {group_col}')
    
    plt.tight_layout()
    plt.show()



    

def plot_target_by_week(df: pd.DataFrame, target_col: str, group_col: str, time_col: str = 'week') -> None:
    """
    Plot target metric over time, grouped by the test variants
    
    Args:
        df: DataFrame with data
        target_col: Column name for the target metric
        group_col: Column name for the grouping variable
        time_col: Column name for the time variable
    """

    grouped = df.groupby([time_col, group_col])[target_col].agg(['mean', 'count', 'std']).reset_index()
    grouped['se'] = grouped['std'] / np.sqrt(grouped['count'])
    grouped['ci_lower'] = grouped['mean'] - 1.96 * grouped['se']
    grouped['ci_upper'] = grouped['mean'] + 1.96 * grouped['se']
    
    plt.figure(figsize=(12, 6))
    
    for group in sorted(df[group_col].unique()):
        subset = grouped[grouped[group_col] == group]
        plt.plot(subset[time_col], subset['mean'], marker='o', label=f'{group_col}={group}')
        plt.fill_between(subset[time_col], subset['ci_lower'], subset['ci_upper'], alpha=0.2)
    
    plt.title(f'{target_col} by {time_col} and {group_col}')
    plt.xlabel(time_col)
    plt.ylabel(f'Mean {target_col}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()






def analyze_outliers(df, column='SalesInThousands'):
    """
    Analyze outliers in a specified column
    
    Args:
        df: DataFrame with data
        column: Column to analyze for outliers
    
    Returns:
        DataFrame with outlier rows
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    print(f"Outlier analysis for {column}:")
    print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"  Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
    print(f"  Found {len(outliers)} outliers ({len(outliers)/len(df):.2%} of data)")
    
    return outliers



    

def summarize_zeros(df, numeric_cols):
    """
    Summarize zero values in numeric columns
    
    Args:
        df: DataFrame with data
        numeric_cols: List of numeric columns to check
    
    Returns:
        DataFrame with zero counts summary
    """
    zeros_summary = pd.DataFrame({
        'Column': numeric_cols,
        'Zero Count': [df[col].eq(0).sum() for col in numeric_cols],
        'Zero Percentage': [df[col].eq(0).sum() / len(df) * 100 for col in numeric_cols]
    })
    
    return zeros_summary




    

def plot_univariate_numerical(df, column, bins=20):
    """
    Plot univariate distribution of a numerical variable
    
    Args:
        df: DataFrame with data
        column: Numerical column to plot
        bins: Number of bins for histogram
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], kde=True, bins=bins)
    plt.title(f'Distribution of {column}')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()





def analyze_weekly_growth(df, target_col='SalesInThousands', group_col='Promotion'):
    """
    Calculate and visualize weekly growth rates
    
    Args:
        df: DataFrame with data
        target_col: Target column name
        group_col: Group column name
    """

    weekly_avg = df.groupby(['week', group_col])[target_col].mean().reset_index()
    weekly_avg = weekly_avg.sort_values([group_col, 'week'])
    

    growth_data = []
    for group in sorted(df[group_col].unique()):
        group_data = weekly_avg[weekly_avg[group_col] == group]
        
        for i in range(1, len(group_data)):
            current_week = group_data.iloc[i]['week']
            current_sales = group_data.iloc[i][target_col]
            prev_sales = group_data.iloc[i-1][target_col]
            
            growth_rate = ((current_sales - prev_sales) / prev_sales) * 100
            
            growth_data.append({
                group_col: group,
                'Week': current_week,
                'Growth_Rate': growth_rate
            })
    
    growth_df = pd.DataFrame(growth_data)
    

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 
    plt.figure(figsize=(12, 6))
    
    for i, group in enumerate(sorted(df[group_col].unique())):
        subset = growth_df[growth_df[group_col] == group]
        plt.plot(subset['Week'], subset['Growth_Rate'], marker='o', 
                 label=f'{group_col} {group}', linewidth=2, color=colors[i % len(colors)])
    
    plt.title('Weekly Sales Growth Rate by Promotion', fontsize=16)
    plt.xlabel('Week', fontsize=12)
    plt.ylabel('Growth Rate (%)', fontsize=12)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(2, 5))
    plt.tight_layout()
    plt.show()
    
 
    avg_growth = growth_df.groupby(group_col)['Growth_Rate'].mean().reset_index()
    print(f"Average weekly growth rate by {group_col}:")
    display(avg_growth)
    
    return growth_df, avg_growth







def analyze_sales_consistency(df, target_col='SalesInThousands', group_col='Promotion'):
    """
    Analyze sales consistency using coefficient of variation
    
    Args:
        df: DataFrame with data
        target_col: Target column name
        group_col: Group column name
    
    Returns:
        DataFrame with consistency metrics
    """
    consistency_data = df.groupby(group_col)[target_col].agg(['mean', 'std']).reset_index()
    consistency_data['cv'] = (consistency_data['std'] / consistency_data['mean']) * 100  # CV as percentage
    
    plt.figure(figsize=(10, 6))
    
    x_positions = np.array(consistency_data[group_col])
    
    bars = plt.bar(x_positions, consistency_data['cv'], width=0.4, 
                  color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    plt.title('Sales Consistency by Promotion', fontsize=16)
    plt.xlabel(group_col, fontsize=12)
    plt.ylabel('Coefficient of Variation (%)', fontsize=12)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tick_params(labelsize=11)
    

    plt.xticks(x_positions)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=11)
    
    plt.figtext(0.5, 0.01, "Note: Lower coefficient of variation indicates more consistent sales performance.", 
                ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.show()
    
    print("Sales consistency analysis (coefficient of variation):")
    print("Promotion 1: {:.1f}%".format(consistency_data.loc[consistency_data[group_col]==1, 'cv'].values[0]))
    print("Promotion 2: {:.1f}%".format(consistency_data.loc[consistency_data[group_col]==2, 'cv'].values[0]))
    print("Promotion 3: {:.1f}%".format(consistency_data.loc[consistency_data[group_col]==3, 'cv'].values[0]))
    print("\nLower values indicate more consistent (less variable) sales across locations.")
    
    return consistency_data




def check_sample_ratio_mismatch(df: pd.DataFrame, group_col: str) -> Tuple[float, float]:
    """
    Check for sample ratio mismatch using chi-square test
    
    Args:
        df: DataFrame with data
        group_col: Column containing the group assignments
        
    Returns:
        tuple: (chi2 statistic, p-value)
    """
    observed = df[group_col].value_counts().values
    n = len(df)
    k = len(observed)
    expected = np.array([n/k] * k)
    
    chi2_stat, p_value = stats.chisquare(observed, expected)
    
    print(f"Sample sizes per group:")
    print(df[group_col].value_counts())
    print(f"\nChi-square test for sample ratio mismatch:")
    print(f"Chi-square statistic: {chi2_stat:.4f}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("WARNING: Significant sample ratio mismatch detected (p < 0.05)")
    else:
        print("No significant sample ratio mismatch detected")
    
    return chi2_stat, p_value




    

def run_t_test(df: pd.DataFrame, target_col: str, group_col: str, group1: str, group2: str) -> Tuple[float, float, float, float]:
    """
    Run t-test between two groups
    
    Args:
        df: DataFrame with data
        target_col: Column name for the target metric
        group_col: Column name for the grouping variable
        group1: First group value
        group2: Second group value
        
    Returns:
        tuple: (mean difference, t-statistic, p-value, cohen's d effect size)
    """
    group1_data = df[df[group_col] == group1][target_col]
    group2_data = df[df[group_col] == group2][target_col]
    
    mean1, mean2 = group1_data.mean(), group2_data.mean()
    mean_diff = mean2 - mean1
    

    t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
    
    # Calculate effect size (Cohen's d)
    n1, n2 = len(group1_data), len(group2_data)
    var1, var2 = group1_data.var(), group2_data.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohen_d = mean_diff / pooled_std
    
    print(f"T-test between {group1} and {group2}:")
    print(f"  Mean {target_col} for {group1}: {mean1:.4f}")
    print(f"  Mean {target_col} for {group2}: {mean2:.4f}")
    print(f"  Mean difference ({group2} - {group1}): {mean_diff:.4f}")
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Cohen's d effect size: {cohen_d:.4f}")
    
    return mean_diff, t_stat, p_value, cohen_d





def run_anova(df: pd.DataFrame, target_col: str, group_col: str) -> Tuple[float, float]:
    """
    Run one-way ANOVA test across all groups
    
    Args:
        df: DataFrame with data
        target_col: Column name for the target metric
        group_col: Column name for the grouping variable
        
    Returns:
        tuple: (F-statistic, p-value)
    """
    groups = [df[df[group_col] == group][target_col].values for group in df[group_col].unique()]
    f_stat, p_value = stats.f_oneway(*groups)
    
    print("One-way ANOVA results:")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("  Significant difference detected among groups (p < 0.05)")
    else:
        print("  No significant difference detected among groups")
    
    return f_stat, p_value




def calculate_power(df: pd.DataFrame, target_col: str, group_col: str, effect_size: float = None) -> tuple:
    """
    Calculate the statistical power of the test
    
    Args:
        df: DataFrame with data
        target_col: Column name for the target metric
        group_col: Column name for the grouping variable
        effect_size: Cohen's d effect size (if None, will be calculated)
        
    Returns:
        tuple: (power, effect_size, min_sample_size)
    """
    from statsmodels.stats.power import TTestIndPower
    
    groups = sorted(df[group_col].unique())
    

    if effect_size is None:
        effect_sizes = []
        for i, group1 in enumerate(groups):
            for group2 in groups[i+1:]:
                group1_data = df[df[group_col] == group1][target_col]
                group2_data = df[df[group_col] == group2][target_col]
                
                # Calculate Cohen's d
                mean1, mean2 = group1_data.mean(), group2_data.mean()
                pooled_std = np.sqrt((group1_data.var() + group2_data.var()) / 2)
                d = abs(mean1 - mean2) / pooled_std
                effect_sizes.append(d)
        
        effect_size = min(effect_sizes) 
    

    sample_sizes = df.groupby(group_col)[target_col].count().values
    min_n = min(sample_sizes)
    
    power_analysis = TTestIndPower()
    power = power_analysis.power(effect_size=effect_size, nobs1=min_n, alpha=0.05)
    
    print(f"  Minimum effect size (Cohen's d): {effect_size:.4f}")
    print(f"  Minimum sample size per group: {min_n}")
    print(f"  Statistical power: {power:.4f}")

    if power < 0.8:
        print("  WARNING: The statistical power is below the recommended threshold of 0.8")
        
 
        required_n = power_analysis.solve_power(effect_size=effect_size, power=0.8, alpha=0.05)
        print(f"  Required sample size per group for 0.8 power: {required_n:.1f}")
    else:
        print("  The statistical power is adequate (≥ 0.8)")
    
    return power, effect_size, min_n





    

def calculate_confidence_interval_analytical(df: pd.DataFrame, target_col: str, group_col: str, 
                                         group_value: Union[str, int], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval analytically
    
    Args:
        df: DataFrame with data
        target_col: Column name for the target metric
        group_col: Column name for the grouping variable
        group_value: The value in group_col to calculate CI for
        confidence: Confidence level (default: 0.95)
        
    Returns:
        tuple: (lower bound, upper bound)
    """
    group_data = df[df[group_col] == group_value][target_col]
    

    ci = sms.DescrStatsW(group_data).tconfint_mean(alpha=1-confidence)
    
    print(f"Analytical {confidence*100:.0f}% confidence interval for {group_col}={group_value}:")
    print(f"  Mean: {group_data.mean():.4f}")
    print(f"  CI: ({ci[0]:.4f}, {ci[1]:.4f})")
    
    return ci





def calculate_confidence_interval_bootstrap(df: pd.DataFrame, target_col: str, group_col: str, 
                                        group_value: Union[str, int], n_bootstrap: int = 5000, 
                                        confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence interval using bootstrap method
    
    Args:
        df: DataFrame with data
        target_col: Column name for the target metric
        group_col: Column name for the grouping variable
        group_value: The value in group_col to calculate CI for
        n_bootstrap: Number of bootstrap samples (default: 5000)
        confidence: Confidence level (default: 0.95)
        
    Returns:
        tuple: (lower bound, upper bound)
    """
    group_data = df[df[group_col] == group_value][target_col]
    
   
    bootstrap_means = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(group_data, size=len(group_data), replace=True)
        bootstrap_means.append(bootstrap_sample.mean())
    
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 - (1 - confidence) / 2) * 100
    ci_lower = np.percentile(bootstrap_means, lower_percentile)
    ci_upper = np.percentile(bootstrap_means, upper_percentile)
    
    print(f"Bootstrap {confidence*100:.0f}% confidence interval for {group_col}={group_value} ({n_bootstrap} samples):")
    print(f"  Mean: {group_data.mean():.4f}")
    print(f"  CI: ({ci_lower:.4f}, {ci_upper:.4f})")
    
    return ci_lower, ci_upper


    

def plot_confidence_intervals(df: pd.DataFrame, target_col: str, group_col: str, 
                           confidence: float = 0.95, method: str = 'analytical') -> None:
    """
    Plot confidence intervals for each group
    
    Args:
        df: DataFrame with data
        target_col: Column name for the target metric
        group_col: Column name for the grouping variable
        confidence: Confidence level (default: 0.95)
        method: Method for CI calculation ('analytical' or 'bootstrap')
    """
    groups = sorted(df[group_col].unique())
    means = []
    ci_lowers = []
    ci_uppers = []
    
    for group in groups:
        group_data = df[df[group_col] == group][target_col]
        means.append(group_data.mean())
        
        if method == 'analytical':
            ci = sms.DescrStatsW(group_data).tconfint_mean(alpha=1-confidence)
            ci_lowers.append(ci[0])
            ci_uppers.append(ci[1])
        else:  
            bootstrap_means = []
            for _ in range(5000):  # Default to 5000 bootstrap samples
                bootstrap_sample = np.random.choice(group_data, size=len(group_data), replace=True)
                bootstrap_means.append(bootstrap_sample.mean())
            
            lower_percentile = (1 - confidence) / 2 * 100
            upper_percentile = (1 - (1 - confidence) / 2) * 100
            ci_lowers.append(np.percentile(bootstrap_means, lower_percentile))
            ci_uppers.append(np.percentile(bootstrap_means, upper_percentile))
    

    groups_str = [str(g) for g in groups]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 
    
    plt.figure(figsize=(10, 6))
    
    bars = plt.bar(groups_str, means, alpha=0.7, width=0.5, 
                  color=[colors[i] for i in range(len(groups))])
    
    plt.errorbar(groups_str, means, yerr=[np.array(means) - np.array(ci_lowers), 
                                       np.array(ci_uppers) - np.array(means)], 
               fmt='o', color='black', capsize=5)
    

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ci_text = f"[{ci_lowers[i]:.2f}, {ci_uppers[i]:.2f}]"
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f"{means[i]:.2f}\n{ci_text}", ha='center', va='bottom', fontsize=10)
    
    plt.title(f'{target_col} by {group_col} with {confidence*100:.0f}% Confidence Intervals ({method})')
    plt.xlabel(group_col)
    plt.ylabel(f'Mean {target_col}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()





def plot_treatment_effects(treatment_effects):
    """
    Plot treatment effects with improved labeling and formatting
    
    Args:
        treatment_effects: DataFrame with treatment effect data
    """
    plt.figure(figsize=(12, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  
    

    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3, label='No Effect')
    

    for i, row in treatment_effects.iterrows():
        promo = row['Promotion']
        effect = row['Effect']
        ci_lower = row['CI Lower']
        ci_upper = row['CI Upper']
        
        plt.plot([promo], [effect], 'o', color=colors[i % len(colors)], markersize=10,
                label=f'Promotion {promo}')

        plt.vlines(x=promo, ymin=ci_lower, ymax=ci_upper, color=colors[i % len(colors)], linewidth=2)
        
        plt.hlines(y=ci_lower, xmin=promo-0.05, xmax=promo+0.05, color=colors[i % len(colors)], linewidth=2)
        plt.hlines(y=ci_upper, xmin=promo-0.05, xmax=promo+0.05, color=colors[i % len(colors)], linewidth=2)
        
        plt.text(promo+0.1, effect, f"Effect: {effect:.2f}", 
                 va='center', ha='left', fontsize=10, color=colors[i % len(colors)])
        plt.text(promo+0.1, ci_upper, f"Upper CI: {ci_upper:.2f}", 
                 va='bottom', ha='left', fontsize=9, color=colors[i % len(colors)])
        plt.text(promo+0.1, ci_lower, f"Lower CI: {ci_lower:.2f}", 
                 va='top', ha='left', fontsize=9, color=colors[i % len(colors)])
        
        if row['Significant']:
            sig_text = "Significant"
        else:
            sig_text = "Not Significant"
            
        plt.text(promo, ci_lower - 0.8, sig_text,
                ha='center', va='top', fontsize=10,
                color=colors[i % len(colors)], weight='bold',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    plt.title('Treatment Effects with 95% Confidence Intervals\n(relative to Promotion 1)', fontsize=14)
    plt.xlabel('Promotion', fontsize=12)
    plt.ylabel('Effect on Sales (in thousands)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.legend(loc='lower right', fontsize=10, framealpha=0.9)
    
    plt.xticks(treatment_effects['Promotion'])
    
    min_y = min(treatment_effects['CI Lower'].min() - 2.0, -14.5)  # Ensure enough space for labels
    max_y = max(1.0, treatment_effects['CI Upper'].max() + 0.5)
    plt.ylim(min_y, max_y)
    
    plt.tight_layout()
    plt.show()



    

def get_pairwise_t_tests(df: pd.DataFrame, target_col: str, group_col: str) -> pd.DataFrame:
    """
    Perform pairwise t-tests between all groups
    
    Args:
        df: DataFrame with data
        target_col: Column name for the target metric
        group_col: Column name for the grouping variable
        
    Returns:
        DataFrame with test results
    """
    groups = sorted(df[group_col].unique())
    results = []
    
    for i, group1 in enumerate(groups):
        for group2 in groups[i+1:]:
            group1_data = df[df[group_col] == group1][target_col]
            group2_data = df[df[group_col] == group2][target_col]
            
            mean1, mean2 = group1_data.mean(), group2_data.mean()
            mean_diff = mean2 - mean1
            

            t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
            
            # Calculate effect size (Cohen's d)
            n1, n2 = len(group1_data), len(group2_data)
            var1, var2 = group1_data.var(), group2_data.var()
            pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            cohen_d = mean_diff / pooled_std
            
            results.append({
                'Group 1': group1,
                'Group 2': group2,
                'Mean 1': mean1,
                'Mean 2': mean2,
                'Mean Difference': mean_diff,
                't-statistic': t_stat,
                'p-value': p_value,
                "Cohen's d": cohen_d,
                'Significant': p_value < 0.05
            })
    
    return pd.DataFrame(results)




def run_multivariate_analysis(df, target_col='SalesInThousands', group_col='Promotion', control_cols=None):
    """
    Run multivariate regression to control for store characteristics
    
    Args:
        df: DataFrame with data
        target_col: Target column name
        group_col: Group column name
        control_cols: List of control variables
    """
    import statsmodels.formula.api as smf
    
    if control_cols is None:
        control_cols = ['AgeOfStore', 'MarketSize']
    
    # Convert Market Size to numerical variable for regression if it's categorical
    df_copy = df.copy()
    if 'MarketSize' in control_cols and df_copy['MarketSize'].dtype == 'object':
        # Create dummy variables for market size
        df_copy = pd.get_dummies(df_copy, columns=['MarketSize'], drop_first=True)
        control_cols = [col for col in control_cols if col != 'MarketSize'] + [col for col in df_copy.columns if 'MarketSize_' in col]
    
    formula = f"{target_col} ~ C({group_col})"
    for col in control_cols:
        formula += f" + {col}"
    
 
    model = smf.ols(formula, data=df_copy).fit()
    
  
    print("Regression Analysis (controlling for store characteristics):")
    print(f"Formula: {formula}")
    print("\nCoefficient Summary:")
    

    print("\nPromotion Effects (compared to baseline):")
    for var in model.params.index:
        if f"C({group_col})" in var and 'Intercept' not in var:
            promo = var.split('[')[1].split(']')[0].split('.')[0]
            coef = model.params[var]
            stderr = model.bse[var]
            pvalue = model.pvalues[var]
            signif = "***" if pvalue < 0.001 else "**" if pvalue < 0.01 else "*" if pvalue < 0.05 else ""
            
            print(f"Promotion {promo}: {coef:.4f} (SE: {stderr:.4f}, p-value: {pvalue:.4f}) {signif}")
    

    print("\nControl Variable Effects:")
    for var in model.params.index:
        if f"C({group_col})" not in var and 'Intercept' not in var:
            coef = model.params[var]
            stderr = model.bse[var]
            pvalue = model.pvalues[var]
            signif = "***" if pvalue < 0.001 else "**" if pvalue < 0.01 else "*" if pvalue < 0.05 else ""
            
            print(f"{var}: {coef:.4f} (SE: {stderr:.4f}, p-value: {pvalue:.4f}) {signif}")
    

    print(f"\nModel R-squared: {model.rsquared:.4f}")
    print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
    print(f"F-statistic: {model.fvalue:.4f}")
    print(f"Prob (F-statistic): {model.f_pvalue:.4f}")
    

    print("\nSignificance codes: *** p<0.001, ** p<0.01, * p<0.05")
    
    return model






def analyze_weekly_effects(df, target_col='SalesInThousands', group_col='Promotion', time_col='week'):
    """
    Run repeated measures analysis to test for time effects and interactions
    
    Args:
        df: DataFrame with data
        target_col: Target column name
        group_col: Group column name
        time_col: Time column name
    """
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    
    print("Repeated Measures Analysis (Week × Promotion):")
    
    formula = f"{target_col} ~ C({group_col}) * C({time_col})"
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    print(f"Formula: {formula}")
    print("\nANOVA Table:")
    display(anova_table)
    
    print("\nInterpretation:")
    p_promotion = anova_table.loc[f'C({group_col})', 'PR(>F)']
    p_week = anova_table.loc[f'C({time_col})', 'PR(>F)']
    p_interaction = anova_table.loc[f'C({group_col}):C({time_col})', 'PR(>F)']
    
    if p_promotion < 0.05:
        print(f"- The effect of {group_col} is statistically significant (p={p_promotion:.4f})")
    else:
        print(f"- No significant effect of {group_col} detected (p={p_promotion:.4f})")
        
    if p_week < 0.05:
        print(f"- The effect of {time_col} is statistically significant (p={p_week:.4f})")
    else:
        print(f"- No significant effect of {time_col} detected (p={p_week:.4f})")
        
    if p_interaction < 0.05:
        print(f"- The interaction between {group_col} and {time_col} is significant (p={p_interaction:.4f})")
        print(f"  This means the effectiveness of promotions changed over time")
    else:
        print(f"- No significant interaction between {group_col} and {time_col} (p={p_interaction:.4f})")
        print(f"  This means the relative effectiveness of promotions remained consistent over time")
    
    plt.figure(figsize=(12, 6))
    

    week_promo_means = df.groupby([time_col, group_col])[target_col].mean().reset_index()
    
    for promo in sorted(df[group_col].unique()):
        subset = week_promo_means[week_promo_means[group_col] == promo]
        plt.plot(subset[time_col], subset[target_col], marker='o', linewidth=2, 
                 label=f'{group_col} {promo}')
    
    plt.title(f'Interaction between {time_col.capitalize()} and {group_col}')
    plt.xlabel(time_col.capitalize())
    plt.ylabel(f'Mean {target_col}')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return model, anova_table






def create_summary_table(df, anova_results, pairwise_results, power_analysis, treatment_effects):
    """
    Create a summary table of all key statistical results
    
    Args:
        df: DataFrame with data
        anova_results: Tuple with (f_stat, p_value) from ANOVA
        pairwise_results: DataFrame with pairwise t-test results
        power_analysis: Tuple with (power, effect_size, min_n)
        treatment_effects: DataFrame with treatment effects
    
    Returns:
        DataFrame with summary results
    """
    summary = {
        'Test': [],
        'Description': [],
        'Result': [],
        'Significance': []
    }
    

    f_stat, p_value = anova_results
    summary['Test'].append('One-way ANOVA')
    summary['Description'].append('Tests if any significant difference exists among promotions')
    summary['Result'].append(f'F={f_stat:.4f}, p={p_value:.4f}')
    summary['Significance'].append('Significant' if p_value < 0.05 else 'Not Significant')
    

    for _, row in pairwise_results.iterrows():
        summary['Test'].append(f'T-test: Promo {int(row["Group 1"])} vs. Promo {int(row["Group 2"])}')
        summary['Description'].append(f'Compares means between Promo {int(row["Group 1"])} and Promo {int(row["Group 2"])}')
        summary['Result'].append(f't={row["t-statistic"]:.4f}, p={row["p-value"]:.4f}, d={row["Cohen\'s d"]:.4f}')
        summary['Significance'].append('Significant' if row["p-value"] < 0.05 else 'Not Significant')
    
    power, effect_size, min_n = power_analysis
    summary['Test'].append('Power Analysis')
    summary['Description'].append('Tests if sample size is adequate for detecting effects')
    summary['Result'].append(f'Power={power:.4f}, Min Effect Size={effect_size:.4f}')
    summary['Significance'].append('Adequate Power' if power >= 0.8 else 'Inadequate Power')
    

    for _, row in treatment_effects.iterrows():
        summary['Test'].append(f'Treatment Effect: Promo {int(row["Promotion"])} vs. Promo {int(row["Control"])}')
        summary['Description'].append(f'Measures causal impact of Promo {int(row["Promotion"])} vs. Promo {int(row["Control"])}')
        summary['Result'].append(f'Effect={row["Effect"]:.4f}, %Lift={row["Percent Lift"]:.2f}%')
        summary['Significance'].append('Significant' if row["Significant"] else 'Not Significant')
    
    summary_df = pd.DataFrame(summary)
    
    return summary_df


#####################
#####################
##DASHBOARD FUNCYIONS 
#####################




def create_dashboard_plots(df):
    """
    Create the main plots for the Fast Food Marketing Campaign dashboard
    
    Args:
        df: DataFrame with the campaign data
        
    Returns:
        tuple: (weekly_trends_fig, sales_distribution_fig, treatment_effects_fig)
    """
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    weekly_data = df.groupby(['week', 'Promotion'])['SalesInThousands'].agg(['mean', 'count', 'std']).reset_index()
    weekly_data['se'] = weekly_data['std'] / np.sqrt(weekly_data['count'])
    weekly_data['ci_lower'] = weekly_data['mean'] - 1.96 * weekly_data['se']
    weekly_data['ci_upper'] = weekly_data['mean'] + 1.96 * weekly_data['se']
    
    # 1. Weekly trends plot
    weekly_fig = go.Figure()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    
    for i, promo in enumerate(sorted(df['Promotion'].unique())):
        subset = weekly_data[weekly_data['Promotion'] == promo]
        
        weekly_fig.add_trace(go.Scatter(
            x=subset['week'],
            y=subset['mean'],
            mode='lines+markers',
            name=f'Promotion {promo}',
            line=dict(color=colors[i], width=3),
            marker=dict(size=10, line=dict(width=2, color='DarkSlateGrey'))
        ))
        
        weekly_fig.add_trace(go.Scatter(
            x=np.concatenate([subset['week'], subset['week'][::-1]]),
            y=np.concatenate([subset['ci_upper'], subset['ci_lower'][::-1]]),
            fill='toself',
            fillcolor=f'rgba{tuple(int(colors[i][1:][j:j+2], 16) for j in (0, 2, 4)) + (0.13,)}',
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    weekly_fig.update_layout(
        title='Average Sales by Week and Promotion',
        xaxis=dict(
            title='Week',
            tickmode='array',
            tickvals=[1, 2, 3, 4],
            ticktext=['Week 1', 'Week 2', 'Week 3', 'Week 4'],
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title='Average Sales (in thousands)',
            gridcolor='lightgray'
        ),
        plot_bgcolor='white',
        hovermode='x unified',
        legend=dict(y=0.99, x=0.01, orientation='h'),
        height=500,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # 2. Sales distribution by promotion
    dist_fig = go.Figure()
    
    for i, promo in enumerate(sorted(df['Promotion'].unique())):
        subset = df[df['Promotion'] == promo]
        
        dist_fig.add_trace(go.Box(
            y=subset['SalesInThousands'],
            name=f'Promotion {promo}',
            marker_color=colors[i],
            boxmean=True
        ))
    
    dist_fig.update_layout(
        title='Sales Distribution by Promotion',
        yaxis=dict(
            title='Sales (in thousands)',
            gridcolor='lightgray'
        ),
        plot_bgcolor='white',
        height=500,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # 3. Treatment effects
    # Calculate treatment effects data
    control_group = 1  # Promotion 1 as baseline
    treatment_effects = []
    for promo in sorted([p for p in df['Promotion'].unique() if p != control_group]):
        control_data = df[df['Promotion'] == control_group]['SalesInThousands']
        treatment_data = df[df['Promotion'] == promo]['SalesInThousands']
        
        mean_control = control_data.mean()
        mean_treatment = treatment_data.mean()
        effect = mean_treatment - mean_control
        
        se_control = control_data.std() / np.sqrt(len(control_data))
        se_treatment = treatment_data.std() / np.sqrt(len(treatment_data))
        se_diff = np.sqrt(se_control**2 + se_treatment**2)
        
        ci_lower = effect - 1.96 * se_diff
        ci_upper = effect + 1.96 * se_diff
        
        percent_lift = (effect / mean_control) * 100
        
        treatment_effects.append({
            'Promotion': promo,
            'Effect': effect,
            'CI Lower': ci_lower,
            'CI Upper': ci_upper,
            'Percent Lift': percent_lift,
            'Significant': (ci_lower > 0) or (ci_upper < 0)
        })
    
    effects_fig = go.Figure()
    
    for i, effect in enumerate(treatment_effects):
        promo = effect['Promotion']
        color = colors[i+1]  # Skip the control color
        
        effects_fig.add_trace(go.Scatter(
            x=[f'Promotion {promo}'],
            y=[effect['Effect']],
            mode='markers',
            name=f'Promotion {promo}',
            marker=dict(color=color, size=12),
            error_y=dict(
                type='data',
                array=[effect['CI Upper'] - effect['Effect']],
                arrayminus=[effect['Effect'] - effect['CI Lower']],
                visible=True,
                color=color,
                thickness=2,
                width=6
            )
        ))
        
        effects_fig.add_annotation(
            x=f'Promotion {promo}',
            y=effect['Effect'],
            text=f"{effect['Effect']:.2f}",
            showarrow=False,
            yshift=15,
            font=dict(color=color, size=12, family="Arial Black")
        )
        
        # Add significance label
        sig_text = "Significant" if effect['Significant'] else "Not Significant"
        effects_fig.add_annotation(
            x=f'Promotion {promo}',
            y=effect['CI Lower'],
            text=sig_text,
            showarrow=False,
            yshift=-20,
            font=dict(color=color, size=10)
        )
    
    effects_fig.add_shape(
        type="line",
        x0=-0.5,
        x1=1.5,
        y0=0,
        y1=0,
        line=dict(color="red", width=2, dash="dash")
    )
    
    effects_fig.update_layout(
        title='Treatment Effects vs. Control (Promotion 1)',
        yaxis=dict(
            title='Effect on Sales (in thousands)',
            gridcolor='lightgray',
            zeroline=False
        ),
        plot_bgcolor='white',
        height=500,
        margin=dict(l=50, r=50, t=80, b=80)
    )
    
    return weekly_fig, dist_fig, effects_fig

    

def create_combined_dashboard(df):
    """
    Create a combined dashboard with multiple visualizations
    
    Args:
        df: DataFrame with the campaign data
        
    Returns:
        fig: Combined dashboard figure
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    
    weekly_fig, dist_fig, effects_fig = create_dashboard_plots(df)
    
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{'colspan': 2}, None],
            [{'type': 'box'}, {'type': 'scatter'}]
        ],
        subplot_titles=(
            'Average Sales by Week and Promotion',
            'Sales Distribution by Promotion',
            'Treatment Effect vs. Control (Promotion 1)'
        ),
        vertical_spacing=0.12
    )
    
    for trace in weekly_fig.data:
        fig.add_trace(trace, row=1, col=1)
    
    for trace in dist_fig.data:
        fig.add_trace(trace, row=2, col=1)
    
    for trace in effects_fig.data:
        fig.add_trace(trace, row=2, col=2)
    
    fig.add_shape(
        type="line",
        x0=-0.5, x1=1.5,
        y0=0, y1=0,
        line=dict(color="red", width=2, dash="dash"),
        row=2, col=2
    )
    
    fig.update_layout(
        title='Fast Food Marketing Campaign A/B Test Dashboard',
        height=900,
        width=1200,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        plot_bgcolor='white'
    )
    
    fig.update_xaxes(
        title_text='Week',
        tickmode='array',
        tickvals=[1, 2, 3, 4],
        ticktext=['Week 1', 'Week 2', 'Week 3', 'Week 4'],
        gridcolor='lightgray',
        row=1, col=1
    )
    
    fig.update_xaxes(
        title_text='Promotion',
        gridcolor='lightgray',
        row=2, col=1
    )
    
    fig.update_xaxes(
        title_text='Promotion',
        gridcolor='lightgray',
        row=2, col=2
    )
    
    # Update y-axes
    fig.update_yaxes(
        title_text='Average Sales (in thousands)',
        gridcolor='lightgray',
        row=1, col=1
    )
    
    fig.update_yaxes(
        title_text='Sales (in thousands)',
        gridcolor='lightgray',
        row=2, col=1
    )
    
    fig.update_yaxes(
        title_text='Effect vs. Control (in thousands)',
        gridcolor='lightgray',
        row=2, col=2
    )
    

    fig.add_annotation(
        x=0.5, y=-0.15,
        xref="paper", yref="paper",
        text="Dashboard Interpretation: Promotion 1 and 3 consistently outperform Promotion 2 across all weeks. The boxplot shows significantly lower sales for Promotion 2, while Promotion 1 has the highest median sales. Treatment effect analysis confirms that Promotion 2 delivers significantly lower sales compared to Promotion 1, while the difference between Promotion 1 and 3 is not statistically significant.",
        showarrow=False,
        font=dict(size=12),
        bgcolor="rgba(255, 255, 255, 0.8)",
        bordercolor="black",
        borderwidth=1,
        borderpad=4
    )
    
    return fig
