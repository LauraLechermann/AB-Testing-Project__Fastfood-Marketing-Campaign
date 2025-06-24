# A/B Testing Project - Fast Food Marketing Campaign

![image](https://github.com/user-attachments/assets/42408990-3933-4008-ad69-dc27f9cc0569)

## Introduction

This project contains the analysis (A/B Tests) of the Fast Food Marketing Campaign dataset, downloaded from:
https://www.kaggle.com/datasets/chebotinaa/fast-food-marketing-campaign-ab-test/code as a `WA_Marketing-Campaign.csv` file.

This dataset includes multivariate testing results for a fast-food chain planning to add a new item to its menu. The company tested three different promotional strategies at various locations to determine which marketing approach would most effectively drive sales of the new product.

## Dataset Information

The dataset contains information about sales of the new menu item at various store locations under different promotional strategies. The variables are:

* <ins>MarketID:</ins> Unique identifier for market
* <ins>MarketSize:</ins> Size of market area by sales
* <ins>LocationID:</ins> Unique identifier for store location
* <ins>AgeOfStore:</ins> Age of store in years
* <ins>Promotion:</ins> One of three promotions that were tested
* <ins>week:</ins> One of four weeks when the promotions were run
* <ins>SalesInThousands:</ins> Sales amount for a specific LocationID, Promotion, and week

Each location was randomly assigned to one of the three promotional strategies.

## 🧪 A/B Testing Objective and Hypothesis

The primary goal of this analysis is to determine which promotional strategy most effectively drives sales of the new menu item. Specifically, we aim to answer:

* <ins>1. Overall Sales:</ins> Which promotion generates the highest average sales?
* <ins>2. Consistency:</ins> Which promotion delivers the most reliable sales performance across different locations?
* <ins>3. Market Impact:</ins> Do specific promotions work better in particular market sizes or store types?

**Null Hypothesis (H₀):** There is no significant difference in sales performance among the three promotional strategies (μ₁ = μ₂ = μ₃).
**Alternative Hypothesis (H₁):** At least one promotion produces significantly different sales performance compared to the others (at least one μᵢ ≠ μⱼ).

## 🚀 Analytical Plan

**1. Data Exploration, Cleaning and Preparation:**

  * Handle missing values, zeros, and outliers
  * Ensure proper distribution of promotions across locations
  * Prepare data structures for analysis <br />

**2. Descriptive Statistics and Visualization:**

  * Analyze sales distributions by promotion
  * Examine sales trends over the four-week period
  * Compare performance across different market sizes and store ages <br />

**3. Hypothesis Testing:**

  * Use one-way ANOVA to test for overall differences among promotions
  * Conduct pairwise t-tests to identify specific differences between promotions
  * Check for sample ratio mismatch to validate randomization
  * Calculate confidence intervals (both analytical and bootstrap)
  * Assess statistical power for detecting meaningful differences <br />

**4. Advanced Analyses:**

  * Control for store characteristics using multivariate regression
  * Evaluate treatment effects compared to baseline promotion
  * Analyze sales consistency using coefficient of variation
  * Examine weekly growth rates to assess sales momentum <br />

**5. Visualization and Dashboard:**

  * Create visualizations for sales by promotion and week
  * Display confidence intervals and statistical significance
  * Build interactive dashboard for stakeholder communication <br />

**6. Decision and Recommendation:**

- Clearly interpret what the statistical results imply for the business
- Provide specific, actionable recommendations based on the data
- <ins>Support the decision with visualizations showing:</ins>
  - Differences in sales between promotional strategies
  - Confidence intervals to express uncertainty
  - Statistical significance of the findings

## Project Files and Structure

* `marketing_campaign_functions.py`: This is a Python module containing all the visualization and analysis functions you'll need. This keeps your code organized and reusable.
* `Fast_Food_AB_Test_Analysis.ipynb`: A complete Jupyter notebook for performing the analysis with proper statistical testing and visualization.
* `Fast_Food_AB_Test_Dashboard.ipynb`: A notebook focused on creating interactive visualizations and dashboards.


## Prerequisites

* Python 3.x
* Required Python packages:
  * pandas
  * numpy
  * seaborn
  * matplotlib
  * statsmodel
  * plotly
* Jupyter Notebook


## Requirements

### Installation Instructions and Cloning the Repository

Follow these steps to set up the project environment and install the required dependencies:

1. Clone the repository:
    ```bash
    git clone https://github.com/LauraLechermann/AB-Testing-Project__Fastfood-Marketing-Campaign.git
    ```
2. Navigate to the project directories:
    ```bash
    cd Fast_Food_AB_Test_Analysis
    ```
    ```bash
    cd Fast_Food_AB_Test_Dashboard
    ```
3. (Optional) Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
5. You are now ready to run the project!
   
7. Follow these steps to open and run the Jupyter Notebook:
   
   Start the Jupyter Notebook by running the following command in your terminal:
   ```bash
     jupyter notebook Fast_Food_AB_Test_Analysis.ipynb
   ```
   ```bash
     jupyter notebook Fast_Food_AB_Test_Dashboard.ipynb
   ```
 This will open Jupyter Notebook in your default web browser.


## Importing the original dataset into Jupyter Notebooks for Analysis:

* Download the Coursera Course dataset from: https://www.kaggle.com/datasets/chebotinaa/fast-food-marketing-campaign-ab-test/code as a `WA_Marketing-Campaign.csv` file and save the file in the same directory as the Jupyter Notebook file
* Load the dataset into a DataFrame in Jupyter Notebooks with the necessary packages:
```bash
import pandas as pd
# Load dataset into Pandas DataFrame
df = pd.read_csv('WA_Marketing-Campaign.csv')

#Display the first 5 rows of the dataset
df.head()
```
* Proceed with data inspection looking for duplicates, missing values and outliers before proceeding with the exploratory data analysis (EDA)


## Visualizations and Interactive Dashboard

**1. Visualizations**

The Jupyter Notebook contains visualizations and graphs plotted with funtions that can be found in a separate `marketing_campaign_functions.py` file. Each visualization function is called separately in the Jupyter Notebook file, e.g. when visualizing the distribution of players in each variant:

```bash
viz.plot_distributions(df, 'SalesInThousands', 'Promotion')
```
When running the Jupyter Notebook file, make sure the `marketing_campaign_functions.py` function is in the same directory as the Jupyter Notebook file to run the analysis successfully!

**2. Interactive Dashboard**

* Ensure you have the Fast Food Marketing Campaign dataset (`WA_Marketing-Campaign.csv`) and the functions file (marketing_campaign_functions.py) are in the same directory as the notebook.
* Open the notebook in Jupyter and run all cells to generate interactive visualizations.
* Make sure to import the `marketing_campaign_functions.py` file and the `WA_Marketing-Campaign.csv` data file alongside other packages to be able to run the notebook file:
```bash
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import marketing_campaign_functions as viz

df = pd.read_csv('WA_Marketing-Campaign.csv')
```

**Interacting with Visualizations:**

* Hover over plot elements to see detailed information
* Use the zoom and pan tools to focus on specific areas
* Click on legend items to toggle visibility of data series
* Use the "Download plot as PNG" button to save visualizations

**Sharing Results:** An HTML version of the dashboard is automatically saved as `fast_food_dashboard.html` which can be shared with stakeholders who don't have Jupyter installed. <br />
**Customization:** Modify the visualization parameters in the functions file if you want to adjust the appearance or behavior of the dashboard.
