# Analyzing Selling Price of Used Cars

This project explores the factors that influence used car prices using Python-based data analysis and visualization.

## Project Overview

The notebook performs:
- Data loading and column naming
- Basic data inspection (`shape`, `describe`, `info`)
- Data cleaning (handling missing/invalid values, converting types)
- Feature engineering (unit conversion, normalization, binning, one-hot encoding)
- Exploratory data analysis (boxplots, scatter plots, grouped statistics, heatmaps)
- Statistical testing (ANOVA)

Main notebook:
- `analyzing_selling_price_used_cars.ipynb`

## Dataset

Files included:
- `data.csv` - primary dataset used in the notebook
- `imports-85.data.txt` - source/reference dataset file

The dataset includes common automobile attributes such as:
- make, fuel type, body style
- engine size, horsepower, curb weight
- city/highway mileage
- price (target variable)

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- SciPy
- Jupyter Notebook

## How to Run

1. Clone this repository.
2. Open the project folder.
3. Install dependencies:
   - `pip install numpy pandas matplotlib seaborn scipy notebook`
4. Launch Jupyter:
   - `jupyter notebook`
5. Open and run:
   - `analyzing_selling_price_used_cars.ipynb`

## Key Analysis Steps

- Assign meaningful headers to raw dataset columns
- Convert and clean the `price` column
- Transform `city-mpg` to liters per 100 km
- Normalize `length`, `width`, and `height`
- Bin prices into `Low`, `Medium`, and `High`
- Visualize relationships between features and price
- Compare groups (e.g., by `make`, `drive-wheels`, `body-style`)
- Apply ANOVA to evaluate statistical differences

## Sample Insights (from EDA)

- Engine size shows a positive relationship with car price.
- Drive wheel type and body style show noticeable price differences.
- Grouped and pivoted summaries make category-based comparisons easier.

## Repository Structure

- `analyzing_selling_price_used_cars.ipynb`
- `data.csv`
- `imports-85.data.txt`
- `README.md`

## Author

Fatah Rahimi
