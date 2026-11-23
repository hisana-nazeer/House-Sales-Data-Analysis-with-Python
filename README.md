# What Makes a House Expensive? King County Home Prices

I wanted to know what actually drives house prices, and whether a fairly simple model could predict them. To find out, I worked with about 21,600 home sales from King County, Washington (the county that includes Seattle), sold between May 2014 and May 2015.

This project was the final assignment in the IBM *Data Analysis with Python* course on Coursera. I've added my own notes along the way.

---

## The short version

- **Size matters most.** Living area (`sqft_living`) has the strongest correlation with price (0.70), followed closely by the house's quality **grade** (0.67).
- **Location helps, in a quieter way.** Latitude correlates with price at 0.31, so homes further north tend to sell for more. Waterfront homes have far more extreme price outliers than other homes.
- **One feature isn't enough.** Square footage alone explains about 49% of the variation in price. With 11 features the model explains 66%, and adding polynomial terms raises that to about 70% on data the model hadn't seen.

---

## What I did

### 1. Got to know the data
I loaded the dataset and checked the data types and summary statistics. A few things stood out right away:
- `bedrooms` and `bathrooms` had missing values (13 and 10).
- One house was listed with **33 bedrooms**, even though 75% of homes have 4 or fewer. That's almost certainly a typo.
- Mean price (~$540K) is above the median (~$450K), so prices are **right-skewed**. A small number of very expensive homes pull the average up.

### 2. Cleaned it up
- Dropped the columns that don't add information (`id` and the leftover index).
- Filled in the missing bedroom and bathroom values with the column means.

### 3. Explored it
- Counted homes by number of floors. Most are 1-storey (10,680) or 2-storey (8,241).
- Used a **boxplot** to compare waterfront and non-waterfront prices.
- Used a **regression plot** to confirm that `sqft_above` is positively correlated with price.
- Ranked every feature by its correlation with price.

### 4. Built and compared models

| Model | Features | R² |
|---|---|---|
| Linear regression | longitude only | 0.0005 |
| Linear regression | `sqft_living` only | 0.49 |
| Multiple linear regression | 11 features | 0.66 |
| Pipeline: scaling → polynomial → linear | 11 features | 0.75* |
| Ridge regression (α = 0.1) | 11 features, test set | 0.65 |
| **Ridge + 2nd-order polynomial (α = 0.1)** | **11 features, test set** | **0.70** |

\*Scored on the same data it was trained on, so it's optimistic. The two Ridge models were evaluated on a held-out 15% test set (3,242 homes), and those are the numbers I'd trust.

The 11 features were: `floors`, `waterfront`, `lat`, `bedrooms`, `sqft_basement`, `view`, `bathrooms`, `sqft_living15`, `sqft_above`, `grade` and `sqft_living`.

---

## What I learned

- **Always check the test score.** The pipeline reached 0.75, but it was graded on data it had already seen. The honest number came from the train/test split.
- **Polynomial features capture curves.** Prices don't rise in a straight line with size or quality, and the squared terms picked up some of that, raising the test R² from 0.65 to 0.70.
- **Some features look useful but aren't on their own.** Longitude by itself explained almost nothing. Location probably matters through neighbourhoods, not a single coordinate.

## If I kept going

- Remove or fix the 33-bedroom house and other obvious outliers instead of leaving them in.
- Predict **log(price)** to handle the skew.
- Turn `zipcode` into neighbourhood categories instead of treating it as a number.
- Try tree-based models such as Random Forest and Gradient Boosting, and tune the Ridge `alpha` with cross-validation.

---

## Tools

Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · Jupyter Notebook

## Run it yourself

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
jupyter
