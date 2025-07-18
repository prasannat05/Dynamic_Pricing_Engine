from flask import Flask, render_template, request, Response, jsonify
import requests
from bs4 import BeautifulSoup
import time, random, re, math, warnings, io, csv
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from pytrends.request import TrendReq
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

app = Flask(__name__)

# Global variables for data sharing between routes
last_competitor_df = None
last_result_dict = None
last_trend_series = None
last_forecast_series = None

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36",
]

def get_amazon_categories(product_name):
    if not product_name.strip():
        return []
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
    }
    url = f"https://www.amazon.in/s?k={product_name.replace(' ', '+')}"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return []
    except Exception:
        return []
    soup = BeautifulSoup(response.text, 'html.parser')

    categories = []
    sidebar = soup.find('div', id='departments')
    if sidebar:
        for li in sidebar.find_all('li'):
            a = li.find('a', href=True)
            if a and a.text.strip():
                categories.append((a.text.strip(), a['href']))
    return categories

def scrape_amazon_competitors(product_name, category_url_suffix=None, max_pages=3):
    products = []
    base_url = f"https://www.amazon.in{category_url_suffix}" if category_url_suffix else f"https://www.amazon.in/s?k={product_name.replace(' ', '+')}"

    for page in range(1, max_pages + 1):
        if '?' in base_url:
            url = f"{base_url}&page={page}"
        else:
            url = f"{base_url}?page={page}"

        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept-Language": "en-US,en;q=0.9",
        }

        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200 or "captcha" in response.text.lower():
                print(f"Failed to retrieve page {page}. Status: {response.status_code}")
                continue
        except Exception as e:
            print(f"Error fetching page {page}: {e}")
            continue

        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.select('[data-component-type="s-search-result"]')

        if not results:
            print(f"No results on page {page}")
            continue

        for item in results:
            try:
                title = item.h2.text.strip()

                # Price - improved error handling
                price_tag = item.select_one('.a-price-whole')
                if not price_tag:
                    continue
                price_text = price_tag.text.strip().replace(',', '')
                if not price_text.isdigit():
                    continue
                price = int(price_text)

                # Rating
                rating = None
                review_count = 0
                rating_tag = item.select_one('.a-icon-alt')
                if rating_tag:
                    rating_text = rating_tag.text.strip()
                    rating_match = re.search(r'(\d+(\.\d+)?)', rating_text)
                    if rating_match:
                        rating = float(rating_match.group(1))

                    # Try to get review count from the next sibling or nearby tag
                    review_tag = item.find('span', {'class': 'a-size-base s-underline-text'})
                    if not review_tag:
                        review_tag = item.find('a', {'class': 'a-size-base'})

                    if review_tag and review_tag.text.strip().replace(',', '').isdigit():
                        review_count = int(review_tag.text.strip().replace(',', ''))

                # Monthly sales
                sales_text = item.get_text(separator=" ")
                monthly_sales = 0
                match = re.search(r'(\d[\d,]*)\+?\s+bought in past month', sales_text)
                if match:
                    monthly_sales = int(match.group(1).replace(',', ''))

                products.append({
                    'title': title,
                    'price': price,
                    'rating': rating,
                    'review_count': review_count,
                    'monthly_sales': monthly_sales
                })

            except Exception as e:
                print(f"Error processing product: {e}")
                continue

        time.sleep(random.uniform(2, 4))  # Be kind to Amazon servers

    return pd.DataFrame(products)

def get_trend_scores(keyword):
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:91.0) Gecko/20100101 Firefox/91.0"
    ]
    session = requests.Session()
    session.headers.update({
        'User-Agent': random.choice(USER_AGENTS),
        'Accept-Language': 'en-US,en;q=0.9'
    })

    pytrends = TrendReq(hl='en-US', tz=330)
    pytrends.session = session

    try:
        time.sleep(random.uniform(1.5, 3.0))
        suggestions = pytrends.suggestions(keyword)

        topic_mid = None
        for suggestion in suggestions:
            if suggestion.get("type") == "Topic":
                topic_mid = suggestion.get("mid")
                break

        if topic_mid:
            pytrends.build_payload([topic_mid], cat=0, timeframe='today 3-m', geo='IN')
        else:
            pytrends.build_payload([keyword], cat=0, timeframe='today 3-m', geo='IN')

        trend_data = pytrends.interest_over_time()
        if not trend_data.empty:
            return trend_data.iloc[:, 0].tolist()

        raise RuntimeError("Google Trends returned empty data")

    except Exception as e:
        if "429" in str(e) or "Too Many Requests" in str(e):
            raise RuntimeError("Google Trends request failed with 429")
        raise RuntimeError(f"Google Trends error: {str(e)}")

def parse_csv_trend_data(file_stream):

    try:
        # Read the CSV file
        df = pd.read_csv(file_stream)
        
        # Handle different CSV formats
        # Case 1: Single column with trend values
        if len(df.columns) == 1:
            trend_values = df.iloc[:, 0].dropna()
        # Case 2: Multiple columns, try to find the trend column
        else:
            # Look for common column names
            trend_column = None
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['trend', 'interest', 'score', 'value']):
                    trend_column = col
                    break
            
            if trend_column:
                trend_values = df[trend_column].dropna()
            else:
                # Use the second column if it exists (first might be dates)
                if len(df.columns) > 1:
                    trend_values = df.iloc[:, 1].dropna()
                else:
                    trend_values = df.iloc[:, 0].dropna()
        
        # Convert to numeric and filter out non-numeric values
        trend_values = pd.to_numeric(trend_values, errors='coerce').dropna()
        
        if len(trend_values) == 0:
            raise ValueError("No valid numeric trend values found in CSV")
        
        # Take last 30 values if more than 30, or pad with mean if less than 10
        trend_list = trend_values.tolist()
        if len(trend_list) > 30:
            trend_list = trend_list[-30:]
        elif len(trend_list) < 10:
            # Pad with mean value if too few data points
            mean_val = np.mean(trend_list)
            while len(trend_list) < 10:
                trend_list.append(mean_val)
        
        print(f"Successfully parsed {len(trend_list)} trend values from CSV")
        return trend_list
        
    except Exception as e:
        raise ValueError(f"Error parsing CSV file: {str(e)}")
    
def augment_data(data, min_records=30, factor_range=(0.7, 1.3)):
    current_len = len(data)
    if current_len >= min_records:
        return data  # No need to augment

    needed = min_records - current_len
    augmented_rows = []

    numeric_cols = ['price', 'rating', 'review_count', 'monthly_sales']
    base_data = data[numeric_cols].dropna()
    
    # Check if we have enough base data to augment
    if base_data.empty:
        return data

    for _ in range(needed):
        base_row = base_data.sample(1).iloc[0].to_dict()

        new_row = {
            'title': f"Synthetic Product {_ + 1}",
            'price': max(50, int(base_row['price'] * random.uniform(*factor_range))),
            'rating': min(5.0, max(1.0, base_row['rating'] * random.uniform(*factor_range))),
            'review_count': max(0, int(base_row['review_count'] * random.uniform(*factor_range))),
            'monthly_sales': max(0, int(base_row['monthly_sales'] * random.uniform(*factor_range))),
        }
        augmented_rows.append(new_row)

    synthetic_df = pd.DataFrame(augmented_rows)
    return pd.concat([data, synthetic_df], ignore_index=True)


@app.route('/api/categories')
def api_categories():
    product = request.args.get('product', '').strip()
    categories = get_amazon_categories(product)
    cat_list = [{"name": c[0], "url": c[1]} for c in categories]
    return jsonify(cat_list)

@app.route('/', methods=['GET', 'POST'])
def index():
    global last_competitor_df, last_result_dict, last_trend_series, last_forecast_series
    
    if request.method == 'POST':
        product_name = request.form['product_name'].strip()
        cost_price_str = request.form.get('cost_price', '').strip()
        age_days_str = request.form.get('age_days', '').strip()
        stock_quantity_str = request.form.get('stock_quantity', '').strip()
        selected_category = request.form.get('category', '')

        # Validate input parameters
        try:
            cost_price = int(cost_price_str)
            age_days = int(age_days_str)
            stock_quantity = int(stock_quantity_str)
        except ValueError:
            error = "Please enter valid numbers for Cost Price, Stock Age, and Stock Quantity."
            return render_template('index.html', categories=[], error=error,
                                   product_name=product_name,
                                   cost_price=cost_price_str,
                                   age_days=age_days_str,
                                   stock_quantity=stock_quantity_str,
                                   selected_category=selected_category)

        categories = get_amazon_categories(product_name)
        selected_cat_url = selected_category if selected_category else (categories[0][1] if categories else None)

        # Scrape competitor data
        data = scrape_amazon_competitors(product_name, selected_cat_url)
        data = data[~((data['review_count'] == 0) & (data['monthly_sales'].fillna(0) == 0))]
        
        # Remove price outliers
        if not data.empty:
            q_low = data['price'].quantile(0.01)
            q_high = data['price'].quantile(0.99)
            data = data[(data['price'] >= q_low) & (data['price'] <= q_high)]

        if data.empty:
            result = {"error": "No competitor data found for this product and category."}
            return render_template("result.html", result=result)

        # Handle trend data - try Google Trends first, then fallback to CSV
        trend_series = []
        trends_failed = False
        
        try:
            trend_series = get_trend_scores(product_name)
            print("Successfully retrieved Google Trends data.")
        except RuntimeError as e:
            print(f"[Trend Error] {e}")
            trends_failed = True
            
            # Try to get CSV file
            trend_csv_file = request.files.get("trend_csv")
            if trend_csv_file and trend_csv_file.filename:
                try:
                    # Reset file pointer to beginning
                    trend_csv_file.seek(0)
                    trend_series = parse_csv_trend_data(trend_csv_file)
                    print("Successfully used uploaded CSV trend data.")
                    trends_failed = False
                except Exception as csv_err:
                    error = f"Google Trends failed and uploaded CSV is invalid: {csv_err}"
                    return render_template("index.html", categories=categories,
                                           error=error,
                                           product_name=product_name,
                                           cost_price=cost_price_str,
                                           age_days=age_days_str,
                                           stock_quantity=stock_quantity_str,
                                           selected_category=selected_category,
                                           trends_failed=True)
            else:
                error = "Google Trends failed (possibly due to rate limit). Please upload a trend CSV file."
                return render_template("index.html", categories=categories,
                                       error=error,
                                       product_name=product_name,
                                       cost_price=cost_price_str,
                                       age_days=age_days_str,
                                       stock_quantity=stock_quantity_str,
                                       selected_category=selected_category,
                                       trends_failed=True)

        # If we still don't have trend data, create dummy data
        if not trend_series:
            print("No trend data available, using dummy values")
            trend_series = [50] * 30  # Neutral trend score

        # Calculate past trend score
        past_trend_score = np.mean(trend_series[-30:]) if trend_series else 50

        # Forecast future trends using ARIMA
        try:
            if len(trend_series) >= 10:  # Need minimum data for ARIMA
                model_arima = ARIMA(trend_series, order=(1, 1, 1))
                model_fit = model_arima.fit()
                forecast = model_fit.forecast(steps=30)
                forecasted_trend_score = np.mean(forecast)
                forecasted_trend_series = list(forecast)
            else:
                raise Exception("Insufficient data for ARIMA")
        except Exception as e:
            print(f"ARIMA forecasting failed: {e}, using trend mean")
            forecasted_trend_score = past_trend_score
            forecasted_trend_series = [past_trend_score] * 30

        # Handle missing values in competitor data
        data['rating'].fillna(data['rating'].mean(), inplace=True)
        data['monthly_sales'] = data['monthly_sales'].replace(0, np.nan)
        data['monthly_sales'].fillna(np.nan, inplace=True)
        data['review_count'].fillna(0, inplace=True)
        data['trend_score'] = past_trend_score
        # Augment data only if needed
        data = augment_data(data, min_records=100)
        valid_sales = data['monthly_sales'].dropna()
        predicted_sales = valid_sales.max() if len(valid_sales) > 0 else stock_quantity
        
        # Prepare features for price prediction
        X_price = data[['rating', 'review_count', 'monthly_sales', 'trend_score']].dropna()
        y_price = data.loc[X_price.index, 'price']

        if len(X_price) == 0:
            result = {"error": "Insufficient data for price prediction after preprocessing."}
            return render_template("result.html", result=result)
        
        y_price_log = np.log1p(y_price)  # log(1 + price)

        # Split data into train/test
        if len(X_price) > 1:
            X_train, X_test, y_train_log, y_test_log = train_test_split(X_price, y_price_log, test_size=0.2, random_state=42)
        else:
            X_train = X_test = X_price
            y_train_log = y_test_log = y_price_log

        # Train model on training data
        price_model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=20,  # Set here, not in .fit()
            random_state=42
        )

        if len(X_price) > 1:
            price_model.fit(
                X_train, y_train_log,
                eval_set=[(X_test, y_test_log)],
                verbose=False  # Optional
            )
        else:
            price_model.set_params(early_stopping_rounds=None)  # Disable early stopping if no validation
            price_model.fit(X_train, y_train_log)

        # Predict on test set and calculate metrics
        if len(X_test) > 0:
            y_pred_log_test = price_model.predict(X_test)
            y_pred_test = np.expm1(y_pred_log_test)
            y_true_test = np.expm1(y_test_log)

            rmse = math.sqrt(mean_squared_error(y_true_test, y_pred_test))
            mae = mean_absolute_error(y_true_test, y_pred_test)

            mask = y_true_test != 0
            mape = np.mean(np.abs((y_true_test[mask] - y_pred_test[mask]) / y_true_test[mask])) * 100 if mask.sum() > 0 else float('inf')
        else:
            rmse = mae = mape = 0
        print(f"RMSE (original scale): {rmse}")
        print(f"MAE (original scale): {mae}")
        print(f"MAPE: {mape}%")

        # Make final prediction using trained model
        predict_price_df = pd.DataFrame([{
            'rating': data['rating'].mean(),
            'review_count': data['review_count'].mean(),
            'monthly_sales': predicted_sales,
            'trend_score': forecasted_trend_score
        }])

        base_price_log = price_model.predict(predict_price_df)[0]
        base_price = np.expm1(base_price_log)
        base_price = max(base_price, cost_price)

        # Apply adjustments
        trend_adj = 0.05 * base_price if forecasted_trend_score > 75 else -0.05 * base_price if forecasted_trend_score < 60 else 0
        stock_adj = -0.05 * base_price if stock_quantity > 500 else 0.05 * base_price if stock_quantity < 100 else 0
        age_adj = -0.2 * base_price if age_days > 90 else 0

        final_price = base_price + trend_adj + stock_adj + age_adj
        final_price = max(final_price, cost_price + 10)

        profit_pct = ((final_price - cost_price) / cost_price) * 100 if cost_price > 0 else 0
        
        # Limit to 4 competitors max for UI
        competitor_data = data.head(4).to_dict(orient='records')
        trend_series_for_graph = trend_series[-30:] if trend_series else [50]*30

        result = {
            "product_name": product_name,
            "base_price": round(base_price, 2),
            "cost_price": cost_price,
            "adjustments": {
                "trend": round(trend_adj, 2),
                "stock": round(stock_adj, 2),
                "age": round(age_adj, 2)
            },
            "final_price": round(final_price, 2),
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "mape": round(mape, 2),
            "profit_pct": round(profit_pct, 2)
        }

        # Store data for download
        last_competitor_df = data.copy()
        last_result_dict = result
        last_trend_series = trend_series_for_graph
        last_forecast_series = forecasted_trend_series

        return render_template("result.html",
                               result=result,
                               competitor_data=competitor_data,
                               trend_series=trend_series_for_graph,
                               forecast_series=forecasted_trend_series)

    return render_template('index.html', categories=[])

@app.route('/download')
def download():
    global last_competitor_df, last_result_dict

    if last_competitor_df is None or last_result_dict is None:
        return "No data available for download", 404

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["Product Pricing Report"])
    writer.writerow([])
    for key, val in last_result_dict.items():
        if isinstance(val, dict):
            writer.writerow([key])
            for subk, subv in val.items():
                writer.writerow([subk, subv])
        else:
            writer.writerow([key, val])
    writer.writerow([])
    writer.writerow(["Competitor Data"])
    if not last_competitor_df.empty:
        writer.writerow(last_competitor_df.columns.tolist())
        for _, row in last_competitor_df.iterrows():
            writer.writerow(row.tolist())

    response = Response(output.getvalue(),
                        mimetype="text/csv",
                        headers={"Content-Disposition":"attachment;filename=pricing_report.csv"})
    return response

@app.route('/docs')
def docs():
    return render_template('docs.html')

if __name__ == '__main__':
    app.run(debug=True)

