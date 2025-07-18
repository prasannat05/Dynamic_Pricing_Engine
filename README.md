Dynamic Price Prediction

🔧 Dynamic Price Prediction System helps Amazon sellers set smart, profitable prices.
🛒 Scrapes live competitor data (price, ratings, reviews) from Amazon.
📈 Uses Google Trends to analyze and forecast product demand.
🤖 Predicts base price using XGBoost based on product features.
📊 ARIMA forecasts future demand to refine pricing strategy.
⚙️ Applies heuristics for stock levels and product age adjustments.
💰 Ensures prices stay above cost while maximizing profit.
🌐 Built with Python, Flask, BeautifulSoup, XGBoost, ARIMA, PyTrends
🚀 Future scope: LSTM/Prophet models, seasonal pricing, and user dashboard.

Project Structure 

dynamic-price-prediction/
├── system.py               # 🧠 Main backend Flask application
├── style.css               # 🎨 CSS styling for frontend UI
├── templates/              # 📄 HTML templates for rendering pages
│   ├── index.html          # 🔧 Input page (user form)
│   ├── result.html         # 📊 Result page with predicted price
│   └── docs.html           # 📝 Additional documentation or help page

