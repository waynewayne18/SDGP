<h1> Systems development group project - Pink Bakery </h1>
A time series forecasting algorithm with an interactive UI that allows you to adjust, view and compare sales predictions over a few weeks. 

<h3>To run:</h3>
<h4>pip install</h4>
<ul>
<li>pandas</li>
<li>numpy</li>
<li>streamlit </li>
<li>plotly</li>
<li>xgboost</li>
<li>scikit-learn</li>
</ul>

<h3>To use:</h3>
<p> run python -m streamlit run app.py, if u have path configured to scripts dont use 'python -m'</p>

<h3>The problem this solves</h3>
<p>Bakeries need to prepare a certain amount of fresh goods until they expire. Too much and there is a food and expenditure wastage, too little and demand can't be met. This is why a forecasting algorithm is needed to make a prediction on how many goods will be sold on a given day. This algorithm takes in a sales history and determines predictions based on  the day of week, month, the month, the quarter and whether it is the weekday/weekend. </p>