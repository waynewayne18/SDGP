import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from functools import reduce
from algo import Algo

st.set_page_config(page_title="Bristol-Pink Dashboard", layout="wide")

# ── Colour palettes ───────────────────────────────────────────────────────────
# Fixed colours per product so every chart uses the same visual identity
COLOR_MAP = {
    "Americano": "#A0522D",
    "Cappuccino": "#C0A080",
    "Croissant":  "#E8A850",
}

# Each training window gets its own colour on the comparison chart
WEEK_COLORS = {
    4: "#ff7eb9",
    5: "#7eb9ff",
    6: "#7effa0",
    7: "#ffb07e",
    8: "#c77eff",
}

# ── Page styling ──────────────────────────────────────────────────────────────
# Dark theme applied globally via CSS injection
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    [data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    h1, h2, h3 { color: #e8789c !important; }
    </style>
""", unsafe_allow_html=True)

# ── File-to-product mapping ───────────────────────────────────────────────────
# Tells the app which products live in which CSV file.
# If a new product CSV is added later, register it here.
FILE_PRODUCT_MAP = {
    "Coffee_Sales.csv":    ["Americano", "Cappuccino"],
    "Croissant_Sales.csv": ["Croissant"],
}


def get_files():
    # Scan the working directory for CSV files and build a display-name -> filename map
    raw = [f for f in os.listdir(".") if f.endswith(".csv")]
    return {f.replace(".csv", "").replace("_", " ").title(): f for f in raw}


file_dict = get_files()

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image(
    "https://via.placeholder.com/150x50/161b22/ff7eb9?text=Bristol-Pink",
    use_container_width=True
)
st.sidebar.title("Data Selection")

# Checkboxes let the user pick which data sources to include
st.sidebar.write("**Data Sources**")
selected = []
for name in file_dict:
    if st.sidebar.checkbox(name, value=True):
        selected.append(name)

st.sidebar.divider()

# Training window controls how many weeks of data are used as the test set
st.sidebar.write("**Training Window**")
training_weeks = st.sidebar.select_slider(
    "Weeks", options=[4, 5, 6, 7, 8], value=6, label_visibility="collapsed"
)
#Forecast length controls how far ahead the model forecasts
st.sidebar.write("**Forecast Length**")
forecasting_for = st.sidebar.select_slider(
    "Weeks", options=[4, 5, 6, 7, 8, 9, 10, 11, 12], value=8, label_visibility="collapsed"
)

st.sidebar.divider()

# Month filter — only affects the historical charts, not the forecast
st.sidebar.write("**Month**")
MONTHS = {
    "All":            None,
    "March 2025":     ("2025-03-01", "2025-03-31"),
    "April 2025":     ("2025-04-01", "2025-04-30"),
    "May 2025":       ("2025-05-01", "2025-05-31"),
    "June 2025":      ("2025-06-01", "2025-06-30"),
    "July 2025":      ("2025-07-01", "2025-07-31"),
    "August 2025":    ("2025-08-01", "2025-08-31"),
    "September 2025": ("2025-09-01", "2025-09-30"),
    "October 2025":   ("2025-10-01", "2025-10-31"),
}
selected_month = st.sidebar.radio(
    "Month", options=list(MONTHS.keys()), index=0, label_visibility="collapsed"
)
# ─────────────────────────────────────────────────────────────────────────────


# ── Model loading ─────────────────────────────────────────────────────────────
# @st.cache_resource keeps the trained model in memory.
# The model is only re-trained when training_weeks changes — not on every page reload.
@st.cache_resource
def load_algo(weeks):
    algo = Algo(forecasting_for, training_weeks=weeks)
    maes = algo.Predictor()   # trains the model and returns MAE per product
    return algo, maes


algo, maes = load_algo(training_weeks)
# ─────────────────────────────────────────────────────────────────────────────

# Build the list of products to display based on which files are selected
active_products = []
for name, filename in file_dict.items():
    if name in selected:
        active_products.extend(FILE_PRODUCT_MAP.get(filename, []))
active_products = sorted(set(active_products))

# ── Main page ─────────────────────────────────────────────────────────────────
st.title("Bristol-Pink — Sales Dashboard")

if not active_products:
    st.warning("Please select a data source from the sidebar to view metrics.")
else:
    tab_analysis, tab_forecast, tab_compare, tab_model = st.tabs(
        ["Market Analysis", "Sales Forecast", "Comparison", "Model Performance"]
    )

    # ── Data loaders ─────────────────────────────────────────────────────────
    # @st.cache_data stores the parsed dataframe so the CSV is only read once
    @st.cache_data
    def load_coffee():
        df = pd.read_csv("Coffee_Sales.csv", skiprows=1)
        df.columns = ["Date", "Cappuccino", "Americano"]
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        return df

    @st.cache_data
    def load_croissant():
        df = pd.read_csv("Croissant_Sales.csv")
        df.columns = ["Date", "Croissant"]
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
        return df

    # Collect only the frames for the selected data sources
    frames = []
    for name, filename in file_dict.items():
        if name in selected:
            if filename == "Coffee_Sales.csv":
                frames.append(load_coffee())
            elif filename == "Croissant_Sales.csv":
                frames.append(load_croissant())

    # Merge all frames on Date so every product shares the same timeline
    df = reduce(lambda l, r: pd.merge(l, r, on="Date", how="outer"), frames)
    df = df.sort_values("Date").reset_index(drop=True)

    # Keep only products that actually exist in the merged dataframe
    active_products = [p for p in active_products if p in df.columns]

    # Apply month filter to the historical dataframe only
    # The forecast is not affected — it always runs from the last available date
    month_range = MONTHS[selected_month]
    if month_range:
        start = pd.Timestamp(month_range[0])
        end   = pd.Timestamp(month_range[1])
        df = df[(df["Date"] >= start) & (df["Date"] <= end)].reset_index(drop=True)

    # ── Market Analysis tab ───────────────────────────────────────────────────
    with tab_analysis:
        st.header("Historical Sales Overview")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Volume Mix")
            # Sum total units per product for the selected period
            summary = df[active_products].sum().reset_index()
            summary.columns = ["Product", "Units"]
            fig_pie = px.pie(
                summary, values="Units", names="Product", hole=0.5,
                color="Product", color_discrete_map=COLOR_MAP
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.subheader("Daily Trends")
            # Line chart showing day-by-day sales for each active product
            fig_line = px.line(
                df, x="Date", y=active_products, markers=True,
                color_discrete_map=COLOR_MAP
            )
            fig_line.update_layout(template="plotly_dark", hovermode="x unified")
            st.plotly_chart(fig_line, use_container_width=True)

    # ── Sales Forecast tab ────────────────────────────────────────────────────
    with tab_forecast:
        st.header("Sales Forecast")
        # The forecast always starts from the last date in the dataset (17 Oct 2025).
        # Restricting it to the selected month would reduce training data and hurt accuracy.
        st.caption("Forecast always starts from the last available date in the dataset.")

        target = st.radio("Select Product", options=active_products, horizontal=True)
        forecast_days = forecasting_for * 7

        # Cache forecast results so the model does not re-run on every interaction
        @st.cache_data
        def get_forecast(forecasting_for, weeks):
            return algo.forecast(forecasting_for)

        forecast_df = get_forecast(forecasting_for, training_weeks)

        # Filter to the selected product and rename columns for display
        pred_df = forecast_df[forecast_df["product"] == target].rename(
            columns={"date": "Date", "forecast": "Predicted_Sales"}
        )

        v_graph, v_table = st.tabs(["Forecast", "Data Table"])

        with v_graph:
            st.subheader(f"{forecast_days}-Day Forecast — {target}")
            fig_pred = px.line(pred_df, x="Date", y="Predicted_Sales")
            fig_pred.update_traces(
                line_color=COLOR_MAP.get(target), line_width=4
            )
            fig_pred.update_layout(template="plotly_dark", dragmode="zoom")
            st.plotly_chart(fig_pred, use_container_width=True)

        with v_table:
            st.subheader(f"{target} — Sales Data")
            st.write("Recent Sales")
            st.dataframe(df[["Date", target]].tail(10), use_container_width=True)
            st.write("Forecast")
            st.dataframe(
                pred_df[["Date", "Predicted_Sales"]], use_container_width=True
            )

    # ── Comparison tab ────────────────────────────────────────────────────────
    with tab_compare:
        st.header("Training Window Comparison")
        st.caption(
            "Select training windows to compare their forecasts on the same chart."
        )

        compare_target = st.radio(
            "Select Product", options=active_products,
            horizontal=True, key="compare_product"
        )

        # Checkboxes instead of a text input — user can only pick from the 5 valid options
        st.write("**Select Training Windows**")
        cols = st.columns(5)
        selected_weeks = []
        defaults = {4: True, 5: False, 6: True, 7: False, 8: True}
        for i, w in enumerate([4, 5, 6, 7, 8]):
            if cols[i].checkbox(f"{w} weeks", value=defaults[w], key=f"cmp_week_{w}"):
                selected_weeks.append(w)
        

        if not selected_weeks:
            st.warning("Please select at least one training window.")
        else:
            fig_compare = go.Figure()

            for weeks in selected_weeks:
                # Each window has its own cached model — no re-training needed
                @st.cache_data
                def get_comparison_forecast(forecasting_for, w):
                    a, _ = load_algo(w)
                    return a.forecast(forecasting_for, days=w * 7, )

                w_forecast = get_comparison_forecast(forecasting_for, weeks)
                w_pred = w_forecast[w_forecast["product"] == compare_target]

                # Add one line per training window to the same figure
                fig_compare.add_trace(go.Scatter(
                    x=w_pred["date"],
                    y=w_pred["forecast"],
                    mode="lines",
                    name=f"{weeks} weeks",
                    line=dict(color=WEEK_COLORS[weeks], width=2)
                ))

            fig_compare.update_layout(
                template="plotly_dark",
                hovermode="x unified",
                xaxis_title="Date",
                yaxis_title="Predicted Sales",
                legend_title="Training Window",
            )
            st.plotly_chart(fig_compare, use_container_width=True)

            # Show MAE for each selected window so accuracy can be compared directly
            st.subheader("MAE by Training Window")
            translator = {"Croissant": 0, "Cappuccino": 1, "Americano": 2}
            product_idx = translator.get(compare_target, 0)

            rows = []
            for weeks in selected_weeks:
                _, w_maes = load_algo(weeks)
                rows.append({
                    "Training Window": f"{weeks} weeks",
                    "Forecast Days":   forecasting_for * 7,
                    "MAE":             f"{w_maes[product_idx]:.2f}",
                })
            st.table(pd.DataFrame(rows))

    # ── Model Performance tab ─────────────────────────────────────────────────
    with tab_model:
        st.header("Model Performance")
        # MAE (Mean Absolute Error) shows how many units the model is off on average.
        # Lower is better. Results update when the training window changes.
        st.caption(
            f"Training window: {training_weeks} weeks  |  "
            f"Test set: last {training_weeks * 7} days"
        )

        product_names = ["Croissant", "Cappuccino", "Americano"]
        metrics = pd.DataFrame({
            "Algorithm": ["XGBoost"] * 3,
            "Product":   product_names,
            "MAE":       [f"{m:.2f}" for m in maes],
        })
        st.table(metrics)

st.markdown("---")
st.caption("Bristol-Pink Bakery © 2025")