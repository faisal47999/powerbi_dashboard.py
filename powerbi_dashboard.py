# Save this in a separate Colab cell
%%writefile powerbi_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from io import BytesIO
import re

# Set page configuration
st.set_page_config(page_title="Power BI Style Dashboard", layout="wide", initial_sidebar_state="expanded")

# CSS for Power BI-like styling
st.markdown("""
<style>
    .sidebar .sidebar-content { background-color: #2e2e2e; color: white; }
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 5px; width: 100%; margin-bottom: 10px; }
    .stButton>button:hover { background-color: #45a049; }
    .main .block-container { background-color: #1e1e1e; padding: 20px; }
    .stMetric { background-color: #333333; padding: 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Function to load and validate data
def load_data(uploaded_file):
    if uploaded_file is None:
        st.warning("Please upload a CSV file to proceed.")
        return None
    try:
        df = pd.read_csv(uploaded_file)
        for col in df.columns:
            if df[col].dtype == "object":
                try:
                    df[col] = pd.to_datetime(df[col])
                except:
                    pass
        return df
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

# Function to analyze columns
def analyze_columns(df):
    column_info = {"numeric": [], "categorical": [], "date": [], "text": []}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            column_info["numeric"].append(col)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            column_info["date"].append(col)
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            unique_ratio = len(df[col].unique()) / len(df[col])
            if unique_ratio < 0.1:
                column_info["categorical"].append(col)
            else:
                column_info["text"].append(col)
    return column_info

# Function to create automatic charts
def create_auto_charts(df, column_info, filtered_df, selected_filters):
    st.subheader("Automatic Visualizations")
    cols = st.columns(2)
    chart_count = 0

    if column_info["date"] and column_info["numeric"]:
        with cols[chart_count % 2]:
            date_col = column_info["date"][0]
            num_col = column_info["numeric"][0]
            fig = px.line(
                filtered_df.groupby(date_col)[num_col].sum().reset_index(),
                x=date_col, y=num_col, title=f"{num_col} Trend Over Time",
                template="plotly_dark"
            )
            fig.update_traces(mode="lines+markers", hovertemplate="%{x}: %{y}")
            st.plotly_chart(fig, use_container_width=True, key=f"auto_line_{chart_count}")
        chart_count += 1

    if column_info["categorical"] and column_info["numeric"]:
        with cols[chart_count % 2]:
            cat_col = column_info["categorical"][0]
            num_col = column_info["numeric"][0]
            fig = px.bar(
                filtered_df.groupby(cat_col)[num_col].sum().reset_index(),
                x=cat_col, y=num_col, title=f"{num_col} by {cat_col}",
                color=cat_col, template="plotly_dark"
            )
            fig.update_traces(hovertemplate="%{x}: %{y}")
            st.plotly_chart(fig, use_container_width=True, key=f"auto_bar_{chart_count}")
        chart_count += 1

    if column_info["categorical"]:
        with cols[chart_count % 2]:
            cat_col = column_info["categorical"][0]
            pie_data = filtered_df[cat_col].value_counts().reset_index()
            pie_data.columns = [cat_col, 'count']
            fig = px.pie(
                pie_data, values='count', names=cat_col, title=f"Distribution of {cat_col}",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True, key=f"auto_pie_{chart_count}")
        chart_count += 1

    if len(column_info["numeric"]) > 1:
        with cols[chart_count % 2]:
            corr = filtered_df[column_info["numeric"]].corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.columns,
                colorscale="Viridis", showscale=True
            ))
            fig.update_layout(title="Correlation Heatmap", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True, key=f"auto_heatmap_{chart_count}")
        chart_count += 1

    if column_info["text"]:
        st.write("Text Column Analysis:")
        for text_col in column_info["text"]:
            st.write(f"{text_col}: Unique values: {len(filtered_df[text_col].unique())}, Sample: {filtered_df[text_col].head(3).tolist()}")

# Function to parse search commands
def parse_search_command(command, df, column_info):
    command = command.lower().strip()
    chart_types = ["line", "bar", "pie", "scatter", "histogram", "box", "area"]
    selected_chart = None
    x_col = None
    y_col = None
    color_col = None

    for chart in chart_types:
        if chart in command:
            selected_chart = chart
            break

    for col in df.columns:
        if col.lower() in command:
            if col in column_info["numeric"] and not y_col:
                y_col = col
            elif col in column_info["categorical"] or col in column_info["date"]:
                x_col = col
            elif col in column_info["categorical"] and "by" in command:
                color_col = col

    return selected_chart, x_col, y_col, color_col

# Function to create chart
def create_chart(df, chart_type, x_col, y_col, color_col, key_prefix="custom"):
    try:
        if chart_type == "line":
            fig = px.line(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
        elif chart_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} by {x_col}")
        elif chart_type == "pie":
            if color_col:
                fig = px.pie(df, names=color_col, values=y_col, title=f"{y_col} Distribution by {color_col}")
            else:
                st.warning("Please specify a categorical column for pie chart.")
                return None
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
        elif chart_type == "histogram":
            fig = px.histogram(df, x=x_col, y=y_col, color=color_col, title=f"Histogram of {x_col}")
        elif chart_type == "box":
            fig = px.box(df, x=x_col, y=y_col, color=color_col, title=f"Box Plot of {y_col} by {x_col}")
        elif chart_type == "area":
            fig = px.area(df, x=x_col, y=y_col, color=color_col, title=f"{y_col} vs {x_col}")
        else:
            st.warning("Invalid chart type.")
            return None

        fig.update_layout(template="plotly_dark")
        fig.update_traces(hovertemplate="%{x}: %{y}")
        return fig
    except Exception as e:
        st.warning(f"Error creating chart: {e}")
        return None

# Function to download data
def to_csv(df):
    output = BytesIO()
    df.to_csv(output, index=False)
    return output.getvalue()

# Initialize session state
if "filters" not in st.session_state:
    st.session_state.filters = {}
if "selected_chart" not in st.session_state:
    st.session_state.selected_chart = None
if "chart_params" not in st.session_state:
    st.session_state.chart_params = {}

# Sidebar
st.sidebar.header("Visuals")
st.sidebar.markdown("**Select Visualization**")
chart_types = [
    ("📈 Line", "line"),
    ("📊 Bar", "bar"),
    ("🥧 Pie", "pie"),
    ("⚡️ Scatter", "scatter"),
    ("📉 Histogram", "histogram"),
    ("📍 Box", "box"),
    ("📋 Area", "area")
]
for label, chart_type in chart_types:
    if st.sidebar.button(label, key=f"visual_{chart_type}"):
        st.session_state.selected_chart = chart_type

st.sidebar.header("Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")

# Load data
df = load_data(uploaded_file)
if df is None:
    st.error("No data loaded. Please upload a valid CSV file.")
    st.stop()

# Analyze columns
column_info = analyze_columns(df)

# Sidebar filters
st.sidebar.subheader("Filters")
if column_info["categorical"]:
    for cat_col in column_info["categorical"]:
        selected = st.sidebar.multiselect(
            f"Filter {cat_col}",
            options=df[cat_col].unique(),
            default=df[cat_col].unique(),
            key=f"filter_{cat_col}"
        )
        st.session_state.filters[cat_col] = selected
if column_info["date"]:
    date_col = column_info["date"][0]
    date_range = st.sidebar.date_input(
        "Select Date Range",
        [df[date_col].min(), df[date_col].max()],
        min_value=df[date_col].min(),
        max_value=df[date_col].max(),
        key="date_range"
    )
    st.session_state.filters["date_range"] = date_range

# Apply filters
filtered_df = df.copy()
for cat_col, selected in st.session_state.filters.items():
    if cat_col != "date_range" and selected:
        filtered_df = filtered_df[filtered_df[cat_col].isin(selected)]
if "date_range" in st.session_state.filters and st.session_state.filters["date_range"]:
    date_col = column_info["date"][0]
    filtered_df = filtered_df[
        (filtered_df[date_col].dt.date >= st.session_state.filters["date_range"][0]) &
        (filtered_df[date_col].dt.date <= st.session_state.filters["date_range"][1])
    ]

# Main dashboard
st.title("Power BI Style Dashboard")
st.markdown("Interactive dashboard with smart analysis and Power BI-like visualizations.")

# Search bar
st.subheader("Search-Based Analysis")
search_command = st.text_input(
    "Enter command (e.g., 'bar chart of sales by region')",
    key="search_bar"
)
if search_command:
    chart_type, x_col, y_col, color_col = parse_search_command(search_command, filtered_df, column_info)
    if chart_type and x_col and y_col:
        fig = create_chart(filtered_df, chart_type, x_col, y_col, color_col, key_prefix="search")
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="search_chart")
    else:
        st.warning("Invalid command. Try: 'chart_type of y_column by x_column'.")

# KPIs
st.subheader("Key Metrics")
cols = st.columns(3)
if column_info["numeric"]:
    num_col = column_info["numeric"][0]
    cols[0].metric(f"Total {num_col}", f"{filtered_df[num_col].sum():,.2f}")
    cols[1].metric(f"Average {num_col}", f"{filtered_df[num_col].mean():,.2f}")
    cols[2].metric("Total Records", f"{len(filtered_df):,}")

# Automatic charts
create_auto_charts(df, column_info, filtered_df, st.session_state.filters)

# Custom visualizations
if st.session_state.selected_chart:
    st.subheader("Selected Visualization")
    x_axis = st.selectbox("Select X-Axis", options=df.columns, key="x_axis")
    y_axis = st.selectbox("Select Y-Axis (Numeric)", options=column_info["numeric"] + ["None"], key="y_axis")
    color_col = st.selectbox("Select Color (Optional)", options=["None"] + column_info["categorical"], key="color_col")

    if x_axis and y_axis != "None":
        fig = create_chart(filtered_df, st.session_state.selected_chart, x_axis, y_axis, color_col)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="selected_chart")

# Download data
st.subheader("Download Data")
csv = to_csv(filtered_df)
st.download_button(
    label="Download Filtered Data as CSV",
    data=csv,
    file_name="filtered_data.csv",
    mime="text/csv",
    key="download_button"
)

# Footer
st.markdown("---")
st.markdown("Developed by [Your Name] | Data Scientist & Developer")
