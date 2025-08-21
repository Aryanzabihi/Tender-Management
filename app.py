import streamlit as st
import pandas as pd
import numpy as np
import datetime

# Force clear all session state on app start
if 'force_reset' not in st.session_state:
    st.session_state.clear()
    st.session_state['force_reset'] = True
    st.rerun()

# Debug info to confirm app is reloading
st.write("🔄 App loaded at:", datetime.datetime.now())
st.write("📁 File path:", __file__)
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from fpdf import FPDF
import tempfile
import qrcode
import plotly.io as pio
import datetime
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time

pio.templates.default = "plotly_white"
CONTINUOUS_COLOR_SCALE = "Turbo"
CATEGORICAL_COLOR_SEQUENCE = px.colors.qualitative.Pastel

# --- PuLP import for optimization ---
try:
    from pulp import LpProblem, LpVariable, LpMaximize, lpSum
except ImportError:
    LpProblem = LpVariable = LpMaximize = lpSum = None

# --- Utility Functions: Variable Detection ---
def get_numeric_columns(df):
    """Return a list of numeric columns, excluding 'supplier'."""
    return [col for col in df.select_dtypes(include=['number']).columns if col != 'supplier']

def get_categorical_columns(df):
    """Return a list of categorical/object columns, excluding 'supplier'."""
    return [col for col in df.select_dtypes(include=['object']).columns if col != 'supplier']

# --- Utility Functions ---
def calculate_risk_scores(df):
    """Calculate various procurement risk scores for each supplier."""
    df = df.copy()
    mean_price = df['price'].mean()
    threshold = mean_price * 0.85
    df['lowball_risk'] = ((mean_price - df['price']) / mean_price).where(df['price'] < threshold, 0)
    if 'quality' in df.columns or 'technical' in df.columns:
        q_col = 'quality' if 'quality' in df.columns else 'technical'
        price_norm = (df['price'] - df['price'].min()) / (df['price'].max() - df['price'].min())
        quality_norm = (df[q_col] - df[q_col].min()) / (df[q_col].max() - df[q_col].min())
        df['drip_pricing_risk'] = price_norm * (1 - quality_norm)
        price_75 = df['price'].quantile(0.75)
        quality_25 = df[q_col].quantile(0.25)
        df['drip_pricing_flag'] = ((df['price'] > price_75) & (df[q_col] < quality_25)).astype(float)
    else:
        df['drip_pricing_risk'] = 0
        df['drip_pricing_flag'] = 0
    price_rank = df['price'].rank(pct=True)
    score_rank = df['score'].rank(pct=True)
    df['signaling_risk'] = ((score_rank > 0.8) & (price_rank > 0.8)).astype(float)
    high_price_threshold = df['price'].quantile(0.75)
    low_score_threshold = df['score'].median()
    df['cover_bid_risk'] = ((df['price'] > high_price_threshold) & (df['score'] < low_score_threshold)).astype(float)
    df['price_z'] = (df['price'] - df['price'].mean()) / df['price'].std(ddof=0)
    df['score_z'] = (df['score'] - df['score'].mean()) / df['score'].std(ddof=0)
    df['decoy_bid_risk'] = ((df['price_z'].abs() > 2) | (df['score_z'].abs() > 2)).astype(float)
    df['price_rounded'] = df['price'].round(-2)
    df['score_rounded'] = df['score'].round(0)
    df['price_similarity'] = df['price_rounded'].duplicated(keep=False)
    df['score_similarity'] = df['score_rounded'].duplicated(keep=False)
    df['bid_similarity_risk'] = (df['price_similarity'] | df['score_similarity']).astype(float)
    risk_cols = [
        'lowball_risk', 'drip_pricing_risk', 'drip_pricing_flag', 'signaling_risk',
        'cover_bid_risk', 'decoy_bid_risk', 'bid_similarity_risk'
    ]
    df['total_risk'] = df[risk_cols].mean(axis=1)
    return df

def get_variable_list(df):
    """Return a list of numeric variables for scoring, excluding supplier/name/id columns."""
    return [
        col for col in df.columns
        if col.lower() not in ['supplier', 'name', 'id']
        and pd.api.types.is_numeric_dtype(df[col])
        and not df[col].isnull().all()
    ]

def normalize_column(col, minimize=False):
    """Normalize a pandas Series to [0,1], optionally minimizing."""
    if col.nunique() == 1:
        return pd.Series([1.0]*len(col), index=col.index)
    if minimize:
        return (col.max() - col) / (col.max() - col.min())
    else:
        return (col - col.min()) / (col.max() - col.min())

def get_weights(variables, scenario):
    """Return variable weights for different scoring scenarios."""
    n = len(variables)
    weights = dict.fromkeys(variables, 1/n)
    if scenario == 'price-focused' and 'price' in variables:
        weights = dict.fromkeys(variables, 0.2/(n-1) if n > 1 else 1)
        weights['price'] = 0.6
    elif scenario == 'quality-focused' and 'quality' in variables:
        weights = dict.fromkeys(variables, 0.2/(n-1) if n > 1 else 1)
        weights['quality'] = 0.6
    return weights

def apply_common_layout(fig):
    """Apply a common layout to Plotly figures for consistent style."""
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Arial', size=14),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    return fig

# --- Caching for Sample Data ---
@st.cache_data
def get_sample_df():
    """Return a sample DataFrame for user guidance."""
    return pd.DataFrame({
        'supplier': ['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D', 'Supplier E', 'Supplier F', 'Supplier G'],
        'price': [1000, 1175, 1080, 1120, 1055, 1205, 1095],
        'currency': ['USD', 'EUR', 'USD', 'EUR', 'USD', 'EUR', 'USD'],
        'discount': [50, 20, 35, 15, 25, 10, 30],
        'payment_terms': ['Net 30', 'Net 60', 'Net 45', 'Net 30', 'Net 60', 'Net 45', 'Net 30'],
        'quantity': [100, 180, 150, 170, 120, 160, 140],
        'min_order_quantity': [50, 90, 70, 85, 60, 80, 65],
        'lead_time_days': [30, 42, 29, 38, 33, 36, 40],
        'quality': [80, 86, 79, 83, 81, 82, 85],
        'technical': [90, 87, 86, 88, 85, 89, 91],
        'warranty_months': [24, 18, 20, 22, 18, 16, 24],
        'certifications': ['ISO9001,CE', 'ISO14001', 'ISO9001', 'CE', 'ISO9001,ISO14001', 'CE', 'ISO9001,CE'],
        'delivery_time_days': [35, 48, 33, 44, 39, 37, 41],
        'delivery_terms': ['FOB', 'CIF', 'FOB', 'CIF', 'FOB', 'CIF', 'FOB'],
        'shipping_cost': [100, 145, 95, 125, 110, 120, 115],
        'compliance': ['Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes'],
        'compliance_notes': ['All docs provided', 'Missing certificate', 'All docs provided', 'All docs provided', 'All docs provided', 'Missing docs', 'All docs provided'],
        'country': ['USA', 'Germany', 'USA', 'France', 'USA', 'Italy', 'Spain'],
        'supplier_type': ['Manufacturer', 'Distributor', 'Manufacturer', 'Distributor', 'Manufacturer', 'Distributor', 'Manufacturer'],
        'experience_years': [10, 6, 8, 7, 9, 5, 11],
        'score': [0.85, 0.79, 0.83, 0.81, 0.82, 0.77, 0.86],
        'remarks': ['Preferred', '', 'Reliable', '', 'Preferred', '', 'Top rated']
    })

# --- Visualization Tab Functions ---
def score_composition_tab(ranked, variables, weights, custom_weights=None):
    """Display a stacked bar chart of score composition with improved readability."""
    if custom_weights:
        contrib_df = pd.DataFrame({var: ranked[var] * custom_weights[var] for var in variables})
    else:
        contrib_df = pd.DataFrame({var: ranked[var] * weights[var] for var in variables})
    contrib_df['supplier'] = ranked['supplier']
    contrib_df = contrib_df.set_index('supplier')
    fig = px.bar(
        contrib_df,
        x=contrib_df.index,
        y=variables,
        title='Score Composition',
        labels={'value': 'Score Contribution', 'supplier': 'Supplier'},
        template='plotly_white',
        width=1200,  # Increased width for readability
        height=500
    )
    fig.update_layout(
        barmode='stack',
        xaxis_title='Supplier',
        yaxis_title='Score',
        xaxis_tickangle=-45,  # Rotate x labels for clarity
        legend_title_text='Variable',
        margin=dict(l=40, r=40, t=60, b=120)
    )
    st.plotly_chart(fig, use_container_width=True)

def price_distribution_tab(filtered_data):
    if 'price' in filtered_data.columns and 'supplier' in filtered_data.columns:
        fig = go.Figure()
        # Box plot for price distribution
        fig.add_trace(go.Box(
            y=filtered_data['price'],
            name='Price',
            boxpoints=False,
            marker_color='rgba(31,119,180,0.5)',
            line_color='rgba(31,119,180,1)'
        ))
        # Assign a unique color to each supplier
        suppliers = filtered_data['supplier'].unique()
        palette = px.colors.qualitative.Dark2
        color_map = {sup: palette[i % len(palette)] for i, sup in enumerate(suppliers)}
        for sup in suppliers:
            sub = filtered_data[filtered_data['supplier'] == sup]
            fig.add_trace(go.Scatter(
                y=sub['price'],
                x=['']*len(sub),
                mode='markers',
                marker=dict(size=10, color=color_map[sup]),
                text=sub['supplier'],
                hovertemplate='Supplier: %{text}<br>Price: %{y}<extra></extra>',
                name=sup,
                showlegend=False
            ))
        fig.update_layout(
            title='Price Distribution of Suppliers',
            yaxis_title='Price',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    # Removed score distribution plot

def supplier_profiles_tab(ranked, variables):
    if len(variables) >= 3:
        fig = go.Figure()
        for i, row in ranked.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row[var] for var in variables],
                theta=variables,
                fill='toself',
                name=row['supplier']
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('Radar chart requires at least 3 numeric variables.')

def price_vs_quality_technical_tab(filtered_data, original_data):
    st.subheader('Price vs. Quality/Technical')
    view_option = st.radio(
        "Show data for:",
        ["Current Scenario", "All Suppliers"],
        horizontal=True
    )
    if view_option == "Current Scenario":
        plot_data = filtered_data.copy()
    else:
        plot_data = original_data.copy()
    # Only plot if there are suppliers to show
    if not plot_data.empty and 'price' in plot_data.columns and ('quality' in plot_data.columns or 'technical' in plot_data.columns):
        y_var = 'quality' if 'quality' in plot_data.columns else 'technical'
        chart = alt.Chart(plot_data).mark_circle(size=60).encode(
            x='price', y=y_var, color='supplier', tooltip=['supplier', 'price', y_var]
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info('No suppliers to display or both price and quality/technical columns are required.')

def supplier_comparison_table(filtered_data):
    supplier_options = filtered_data['supplier'].unique().tolist() if 'supplier' in filtered_data.columns else []
    selected_suppliers = st.multiselect('Select suppliers to compare', supplier_options)
    if selected_suppliers:
        st.dataframe(filtered_data[filtered_data['supplier'].isin(selected_suppliers)], use_container_width=True)
    else:
        st.info('Select suppliers to compare their details.')

def correlation_heatmap_tab(filtered_data):
    corr = filtered_data.select_dtypes(include='number').corr()
    if corr.empty:
        st.info('Not enough numeric data for correlation heatmap.')
        return
    labels = corr.columns.tolist()
    z = corr.values
    hovertext = [[f'{labels[i]} vs {labels[j]}<br>Corr: {z[i][j]:.2f}' for j in range(len(labels))] for i in range(len(labels))]
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=hovertext,
        hoverinfo='text',
        colorbar=dict(title='Correlation')
    ))
    # Add correlation values as annotations if not too many variables
    if len(labels) <= 12:
        for i in range(len(labels)):
            for j in range(len(labels)):
                fig.add_annotation(
                    x=labels[j], y=labels[i],
                    text=f'{z[i][j]:.2f}',
                    showarrow=False,
                    font=dict(size=12, color='black' if abs(z[i][j]) < 0.7 else 'white')
                )
    fig.update_layout(
        title='Correlation Heatmap',
        xaxis=dict(tickangle=45, side='top'),
        yaxis=dict(autorange='reversed'),
        width=700, height=700,
        margin=dict(l=100, b=100, t=50, r=50)
    )
    st.plotly_chart(fig, use_container_width=True)

def pareto_chart_tab(filtered_data):
    if 'price' in filtered_data.columns:
        pareto_df = filtered_data[['supplier', 'price']].copy()
        pareto_df = pareto_df.groupby('supplier', as_index=False).sum()
        pareto_df = pareto_df.sort_values('price', ascending=False)
        pareto_df['cum_pct'] = pareto_df['price'].cumsum() / pareto_df['price'].sum() * 100
        fig = go.Figure()
        fig.add_bar(x=pareto_df['supplier'], y=pareto_df['price'], name='Price')
        fig.add_scatter(x=pareto_df['supplier'], y=pareto_df['cum_pct'], name='Cumulative %', yaxis='y2')
        fig.update_layout(
            title='Pareto Chart: Supplier Price Contribution',
            yaxis=dict(title='Price'),
            yaxis2=dict(title='Cumulative %', overlaying='y', side='right', range=[0, 100]),
            xaxis_title='Supplier',
            legend=dict(orientation='h'),
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info('No price column available.')

def advanced_analytics_tab(ranked, filtered_data, variables):
    st.header('Advanced Analytics')
    analytics_tabs = st.tabs([
        "Anomaly Detection",
        "Principal Component Analysis (PCA)",
        "Price Feature Importance",
        "Procurement Risk Scoring"
    ])
    with analytics_tabs[0]:
        st.subheader('Anomaly Detection for Supplier Offers')
        st.write('Identify outliers or suspicious offers using Isolation Forest (multidimensional outlier detection).')
        if not filtered_data.empty:
            model = IsolationForest(contamination=0.1, random_state=42)
            anomalies = model.fit_predict(filtered_data[variables].fillna(0))
            ranked['anomaly'] = anomalies
            outliers = ranked[ranked['anomaly'] == -1]
            st.write('🚨 Detected Anomalies (Suspicious Offers):')
            cols = ['supplier', 'score'] + variables
            cols = list(dict.fromkeys(cols))
            st.dataframe(outliers[cols], use_container_width=True)
            plot_df = ranked.copy()
            plot_df['anomaly_label'] = plot_df['anomaly'].map({-1: 'Outlier', 1: 'Normal'})
            if 'price' in variables and ('quality' in variables or 'technical' in variables):
                y_var = 'quality' if 'quality' in variables else 'technical'
                fig = px.scatter(plot_df, x='price', y=y_var, color='anomaly_label',
                                 hover_data=['supplier'], title='Isolation Forest Outliers: Price vs. Quality/Technical', symbol='anomaly_label')
            else:
                fig = px.scatter(plot_df, x=variables[0], y=variables[1], color='anomaly_label',
                                 hover_data=['supplier'], title=f'Isolation Forest Outliers: {variables[0]} vs. {variables[1]}', symbol='anomaly_label')
            st.plotly_chart(fig, use_container_width=True)
    with analytics_tabs[1]:
        st.subheader('Principal Component Analysis (PCA)')
        st.write('Reduce dimensionality for better visual interpretation and highlight key contributing variables.')
        if not filtered_data.empty:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(filtered_data[variables])
            pca = PCA(n_components=2)
            components = pca.fit_transform(scaled_data)
            pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
            pca_df['supplier'] = filtered_data['supplier'].values
            chart = alt.Chart(pca_df).mark_circle(size=60).encode(
                x='PC1', y='PC2', color='supplier', tooltip=['supplier', 'PC1', 'PC2']
            ).properties(title='PCA: Supplier Distribution')
            st.altair_chart(chart, use_container_width=True)
    with analytics_tabs[2]:
        st.subheader('Price Feature Importance')
        st.write('See which variables are most important in explaining supplier price using a Random Forest model.')
        if not ranked.empty and 'price' in ranked.columns:
            feature_vars = [v for v in variables if v != 'price']
            X = ranked[feature_vars]
            y = ranked['price']
            model = RandomForestRegressor(random_state=42)
            model.fit(X, y)
            importances = model.feature_importances_
            imp_df = pd.DataFrame({'Variable': feature_vars, 'Importance': importances}).sort_values('Importance', ascending=False).reset_index(drop=True)
            imp_df.index = imp_df.index + 1
            imp_df.index.name = 'Rank'
            fig = px.bar(
                imp_df,
                x='Variable',
                y='Importance',
                color='Importance',
                color_continuous_scale='Viridis',
                title='Feature Importance for Price'
            )
            fig.update_coloraxes(showscale=True, colorbar_title='Importance')
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(imp_df, use_container_width=True)
        else:
            st.info('Not enough data to compute feature importance.')
    with analytics_tabs[3]:
        st.header('Advanced Analytics: Procurement Risk Scoring')
        risk_df = calculate_risk_scores(ranked)
        st.subheader('Supplier Risk Scoring')
        st.write('Each supplier is scored on risk behaviors: Lowballing, Drip Pricing, Drip Pricing Flag, Market Signaling, Cover Bidding, Decoy Bidding, and Bid Similarity. The total risk is the average of these scores.')
        # Show only one column per risk and total_risk
        risk_table_cols = ['supplier', 'total_risk', 'lowball_risk', 'drip_pricing_risk', 'drip_pricing_flag', 'signaling_risk', 'cover_bid_risk', 'decoy_bid_risk', 'bid_similarity_risk']
        st.dataframe(risk_df[risk_table_cols], use_container_width=True)
        fig = px.bar(risk_df, y='supplier', x='total_risk', color='total_risk', orientation='h',
                     title='Overall Supplier Risk Score',
                     labels={'total_risk': 'Risk Score', 'supplier': 'Supplier'}, template='plotly_white')
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

# --- Filter Suppliers Tab Plot Functions ---
def plot_bubble_price_quality_delivery(df):
    if all(v in df.columns for v in ['price', 'quality', 'delivery_time_days']):
        fig = px.scatter(
            df,
            x='price',
            y='quality',
            size='delivery_time_days',
            color='supplier',
            color_discrete_sequence=px.colors.qualitative.Dark2,
            title='Price vs. Quality (Bubble = Delivery Time)',
            labels={'price': 'Price', 'quality': 'Quality', 'delivery_time_days': 'Delivery Time (days)', 'supplier': 'Supplier'}
        )
        fig.update_layout(xaxis_title='Price', yaxis_title='Quality')
        st.plotly_chart(fig, use_container_width=True, key='filtertab_bubble_price_quality_delivery')

def plot_line_lead_time(df):
    if 'lead_time_days' in df.columns:
        fig = px.line(
            df.sort_values('lead_time_days'),
            x='supplier',
            y='lead_time_days',
            markers=True,
            color_discrete_sequence=px.colors.qualitative.Dark2,
            title='Lead Time Days by Supplier (Line)',
            labels={'supplier': 'Supplier', 'lead_time_days': 'Lead Time Days'}
        )
        st.plotly_chart(fig, use_container_width=True, key='filtertab_line_lead_time_days')

def plot_bar_delivery_time(df):
    if 'delivery_time_days' in df.columns:
        fig = px.bar(
            df,
            x='supplier',
            y='delivery_time_days',
            color='supplier',
            color_discrete_sequence=px.colors.qualitative.Dark2,
            title='Delivery Time Days by Supplier (Bar)',
            labels={'supplier': 'Supplier', 'delivery_time_days': 'Delivery Time Days'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True, key='filtertab_bar_delivery_time_days')

def plot_lollipop_experience(df):
    if 'experience_years' in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['supplier'],
            y=df['experience_years'],
            mode='markers',
            marker=dict(size=12, color=px.colors.qualitative.Dark2[0]),
            name='Experience'
        ))
        for i, row in df.iterrows():
            fig.add_trace(go.Scatter(
                x=[row['supplier'], row['supplier']],
                y=[0, row['experience_years']],
                mode='lines',
                line=dict(color=px.colors.qualitative.Dark2[1], width=3),
                showlegend=False
            ))
        fig.update_layout(title='Experience Years by Supplier (Lollipop)', xaxis_title='Supplier', yaxis_title='Experience Years')
        st.plotly_chart(fig, use_container_width=True, key='filtertab_lollipop_experience_years')

def plot_dot_warranty(df):
    if 'warranty_months' in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['supplier'],
            y=df['warranty_months'],
            mode='markers',
            marker=dict(size=10, color=px.colors.qualitative.Dark2[2]),
            name='Warranty'
        ))
        fig.update_layout(title='Warranty Months by Supplier (Dot Plot)', xaxis_title='Supplier', yaxis_title='Warranty Months')
        st.plotly_chart(fig, use_container_width=True, key='filtertab_dot_warranty_months')

def plot_stackedbar_compliance(df):
    if 'compliance' in df.columns:
        compliance_counts = df.groupby(['supplier', 'compliance']).size().reset_index(name='count')
        fig = px.bar(
            compliance_counts,
            x='supplier',
            y='count',
            color='compliance',
            title='Compliance by Supplier (Stacked Bar)',
            labels={'supplier': 'Supplier', 'count': 'Count', 'compliance': 'Compliance'}
        )
        st.plotly_chart(fig, use_container_width=True, key='filtertab_stackedbar_compliance')

def plot_violin_price(df):
    if 'price' in df.columns:
        fig = go.Figure()
        fig.add_trace(go.Violin(
            y=df['price'],
            box_visible=True,
            meanline_visible=True,
            line_color=px.colors.qualitative.Dark2[0],
            fillcolor='rgba(31,119,180,0.3)',
            name='Price Distribution',
            points=False
        ))
        fig.add_trace(go.Scatter(
            y=df['price'],
            x=['']*len(df),
            mode='markers',
            marker=dict(size=10, color=px.colors.qualitative.Dark2[1]),
            text=df['supplier'],
            hovertemplate='Supplier: %{text}<br>Price: %{y}<extra></extra>',
            name='Supplier'
        ))
        fig.update_layout(
            title='Distribution of Price (Violin, Supplier Overlay)',
            yaxis_title='Price',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True, key='filtertab_violin_price')

def plot_scatter_price_delivery(df):
    if 'price' in df.columns and 'delivery_time_days' in df.columns:
        fig = px.scatter(
            df,
            x='price',
            y='delivery_time_days',
            color='supplier',
            color_discrete_sequence=px.colors.qualitative.Dark2,
            title='Price vs. Delivery Time Days (Scatter)',
            labels={'price': 'Price', 'delivery_time_days': 'Delivery Time Days', 'supplier': 'Supplier'}
        )
        st.plotly_chart(fig, use_container_width=True, key='filtertab_scatter_price_delivery')

st.set_page_config(page_title='Smart Tender Evaluation', layout='wide')

st.title('Smart Tender Evaluation')

with st.expander('How to use this app', expanded=True):
    st.markdown('''
    1. **Upload** your tender offers as a CSV file using the sidebar.
    2. **Apply filters** in the sidebar to focus on relevant offers.
    3. **Review** the detected variables and supplier data.
    4. **Explore** automatic scoring scenarios (Equal, Price-focused, Quality-focused).
    5. **Set custom weights** for variables in the sidebar to create your own scenario.
    6. **Download** results and view descriptive charts for deeper insights.
    
    > **Note:** For generating PDF reports with charts, you must [download and run the app locally](https://www.dropbox.com/scl/fo/d1npg1g1gs8416vqqys22/AK7LMgDzIeTsshkcrTz9iYY?rlkey=nozu94wfmenl0p9kij0hm9x8i&st=b37syxar&dl=0). Google Chrome is required for PDF generation with charts.
    ''')

# Sidebar controls - Organized in 3 sections
with st.sidebar:
    # ===== SECTION 1: DATA MANAGEMENT =====
    st.header('📊 Data Management')
    
    # Template download button
    st.markdown("**📋 Get Started with Template**")
    template_df = pd.DataFrame({
        'supplier': ['Supplier A', 'Supplier B', 'Supplier C'],
        'price': [1000, 1175, 1080],
        'currency': ['USD', 'EUR', 'USD'],
        'discount': [50, 20, 35],
        'payment_terms': ['Net 30', 'Net 60', 'Net 45'],
        'quantity': [100, 180, 150],
        'lead_time_days': [30, 42, 29],
        'quality': [80, 86, 79],
        'technical': [90, 87, 86],
        'warranty_months': [24, 18, 20],
        'certifications': ['ISO9001,CE', 'ISO14001', 'ISO9001'],
        'delivery_time_days': [35, 48, 33],
        'delivery_terms': ['FOB', 'CIF', 'FOB'],
        'shipping_cost': [100, 145, 95],
        'compliance': ['Yes', 'No', 'Yes'],
        'compliance_notes': ['All docs provided', 'Missing certificate', 'All docs provided'],
        'country': ['USA', 'Germany', 'USA'],
        'supplier_type': ['Manufacturer', 'Distributor', 'Manufacturer'],
        'experience_years': [10, 6, 8],
        'score': [0.85, 0.79, 0.83],
        'remarks': ['Preferred', '', 'Reliable']
    })
    template_csv = template_df.to_csv(index=False)
    st.download_button(
        label='📥 Download Template CSV',
        data=template_csv,
        file_name='tender_evaluation_template.csv',
        mime='text/csv',
        help='Download a clean template CSV file to get started with your tender evaluation.',
        key='template_download'
    )
    
    # Load Sample Data button
    st.markdown("**🧪 Test App with Sample Data**")
    if st.button('📊 Load Sample Dataset', key='load_sample_data', help='Click to load sample data and test the app functionality'):
        st.session_state['sample_data_loaded'] = True
        st.session_state['sample_data'] = pd.DataFrame({
            'supplier': ['Supplier A', 'Supplier B', 'Supplier C', 'Supplier D', 'Supplier E', 'Supplier F', 'Supplier G'],
            'price': [1000, 1175, 1080, 1120, 1055, 1205, 1095],
            'currency': ['USD', 'EUR', 'USD', 'EUR', 'USD', 'EUR', 'USD'],
            'discount': [50, 20, 35, 15, 25, 10, 30],
            'payment_terms': ['Net 30', 'Net 60', 'Net 45', 'Net 30', 'Net 60', 'Net 45', 'Net 30'],
            'quantity': [100, 180, 150, 170, 120, 160, 140],
            'min_order_quantity': [50, 90, 70, 85, 60, 80, 65],
            'lead_time_days': [30, 42, 29, 38, 33, 36, 40],
            'quality': [80, 86, 79, 83, 81, 82, 85],
            'technical': [90, 87, 86, 88, 85, 89, 91],
            'warranty_months': [24, 18, 20, 22, 18, 16, 24],
            'certifications': ['ISO9001,CE', 'ISO14001', 'ISO9001', 'CE', 'ISO9001,ISO14001', 'CE', 'ISO9001,CE'],
            'delivery_time_days': [35, 48, 33, 44, 39, 37, 41],
            'delivery_terms': ['FOB', 'CIF', 'FOB', 'CIF', 'FOB', 'CIF', 'FOB'],
            'shipping_cost': [100, 145, 95, 125, 110, 120, 115],
            'compliance': ['Yes', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes'],
            'compliance_notes': ['All docs provided', 'Missing certificate', 'All docs provided', 'All docs provided', 'All docs provided', 'Missing docs', 'All docs provided'],
            'country': ['USA', 'Germany', 'USA', 'France', 'USA', 'Italy', 'Spain'],
            'supplier_type': ['Manufacturer', 'Distributor', 'Manufacturer', 'Distributor', 'Manufacturer', 'Distributor', 'Manufacturer'],
            'experience_years': [10, 6, 8, 7, 9, 5, 11],
            'score': [0.85, 0.79, 0.83, 0.81, 0.82, 0.77, 0.86],
            'remarks': ['Preferred', '', 'Reliable', '', 'Preferred', '', 'Top rated']
        })
        st.success('✅ Sample data loaded successfully! You can now explore all app features.')
        st.rerun()
    
    # Show status if sample data is loaded
    if st.session_state.get('sample_data_loaded', False):
        st.info('📊 Sample data is currently loaded. Use the tabs above to explore the app features.')
    
    # File uploader
    uploaded_file = st.file_uploader('📁 Upload Tender Offers CSV', type=['csv'], help='Upload a CSV file with supplier offers and variables.')
    
    # Manual data entry
    if 'show_manual_entry' not in st.session_state:
        st.session_state['show_manual_entry'] = False
    if st.button('✏️ Manual Data Entry', key='show_manual_entry_btn'):
        st.session_state['show_manual_entry'] = not st.session_state['show_manual_entry']
    if st.session_state['show_manual_entry']:
        st.subheader('Manual Data Entry')
        # Categorized variable schema
        variable_schema = {
            'Price': ['price', 'currency', 'discount', 'payment_terms', 'shipping_cost'],
            'Quantity': ['quantity', 'min_order_quantity'],
            'Delivery': ['lead_time_days', 'delivery_time_days', 'delivery_terms'],
            'Quality/Technical': ['quality', 'technical', 'warranty_months', 'certifications'],
            'Supplier Info': ['supplier', 'country', 'supplier_type', 'experience_years'],
            'Compliance': ['compliance', 'compliance_notes'],
            'Other': ['score', 'remarks']
        }
        # Flat list for logic
        possible_vars = [v for group in variable_schema.values() for v in group]
        default_vars = ['supplier', 'price', 'quality', 'delivery_time_days']
        # For categorized multiselect (no external dependency)
        def categorized_multiselect(label, schema, selected, key=None):
            options = []
            for cat, vars in schema.items():
                options += [f"{cat}: {v}" for v in vars]
            selected_labels = [f"{cat}: {v}" for cat, vars in schema.items() for v in vars if v in selected]
            chosen = st.multiselect(label, options, default=selected_labels, key=key)
            return [c.split(': ', 1)[1] for c in chosen]
        if 'manual_vars' not in st.session_state:
            st.session_state['manual_vars'] = default_vars.copy()
        if st.button('Add More Variables', key='show_more_vars_btn'):
            st.session_state['show_more_vars'] = not st.session_state.get('show_more_vars', False)
        if st.session_state.get('show_more_vars', False):
            more_vars = categorized_multiselect('Select additional variables to add:', variable_schema, [v for v in possible_vars if v not in st.session_state['manual_vars']], key='more_vars_multiselect')
            if st.button('Confirm Variables', key='confirm_more_vars_btn'):
                st.session_state['manual_vars'] += [v for v in more_vars if v not in st.session_state['manual_vars']]
                st.session_state['show_more_vars'] = False
        # Manual data entry form
        if 'manual_data' not in st.session_state:
            st.session_state['manual_data'] = []
        with st.form('manual_data_form', clear_on_submit=True):
            entry = {}
            for var in st.session_state['manual_vars']:
                if var == 'supplier':
                    entry[var] = st.text_input('Supplier')
                elif var in ['compliance', 'remarks']:
                    entry[var] = st.text_input(var.replace('_', ' ').capitalize())
                else:
                    entry[var] = st.number_input(var.replace('_', ' ').capitalize(), min_value=0.0, step=1.0)
            submitted = st.form_submit_button('Add Supplier')
            if submitted and entry.get('supplier'):
                st.session_state['manual_data'].append(entry)
        if st.session_state['manual_data']:
            manual_df = pd.DataFrame(st.session_state['manual_data'])
            st.dataframe(manual_df, use_container_width=True)
            use_manual = st.checkbox('Use manually entered data for analysis', key='use_manual_data')
    
    # Reset button
    if st.button("🔄 Reset App Completely", key="manual_reset"):
        st.session_state.clear()
        st.rerun()
    
    st.markdown("---")
    
    # ===== SECTION 2: REPORTS & EXPORTS =====
    st.header('📋 Reports & Exports')
    
    # Include Dashboard, Scoring Model, SSBI, Advanced Analytics, Negotiation Strategy, and What-If Analysis in the sidebar options for report generation
    report_options = {
        "Dashboard": st.checkbox("Dashboard", value=True),
        "Scoring Model": st.checkbox("Scoring Model", value=False),
        "SSBI": st.checkbox("SSBI", value=False),
        "Advanced Analytics": st.checkbox("Advanced Analytics", value=False),
        "Negotiation Strategy": st.checkbox("Negotiation Strategy", value=False),
        "What-If Analysis": st.checkbox("What-If Analysis", value=False),
    }

    generate_report = st.button("📄 Generate PDF Report")
    
    # Add progress bar under the button
    if generate_report:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    st.markdown("---")
    
    # ===== SECTION 3: DEVELOPER INFO & SUPPORT =====
    st.header('👨‍💻 Developer & Support')
    st.markdown("**Developed by Aryan Zabihi**")
    
    # All links in one unified section
    st.markdown("[🔗 GitHub](https://github.com/Aryanzabihi)")
    st.markdown("[💼 LinkedIn](https://www.linkedin.com/in/aryanzabihi/)")
    st.markdown("💝 [Donate via PayPal](https://www.paypal.com/donate/?hosted_button_id=C9W46U77KNU9S)")
    
    # Custom CSS to remove underlines from hyperlinks
    st.markdown("""
        <style>
        .stMarkdown a {
            text-decoration: none !important;
        }
        </style>
        """, unsafe_allow_html=True)


# Initialize data and variables
data = pd.DataFrame()
original_data = pd.DataFrame()
variables = []

if st.session_state.get('use_manual_data', False) and st.session_state.get('manual_data'):
    data = pd.DataFrame(st.session_state['manual_data'])
    st.header('Manual Data')
    st.dataframe(data, use_container_width=True)
    variables = get_variable_list(data)
elif st.session_state.get('sample_data_loaded', False) and st.session_state.get('sample_data') is not None:
    # Load sample data for testing
    original_data = st.session_state['sample_data'].copy()
    data = original_data.copy()
    st.header('📊 Sample Data (Testing Mode)')
    st.info('This is sample data loaded for testing the app. You can explore all features with this data.')
    st.dataframe(data, use_container_width=True)
    
    # Apply compliance filtering if column exists
    if 'compliance' in data.columns:
        data = data[data['compliance'].str.lower() == 'yes']

    # Force numeric conversion for all relevant columns
    exclude_cols = ['supplier', 'name', 'id', 'compliance', 'certifications', 'compliance_notes', 'payment_terms', 'supplier_type']
    for col in data.columns:
        if col.lower() not in exclude_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    variables = get_variable_list(data)
elif uploaded_file is not None:
    original_data = pd.read_csv(uploaded_file)
    data = original_data.copy()
    st.header('Uploaded Data')
    st.dataframe(data, use_container_width=True)

    # Apply compliance filtering if column exists
    if 'compliance' in data.columns:
        data = data[data['compliance'].str.lower() == 'yes']

    # Force numeric conversion for all relevant columns
    exclude_cols = ['supplier', 'name', 'id', 'compliance', 'certifications', 'compliance_notes', 'payment_terms', 'supplier_type']
    for col in data.columns:
        if col.lower() not in exclude_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    variables = get_variable_list(data)

# --- Validation ---
if not data.empty:
    missing_vars = [var for var in variables if data[var].isnull().any()]
    if missing_vars:
        st.warning(f"Missing values detected in: {', '.join(missing_vars)}. These rows will be included but may affect scoring.")
    
    # --- Main Tabs: Dashboard, Scoring Model, Advanced Analytics, Negotiation Strategy, and SSBI ---
    main_tab_labels = ["Dashboard", "Scoring Model", "SSBI", "Advanced Analytics", "Negotiation Strategy", "What-If Analysis"]
    main_tabs = st.tabs(main_tab_labels)

    with main_tabs[0]:  # Dashboard Tab
        st.header('Dashboard')
        dashboard_tabs = st.tabs(["Supplier Comparison", "Delivery & Lead Time Insights", "Compliance & Risk Overview", "Payment & Terms", "Supplier Experience & Type"])
        with dashboard_tabs[0]:
            st.header('Supplier Comparison')
            # 1. Bar chart: supplier vs price (sorted ascending for best deals)
            if 'supplier' in data.columns and 'price' in data.columns:
                st.subheader('Supplier vs Price (Best Deals First)')
                sorted_df = data.sort_values('price')
                fig_price = px.bar(
                    sorted_df,
                    x='supplier',
                    y='price',
                    color='price',
                    color_continuous_scale=CONTINUOUS_COLOR_SCALE,
                    title='Supplier vs Price',
                    labels={'price': 'Price', 'supplier': 'Supplier'},
                    hover_data=['supplier', 'price']
                )
                fig_price.update_traces(marker_line_width=2)
                fig_price.update_layout(yaxis=dict(showgrid=True), xaxis=dict(categoryorder='total ascending'))
                fig_price = apply_common_layout(fig_price)
                st.plotly_chart(fig_price, use_container_width=True)
            else:
                st.info('Price or supplier column missing for price comparison.')

            # 2. Bar chart: supplier vs quality
            if 'supplier' in data.columns and 'quality' in data.columns:
                st.subheader('Supplier vs Quality')
                fig_quality = px.bar(
                    data,
                    x='supplier',
                    y='quality',
                    color='quality',
                    color_continuous_scale=CONTINUOUS_COLOR_SCALE,
                    title='Supplier vs Quality',
                    labels={'quality': 'Quality', 'supplier': 'Supplier'},
                    hover_data=['supplier', 'quality']
                )
                fig_quality.update_traces(marker_line_width=2)
                fig_quality.update_layout(yaxis=dict(showgrid=True), xaxis=dict(categoryorder='total descending'))
                fig_quality = apply_common_layout(fig_quality)
                st.plotly_chart(fig_quality, use_container_width=True)
            else:
                st.info('Quality or supplier column missing for quality comparison.')

            # 3. Radar chart: quality, technical, warranty_months per supplier
            radar_cols = ['price', 'delivery_time_days', 'warranty_months', 'quality']
            if all(col in data.columns for col in radar_cols) and 'supplier' in data.columns:
                st.subheader('Supplier Profile Radar Chart')
                # Normalize each variable to [0,1] for fair comparison
                norm_df = data.copy()
                from inspect import signature
                # Use minimize=True for price and delivery_time_days (lower is better), False for others
                norm_df['price'] = normalize_column(norm_df['price'], minimize=True)
                norm_df['delivery_time_days'] = normalize_column(norm_df['delivery_time_days'], minimize=True)
                norm_df['warranty_months'] = normalize_column(norm_df['warranty_months'], minimize=False)
                norm_df['quality'] = normalize_column(norm_df['quality'], minimize=False)
                # Debug: Show normalized values for each supplier and variable
                st.write('**Radar Chart Normalized Data:**')
                st.dataframe(norm_df[['supplier'] + radar_cols], use_container_width=True)
                fig_radar = go.Figure()
                for i, row in norm_df.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[row[c] for c in radar_cols],
                        theta=radar_cols,
                        fill='toself',
                        name=row['supplier'],
                        line=dict(width=3)
                    ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    template='plotly_white'
                )
                fig_radar = apply_common_layout(fig_radar)
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                st.info('Radar chart requires columns: price, delivery_time_days, warranty_months, quality, and supplier.')

            # 4. Table with calculated Total Cost = price – discount + shipping_cost
            st.subheader('Supplier Total Cost Table')
            if all(col in data.columns for col in ['price', 'discount', 'shipping_cost', 'supplier']):
                df_cost = data.copy()
                df_cost['Total Cost'] = df_cost['price'] - df_cost['discount'] + df_cost['shipping_cost']
                st.dataframe(df_cost[['supplier', 'price', 'discount', 'shipping_cost', 'Total Cost']], use_container_width=True)
            else:
                st.info('Total Cost table requires columns: price, discount, shipping_cost, and supplier.')

        with dashboard_tabs[1]:
            st.header('Delivery & Lead Time Insights')
            kpi_cols = st.columns(2)
            avg_lead = data['lead_time_days'].mean() if 'lead_time_days' in data.columns else None
            min_delivery = data['delivery_time_days'].min() if 'delivery_time_days' in data.columns else None
            kpi_cols[0].metric('Average Lead Time', f"{avg_lead:.2f} days" if avg_lead is not None else '-')
            kpi_cols[1].metric('Min Delivery Time', f"{min_delivery:.2f} days" if min_delivery is not None else '-')

            # Scatter plot: lead_time_days vs delivery_time_days (bubble size = quantity)
            if all(col in data.columns for col in ['lead_time_days', 'delivery_time_days', 'quantity']):
                st.subheader('Lead Time vs Delivery Time (Bubble = Quantity)')
                fig_scatter = px.scatter(
                    data,
                    x='lead_time_days',
                    y='delivery_time_days',
                    size='quantity',
                    color='quantity',
                    color_continuous_scale=CONTINUOUS_COLOR_SCALE,
                    hover_data=['supplier', 'lead_time_days', 'delivery_time_days', 'quantity'],
                    title='Lead Time vs Delivery Time',
                    labels={'lead_time_days': 'Lead Time (days)', 'delivery_time_days': 'Delivery Time (days)', 'quantity': 'Quantity'}
                )
                fig_scatter.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
                fig_scatter = apply_common_layout(fig_scatter)
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info('Scatter plot requires columns: lead_time_days, delivery_time_days, and quantity.')

            # Bar chart: supplier vs lead_time_days
            if 'supplier' in data.columns and 'lead_time_days' in data.columns:
                st.subheader('Supplier vs Lead Time')
                fig_lead = px.bar(
                    data,
                    x='supplier',
                    y='lead_time_days',
                    color='lead_time_days',
                    color_continuous_scale=CONTINUOUS_COLOR_SCALE,
                    hover_data=['supplier', 'lead_time_days'],
                    title='Supplier vs Lead Time (days)',
                    labels={'supplier': 'Supplier', 'lead_time_days': 'Lead Time Days'}
                )
                fig_lead.update_traces(marker_line_width=2)
                fig_lead.update_layout(yaxis=dict(showgrid=True), xaxis=dict(categoryorder='total descending'))
                fig_lead = apply_common_layout(fig_lead)
                st.plotly_chart(fig_lead, use_container_width=True)
            else:
                st.info('Bar chart requires columns: supplier and lead_time_days.')

        with dashboard_tabs[2]:
            st.header('Compliance & Risk Overview')
            # Pie chart: compliance (Yes/No)
            if 'compliance' in data.columns:
                st.subheader('Compliance Distribution')
                compliance_counts = data['compliance'].value_counts(dropna=False).reset_index()
                compliance_counts.columns = ['Compliance', 'Count']
                fig_pie = px.pie(
                    compliance_counts,
                    names='Compliance',
                    values='Count',
                    color='Compliance',
                    color_discrete_sequence=CATEGORICAL_COLOR_SEQUENCE,
                    title='Compliance (Yes/No)',
                    hole=0.4
                )
                fig_pie.update_traces(textinfo='percent+label', pull=[0.05 if c == compliance_counts['Compliance'].iloc[0] else 0 for c in compliance_counts['Compliance']])
                fig_pie = apply_common_layout(fig_pie)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info('Compliance column not found for pie chart.')

            # Table or stacked bar: supplier vs compliance_notes
            if all(col in data.columns for col in ['supplier', 'compliance', 'compliance_notes', 'certifications']):
                st.subheader('Supplier Compliance & Certifications Overview')
                merged_table = data[['supplier', 'compliance', 'compliance_notes', 'certifications']].copy()
                st.dataframe(merged_table, use_container_width=True)
                # Highlight suppliers with missing certifications
                missing_cert = merged_table[merged_table['certifications'].isnull() | (merged_table['certifications'].astype(str).str.strip() == '')]
                if not missing_cert.empty:
                    st.warning(f"Suppliers missing certifications: {', '.join(missing_cert['supplier'].astype(str))}")
                    st.dataframe(missing_cert[['supplier', 'certifications']], use_container_width=True)
            else:
                st.info('One or more columns missing: supplier, compliance, compliance_notes, certifications.')

        with dashboard_tabs[3]:
            st.header('Payment & Terms')
            # KPI cards for payment terms
            if 'payment_terms' in data.columns:
                most_common = data['payment_terms'].mode().iloc[0] if not data['payment_terms'].mode().empty else None
                most_common_count = data['payment_terms'].value_counts().iloc[0] if not data['payment_terms'].value_counts().empty else 0
                unique_terms = data['payment_terms'].nunique()
                kpi_cols = st.columns(3)
                kpi_cols[0].metric('Most Common Payment Term', most_common if most_common else '-')
                kpi_cols[1].metric('Suppliers with Most Common Term', most_common_count)
                kpi_cols[2].metric('Unique Payment Terms', unique_terms)
            else:
                st.info('payment_terms column not found for KPIs.')

            # KPI for supplier types
            if 'supplier_type' in data.columns:
                unique_types = data['supplier_type'].nunique()
                st.metric('Unique Supplier Types', unique_types)
            else:
                st.info('supplier_type column not found for KPI.')

            # Table for reference
            if 'supplier' in data.columns and 'payment_terms' in data.columns:
                st.subheader('Supplier Payment Terms Table')
                if 'supplier_type' in data.columns:
                    st.dataframe(data[['supplier', 'payment_terms', 'supplier_type']], use_container_width=True)
                else:
                    st.dataframe(data[['supplier', 'payment_terms']], use_container_width=True)

        with dashboard_tabs[4]:
            st.header('Supplier Experience & Type')
            # Bar chart: supplier vs experience_years
            if 'supplier' in data.columns and 'experience_years' in data.columns:
                st.subheader('Supplier vs Experience Years')
                fig_bar = px.bar(
                    data,
                    x='supplier',
                    y='experience_years',
                    color='experience_years',
                    color_continuous_scale=CONTINUOUS_COLOR_SCALE,
                    title='Supplier Experience (Years)',
                    labels={'experience_years': 'Experience (Years)', 'supplier': 'Supplier'},
                    hover_data=['supplier', 'experience_years']
                )
                fig_bar.update_traces(marker_line_width=2)
                fig_bar.update_layout(yaxis=dict(showgrid=True), xaxis=dict(categoryorder='total descending'))
                fig_bar = apply_common_layout(fig_bar)
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info('supplier or experience_years column not found for bar chart.')
            # Pie chart: supplier_type distribution
            if 'supplier_type' in data.columns:
                st.subheader('Supplier Type Distribution')
                type_counts = data['supplier_type'].value_counts().reset_index()
                type_counts.columns = ['supplier_type', 'count']
                fig_pie = px.pie(
                    type_counts,
                    names='supplier_type',
                    values='count',
                    color='supplier_type',
                    color_discrete_sequence=CATEGORICAL_COLOR_SEQUENCE,
                    title='Supplier Type Distribution',
                    hole=0.4
                )
                fig_pie.update_traces(textinfo='percent+label', pull=[0.05 if c == type_counts['supplier_type'].iloc[0] else 0 for c in type_counts['supplier_type']])
                fig_pie = apply_common_layout(fig_pie)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info('supplier_type column not found for pie chart.')
            # Scatter: experience_years vs score
            if 'experience_years' in data.columns and 'score' in data.columns:
                st.subheader('Experience Years vs Score')
                fig_scatter = px.scatter(
                    data,
                    x='experience_years',
                    y='score',
                    color='score',
                    color_continuous_scale=CONTINUOUS_COLOR_SCALE,
                    size='experience_years',
                    hover_data=['supplier', 'experience_years', 'score'],
                    title='Experience Years vs Score',
                    labels={'experience_years': 'Experience (Years)', 'score': 'Score'}
                )
                fig_scatter.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
                fig_scatter = apply_common_layout(fig_scatter)
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.info('experience_years or score column not found for scatter plot.')

    with main_tabs[1]:  # Scoring Model Tab
        st.subheader('Select Scoring Scenario')
        scenario_labels = ['Equal Weights', 'Price-Focused', 'Quality-Focused', 'Custom Weights']
        scenario_keys = ['equal', 'price-focused', 'quality-focused', 'custom']
        if 'selected_scenario' not in st.session_state:
            st.session_state.selected_scenario = 'equal'
        cols = st.columns(len(scenario_labels))
        for i, label in enumerate(scenario_labels):
            if cols[i].button(label, key=f'scenario_{scenario_keys[i]}'):
                st.session_state.selected_scenario = scenario_keys[i]
        scenario = st.session_state.selected_scenario

        # Define custom_weights_section container for use below
        custom_weights_section = st.container()
        # --- Scenario Logic ---
        if scenario in ['equal', 'price-focused', 'quality-focused']:
            weights = get_weights(variables, scenario)
            norm_data = data.copy() # Use 'data' for normalization
            for var in variables:
                minimize = (var.lower() == 'price')
                norm_data[var] = normalize_column(data[var], minimize=minimize)
            norm_data['score'] = sum(norm_data[var] * weights[var] for var in variables)
            ranked = norm_data.sort_values('score', ascending=False)
            cols = ['supplier', 'score'] + variables
            cols = list(dict.fromkeys(cols))
            st.dataframe(ranked[cols], use_container_width=True)
            st.success(f"Best offer: {ranked.iloc[0]['supplier']} (Score: {ranked.iloc[0]['score']:.3f})")

            # Bar plot of supplier scores, highlight best offer
            highlight = ['Best' if i == 0 else 'Other' for i in range(len(ranked))]
            fig = px.bar(
                ranked,
                x='supplier',
                y='score',
                color=highlight,
                color_discrete_map={'Best': 'green', 'Other': 'gray'},
                title='Supplier Scores (Best Offer Highlighted)'
            )
            st.plotly_chart(fig, use_container_width=True)
        elif scenario == 'custom':
            with custom_weights_section:
                st.header('Custom Weights')
                st.write('Set custom weights for each variable (must sum to 1.0)')
                custom_weights = {}
                total_weight = 0.0
                for var in variables:
                    default = round(1/len(variables), 2)
                    custom_weights[var] = st.slider(f"Weight for {var}", 0.0, 1.0, default, 0.01, key=f'custom_{var}')
                    total_weight += custom_weights[var]
                if abs(total_weight - 1.0) > 0.01:
                    st.warning(f"Weights sum to {total_weight:.2f}. They will be normalized to sum to 1.0.")
                    factor = 1.0 / total_weight if total_weight > 0 else 0
                    for var in custom_weights:
                        custom_weights[var] *= factor
                st.write('Custom Weights:', {k: round(v, 3) for k, v in custom_weights.items()})
            st.subheader('Scenario: Custom Weights')
            norm_data = data.copy() # Use 'data' for normalization
            for var in variables:
                minimize = (var.lower() == 'price')
                norm_data[var] = normalize_column(data[var], minimize=minimize)
            norm_data['score'] = sum(norm_data[var] * custom_weights[var] for var in variables)
            ranked = norm_data.sort_values('score', ascending=False)
            cols = ['supplier', 'score'] + variables
            cols = list(dict.fromkeys(cols))
            st.dataframe(ranked[cols], use_container_width=True)
            st.success(f"Best offer: {ranked.iloc[0]['supplier']} (Score: {ranked.iloc[0]['score']:.3f})")
        # Save ranked and scenario for use in other tabs
        st.session_state['ranked'] = ranked
        st.session_state['scenario'] = scenario
        if scenario in ['equal', 'price-focused', 'quality-focused']:
            st.session_state['weights'] = weights
            st.session_state['custom_weights'] = None
        else:
            st.session_state['weights'] = None
            st.session_state['custom_weights'] = custom_weights
        st.session_state['variables'] = variables
        st.session_state['filtered_data'] = data # Use 'data' for filtered_data
        st.session_state['original_data'] = original_data

    with main_tabs[2]:  # SSBI Tab
        # Retrieve from session state
        ranked = st.session_state.get('ranked', None)
        scenario = st.session_state.get('scenario', None)
        weights = st.session_state.get('weights', None)
        custom_weights = st.session_state.get('custom_weights', None)
        variables = st.session_state.get('variables', None)
        filtered_data = st.session_state.get('filtered_data', None)
        original_data = st.session_state.get('original_data', None)
        if ranked is not None and variables is not None:
            # --- Price Summary KPI Cards (very top of SSBI tab, above sub-tabs) ---
            if 'price' in filtered_data.columns:
                price_min = filtered_data['price'].min()
                price_max = filtered_data['price'].max()
                price_mean = filtered_data['price'].mean()
                price_median = filtered_data['price'].median()
                price_std = filtered_data['price'].std()
                kpi_cols = st.columns(5)
                kpi_cols[0].metric("Min Price", f"{price_min:,.2f}")
                kpi_cols[1].metric("Max Price", f"{price_max:,.2f}")
                kpi_cols[2].metric("Mean Price", f"{price_mean:,.2f}")
                kpi_cols[3].metric("Median Price", f"{price_median:,.2f}")
                kpi_cols[4].metric("Std Dev", f"{price_std:,.2f}")
                st.markdown("---")
            ssbi_tab_labels = [
                'Score Composition',
                'Price Distribution',
                'Supplier Profiles',
                'Price vs. Quality/Technical',
                'Supplier Comparison',
                'Correlation Heatmap',
                'Pareto Chart',
                'Filter Suppliers',
            ]
            ssbi_tabs = st.tabs(ssbi_tab_labels)
            with ssbi_tabs[0]:
                if scenario == 'custom' and custom_weights is not None:
                    score_composition_tab(ranked, variables, custom_weights)
                else:
                    score_composition_tab(ranked, variables, weights)
            with ssbi_tabs[1]:
                price_distribution_tab(filtered_data)
            with ssbi_tabs[2]:
                supplier_profiles_tab(ranked, variables)
            with ssbi_tabs[3]:
                price_vs_quality_technical_tab(filtered_data, original_data)
            with ssbi_tabs[4]:
                supplier_comparison_table(filtered_data)
            with ssbi_tabs[5]:
                correlation_heatmap_tab(filtered_data)
            with ssbi_tabs[6]:
                pareto_chart_tab(filtered_data)
            with ssbi_tabs[7]:
                st.subheader('Filter Suppliers')
                filter_dict = {}
                filtered_data_tab = data.copy()
                for var in variables:
                    min_val = float(filtered_data_tab[var].min()) if not pd.isnull(filtered_data_tab[var].min()) else 0.0
                    max_val = float(filtered_data_tab[var].max()) if not pd.isnull(filtered_data_tab[var].max()) else 1.0
                    if min_val == max_val:
                        st.info(f"All values for '{var}' are {min_val}. No filter applied.")
                        f_min, f_max = min_val, max_val
                    else:
                        step = 1.0 if max_val - min_val > 10 else 0.01
                        f_min, f_max = st.slider(f"{var} range", min_val, max_val, (min_val, max_val), step=step, key=f'filtertab_{var}')
                    filter_dict[var] = (f_min, f_max)
                    filtered_data_tab = filtered_data_tab[(filtered_data_tab[var] >= f_min) & (filtered_data_tab[var] <= f_max)]
                if st.button('Reset Filters', key='reset_filters_tab'):
                    st.rerun()
                st.caption('Adjust the sliders to filter suppliers by variable range.')

                # Show a table of numeric variables for the filtered suppliers (move above plots)
                numeric_cols = get_numeric_columns(filtered_data_tab)
                table_cols = ['supplier'] + [col for col in numeric_cols if col != 'supplier']
                if table_cols:
                    st.subheader('Filtered Numeric Variables Table')
                    st.dataframe(filtered_data_tab[table_cols], use_container_width=True)

                # Robust error handling for empty filter result
                if filtered_data_tab.empty:
                    st.warning('No suppliers match the current filter settings.')
                else:
                    plot_bubble_price_quality_delivery(filtered_data_tab)
                    plot_line_lead_time(filtered_data_tab)
                    plot_bar_delivery_time(filtered_data_tab)
                    plot_lollipop_experience(filtered_data_tab)
                    plot_dot_warranty(filtered_data_tab)
                    plot_stackedbar_compliance(filtered_data_tab)
                    plot_violin_price(filtered_data_tab)
                    plot_scatter_price_delivery(filtered_data_tab)
        else:
            st.info('Please select a scenario in the Scoring Model tab first.')

    with main_tabs[3]:  # Advanced Analytics Tab
        ranked = st.session_state.get('ranked', None)
        variables = st.session_state.get('variables', None)
        filtered_data = st.session_state.get('filtered_data', None)
        if ranked is not None and variables is not None:
            advanced_analytics_tab(ranked, filtered_data, variables)
        else:
            st.info('Please select a scenario in the Scoring Model tab first.')

    with main_tabs[4]:  # Negotiation Strategy Tab
        st.header("Smart Negotiation Strategy & Trade-Off Analysis")

        def get_negotiation_data():
            # Use main uploaded/filtered data from session state
            df = st.session_state.get('filtered_data', None)
            if df is None or df.empty:
                st.warning("No supplier data available. Please upload and filter your data in the Scoring Model tab.")
                return None
            # Auto-select columns
            col_map = {c.lower(): c for c in df.columns}
            price_col = col_map.get('price')
            quality_col = col_map.get('quality')
            delivery_col = col_map.get('delivery') or col_map.get('delivery_time') or col_map.get('delivery_time_days')
            if not (price_col and quality_col and delivery_col):
                st.error("Your data must include columns named 'price', 'quality', and 'delivery' (or 'delivery_time'/'delivery_time_days'). Please check your file.")
                return None
            return df.copy(), price_col, quality_col, delivery_col

        def calculate_scores(df, price_col, quality_col, delivery_col):
            # Score: higher quality, lower price, faster delivery
            df = df.copy()
            df['score'] = df[quality_col] / (df[price_col] * (1 + 0.01 * df[delivery_col]))
            return df

        def show_feature_importance(df, price_col, quality_col, delivery_col):
            st.subheader("Variable Impact Analysis")
            X_cols = list(dict.fromkeys([price_col, quality_col, delivery_col]))
            X = df[X_cols]
            y = df['score']
            # Drop rows with NaN in X or y
            mask = ~(X.isna().any(axis=1) | y.isna())
            X_clean = X[mask]
            y_clean = y[mask]
            dropped = len(X) - len(X_clean)
            if dropped > 0:
                st.warning(f"{dropped} row(s) with missing values were excluded from feature importance analysis.")
            if len(X_clean) == 0:
                st.info("No valid data for feature importance analysis.")
                return
            rf = RandomForestRegressor(random_state=42)
            rf.fit(X_clean, y_clean)
            importances = pd.Series(rf.feature_importances_, index=X_clean.columns).sort_values(ascending=False)
            st.write("Feature importances (impact on selection score):")
            st.bar_chart(importances)

        def show_tradeoff_matrix(df, price_col, quality_col, delivery_col):
            st.subheader("Trade-Off Matrix")
            tradeoff_vars = [price_col, quality_col, delivery_col]
            if all(v in df.columns for v in tradeoff_vars):
                fig = px.scatter_3d(
                    df,
                    x=tradeoff_vars[0],
                    y=tradeoff_vars[1],
                    z=tradeoff_vars[2],
                    color='winner',
                    text='supplier',
                    labels={v: v.capitalize() for v in tradeoff_vars},
                    title=f"Supplier Trade-Off Matrix: {tradeoff_vars[0].capitalize()} vs. {tradeoff_vars[1].capitalize()} vs. {tradeoff_vars[2].capitalize()}"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Trade-off matrix requires 'price', 'quality', and 'delivery' columns.")

        def show_ranked_table(df, price_col, quality_col, delivery_col, threshold_score):
            st.subheader("Ranked Supplier Table")
            table_cols = ['supplier', price_col, quality_col, delivery_col, 'score', 'winner']
            table_cols = [col for col in table_cols if col in df.columns]
            df_table = df[table_cols].copy()
            df_table = df_table.loc[:, ~df_table.columns.duplicated()].reset_index(drop=True)
            try:
                st.dataframe(
                    df_table.style.background_gradient(subset=['score'], cmap='Greens'),
                    use_container_width=True
                )
            except Exception as e:
                st.warning(f"Could not apply styling due to: {e}. Showing plain table.")
                st.dataframe(df_table, use_container_width=True)
            st.info(f"Current best supplier(s): {', '.join(df.loc[df['winner'], 'supplier'])}")
            st.write(f"Acceptance threshold (score of last winner): **{threshold_score:.4f}**")

        def negotiation_advice(df, price_col, quality_col, delivery_col, threshold_score):
            st.subheader("Intelligent Negotiation Advice (Optimal Set Benchmarking)")
            advice = []
            # Only consider suppliers in the optimal set (winners)
            winners = df[df['winner']].copy()
            if winners.empty:
                st.info("No suppliers in the optimal set to benchmark.")
                return
            # Find the best values among winners
            min_price = winners[price_col].min()
            max_quality = winners[quality_col].max()
            min_delivery = winners[delivery_col].min()
            for idx, row in winners.iterrows():
                suggestions = []
                # Price: match lowest
                if row[price_col] > min_price:
                    diff = row[price_col] - min_price
                    suggestions.append(f"reduce price by €{diff:.2f} to match the lowest winner's price")
                # Quality: match highest
                if row[quality_col] < max_quality:
                    diff = max_quality - row[quality_col]
                    suggestions.append(f"increase quality by {diff:.2f} to match the highest winner's quality")
                # Delivery: match fastest
                if row[delivery_col] > min_delivery:
                    diff = row[delivery_col] - min_delivery
                    suggestions.append(f"reduce delivery time by {diff:.2f} days to match the fastest winner's delivery")
                if suggestions:
                    msg = f"{row['supplier']}: " + ", and ".join(suggestions) + "."
                else:
                    msg = f"{row['supplier']}: Already matches the best terms among winners."
                advice.append(msg)
            if advice:
                for tip in advice:
                    st.info(tip)
            else:
                st.write("All suppliers in the optimal set already match the best terms.")

        # --- Main logic ---
        result = get_negotiation_data()
        if result is not None:
            df, price_col, quality_col, delivery_col = result
            # Calculate scores and rank
            df = calculate_scores(df, price_col, quality_col, delivery_col)
            top_k = st.number_input("Number of winners (top_k)", min_value=1, max_value=len(df), value=min(3, len(df)), key='neg_topk')
            df = df.sort_values('score', ascending=False).reset_index(drop=True)
            df['winner'] = False
            df.loc[:top_k-1, 'winner'] = True
            threshold_score = df.loc[top_k-1, 'score']
            # Show analysis
            show_feature_importance(df, price_col, quality_col, delivery_col)
            show_tradeoff_matrix(df, price_col, quality_col, delivery_col)
            show_ranked_table(df, price_col, quality_col, delivery_col, threshold_score)
            negotiation_advice(df, price_col, quality_col, delivery_col, threshold_score)

    with main_tabs[5]:  # What-If Analysis Tab
        st.header('What-If Analysis')
        if 'supplier' in data.columns and 'score' in data.columns:
            suppliers = data['supplier'].unique().tolist()
            selected_supplier = st.selectbox('Select Supplier to Simulate', suppliers)
            orig_row = data[data['supplier'] == selected_supplier].iloc[0]
            st.write('Original Values:')
            st.write(orig_row)
            # --- Feature Importance for Score ---
            feature_vars = [col for col in data.select_dtypes(include='number').columns if col not in ['score']]
            X = data[feature_vars]
            y = data['score']
            rf = RandomForestRegressor(random_state=42)
            rf.fit(X, y)
            importances = rf.feature_importances_
            top_vars = [feature_vars[i] for i in importances.argsort()[::-1][:5]]  # Top 5 impactful variables
            st.info(f"Most impactful variables for score: {', '.join(top_vars)}")
            # Sliders for what-if (for top variables)
            sim_data = data.copy()
            new_vals = {}
            for var in top_vars:
                min_val = float(data[var].min())
                max_val = float(data[var].max())
                default_val = float(orig_row[var])
                step = 1.0 if max_val - min_val > 10 else 0.01
                new_val = st.slider(f"Simulate {var}", min_val, max_val, default_val, step=step, key=f"simulate_{var}")
                new_vals[var] = new_val
                sim_data.loc[sim_data['supplier'] == selected_supplier, var] = new_val
            # Recalculate score and rank
            # Use the same formula as before if available, else just recalc score as sum of normalized top_vars
            if all(v in ['price', 'quality', 'delivery_time_days'] for v in top_vars):
                sim_data['score'] = sim_data['quality'] / (sim_data['price'] * (1 + 0.01 * sim_data['delivery_time_days']))
            else:
                # Fallback: sum of normalized top_vars
                for var in top_vars:
                    sim_data[var + '_norm'] = (sim_data[var] - sim_data[var].min()) / (sim_data[var].max() - sim_data[var].min() + 1e-9)
                sim_data['score'] = sim_data[[v + '_norm' for v in top_vars]].sum(axis=1)
            sim_data = sim_data.sort_values('score', ascending=False).reset_index(drop=True)
            sim_data['rank'] = sim_data['score'].rank(ascending=False, method='min').astype(int)
            # Show new score and rank for selected supplier
            new_row = sim_data[sim_data['supplier'] == selected_supplier].iloc[0]
            st.success(f"New Score: {new_row['score']:.4f} | New Rank: {new_row['rank']}")
            st.dataframe(sim_data[['supplier'] + top_vars + ['score', 'rank']], use_container_width=True)
        else:
            st.info('What-If Analysis requires columns: supplier and score.')

    # Download results (remains outside as before)
    csv = ranked[['supplier', 'score'] + variables].to_csv(index=False)
    st.download_button(
        label="Download results as CSV",
        data=csv,
        file_name=f"tender_results_custom_weights.csv",
        mime='text/csv'
    )

else:
    st.info('Please upload a CSV file to begin using the controls in the sidebar.')

# --- Set Streamlit and Plotly theme based on dark mode ---
# The dark mode logic is now handled by Streamlit's built-in theme setting.
# Plotly charts will use the default template.

# --- Improved PDF Reporting System ---

# --- Executive Summary Page Helper ---
def add_executive_summary(pdf, summary):
    pdf.add_page()
    pdf.set_font('Arial', 'B', 20)
    pdf.cell(0, 16, 'Executive Summary', ln=True, align='C')
    pdf.ln(8)
    pdf.set_font('Arial', '', 12)
    for section in summary:
        pdf.set_font('Arial', 'B', 13)
        pdf.cell(0, 10, section['title'], ln=True)
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 8, safe_text(section['content']))
        pdf.ln(2)
    pdf.ln(4)
    pdf.set_font('Arial', 'I', 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, 'This summary provides a high-level overview of the tender evaluation.', ln=True, align='C')
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

# --- Comparison Summary Table Helper ---
def add_comparison_summary_table(pdf, df):
    if df.empty:
        return
    display_cols = ['Supplier', 'Score', 'Risk', 'Price', 'Delivery', 'Quality', 'Status']
    pdf.set_font('Arial', 'B', 12)
    col_width = max(25, int(180 / len(display_cols)))
    for col in display_cols:
        pdf.cell(col_width, 10, col, border=1, align='C', fill=True)
    pdf.ln()
    pdf.set_font('Arial', '', 11)
    for _, row in df.iterrows():
        for col in display_cols:
            val = row.get(col, '')
            if col == 'Status':
                if '[BEST]' in str(val) or '[RISK]' in str(val):
                    pdf.set_font('Arial', 'B', 11)
                else:
                    pdf.set_font('Arial', '', 11)
            pdf.cell(col_width, 10, safe_text(str(val)), border=1, align='C')
        pdf.ln()
    pdf.ln(2)
    pdf.set_font('Arial', 'I', 9)
    pdf.cell(0, 8, '[BEST] = Top performer, [RISK] = Red flag', ln=True)
    pdf.ln(2)

# --- Update PDFReport for header/footer branding ---
class PDFReport(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.section_titles = []  # For TOC
        self.section_pages = []
        self.toc_links = []
        self.subsection_titles = []  # For subsections
        self.subsection_pages = []
        self.subsection_links = []
        self.report_title = "Smart Tender Evaluation"
        self.font_heading = 'Arial'  # Use Arial as fallback
        self.font_body = 'Arial'
        # If you have Montserrat/Open Sans TTF, you can register them here
        # self.add_font('Montserrat', '', 'Montserrat-Regular.ttf', uni=True)
        # self.add_font('Montserrat', 'B', 'Montserrat-Bold.ttf', uni=True)
        # self.font_heading = 'Montserrat'
        # self.font_body = 'OpenSans'

    def header(self):
        if hasattr(self, 'is_cover') and self.is_cover:
            return  # No header on cover page
        if hasattr(self, 'is_toc') and self.is_toc:
            return  # No header on TOC
        self.set_font(self.font_heading, 'B', 12)
        self.set_text_color(40, 40, 80)
        self.cell(0, 8, self.report_title, 0, 1, 'C')
        self.ln(2)
        self.set_text_color(0, 0, 0)
    def footer(self):
        if hasattr(self, 'is_cover') and self.is_cover:
            return  # No footer on cover page
        self.set_y(-15)
        self.set_font(self.font_body, 'I', 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Page {self.page_no()}", 0, 0, 'C')
        self.set_text_color(0, 0, 0)
    def set_section_title(self, title, level=1):
        self.section_title = title
        if level == 1:
            self.section_titles.append(title)
            self.section_pages.append(self.page_no())
            self.toc_links.append(self.add_link())
            self.set_link(self.toc_links[-1], self.page_no())
        elif level == 2:
            self.subsection_titles.append(title)
            self.subsection_pages.append(self.page_no())
            self.subsection_links.append(self.add_link())
            self.set_link(self.subsection_links[-1], self.page_no())
    def add_numbered_heading(self, text, level=1, number=None):
        if level == 1:
            self.set_font(self.font_heading, 'B', 18)
            self.ln(8)
            if number is not None:
                text = f"{number}. {text}"
            self.cell(0, 12, text, ln=True, align='L')
            self.ln(2)
        elif level == 2:
            self.set_font(self.font_heading, 'B', 14)
            self.ln(4)
            if number is not None:
                text = f"{number} {text}"
            self.cell(0, 10, text, ln=True, align='L')
            self.ln(1)
        else:
            self.set_font(self.font_heading, '', 12)
            self.ln(2)
            self.cell(0, 8, text, ln=True, align='L')
            self.ln(1)

# --- Cover Page Helper ---
def add_cover_page(pdf, report_title, subtitle, developer_name):
    pdf.is_cover = True
    pdf.add_page()
    pdf.set_font(pdf.font_heading, 'B', 28)
    pdf.ln(40)
    pdf.cell(0, 20, report_title, ln=True, align='C')
    pdf.set_font(pdf.font_heading, '', 18)
    pdf.ln(8)
    pdf.cell(0, 12, subtitle, ln=True, align='C')
    pdf.ln(10)
    pdf.set_font(pdf.font_body, '', 14)
    pdf.cell(0, 10, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font(pdf.font_body, 'I', 13)
    pdf.cell(0, 10, f"Developed by {developer_name}", ln=True, align='C')
    pdf.ln(40)
    pdf.is_cover = False

# --- Table of Contents Helper ---
def add_table_of_contents(pdf):
    """Add a professional Table of Contents with page numbers."""
    pdf.add_page()
    pdf.is_toc = True
    
    # Title
    pdf.set_font(pdf.font_heading, 'B', 18)
    pdf.cell(0, 16, 'Table of Contents', ln=True, align='C')
    pdf.ln(8)
    
    # Main sections (bold, larger font)
    pdf.set_font(pdf.font_body, 'B', 12)
    for i, (title, page, link) in enumerate(zip(pdf.section_titles, pdf.section_pages, pdf.toc_links), 1):
        # Calculate available width for text (leaving space for page number)
        page_num_width = 15  # Width needed for page number
        available_width = 180 - page_num_width  # Total width minus page number space
        
        # Create the text with number
        toc_text = f"{i}. {title}"
        
        # Get text width and truncate if necessary
        text_width = pdf.get_string_width(toc_text)
        if text_width > available_width:
            # Truncate text to fit
            while text_width > available_width and len(toc_text) > 5:
                toc_text = toc_text[:-1]
                text_width = pdf.get_string_width(toc_text)
            toc_text = toc_text[:-3] + "..."
        
        # Add text and page number on same line
        pdf.cell(available_width, 10, toc_text, ln=False, link=link)
        pdf.cell(page_num_width, 10, str(page), ln=True, align='R')
    
    # Subsections (indented, regular font)
    if pdf.subsection_titles:
        pdf.ln(4)
        pdf.set_font(pdf.font_body, '', 10)
        for i, (title, page, link) in enumerate(zip(pdf.subsection_titles, pdf.subsection_pages, pdf.subsection_links), 1):
            # Calculate available width for text (leaving space for page number)
            page_num_width = 15  # Width needed for page number
            available_width = 180 - page_num_width  # Total width minus page number space
            
            # Create the indented text
            toc_text = f"   {title}"
            
            # Get text width and truncate if necessary
            text_width = pdf.get_string_width(toc_text)
            if text_width > available_width:
                # Truncate text to fit
                while text_width > available_width and len(toc_text) > 8:
                    toc_text = toc_text[:-1]
                    text_width = pdf.get_string_width(toc_text)
                toc_text = toc_text[:-3] + "..."
            
            # Add text and page number on same line
            pdf.cell(available_width, 8, toc_text, ln=False, link=link)
            pdf.cell(page_num_width, 8, str(page), ln=True, align='R')
    
    pdf.ln(4)
    pdf.is_toc = False

# --- Risk/Scoring Explanations (best-practice defaults) ---
RISK_EXPLANATIONS = {
    "Lowballing": "A bid much lower than the average, possibly unsustainable.",
    "Drip Pricing": "Price appears low but may increase with hidden costs.",
    "Drip Pricing Flag": "Supplier has high price and low quality/technical score.",
    "Market Signaling": "High price and high score, possibly signaling to competitors.",
    "Cover Bidding": "High price and low score, possibly not a serious bid.",
    "Decoy Bidding": "Bid is an outlier in price or score, possibly to distract.",
    "Bid Similarity": "Multiple bids with very similar price or score, may indicate collusion."
}

# --- Helper functions for PDF generation ---
def safe_text(text):
    """Replace en dash, em dash, right single quote, and euro sign with ASCII equivalents."""
    return (
        str(text)
        .replace('–', '-')
        .replace('—', '-')
        .replace("'", "'")
        .replace('€', 'EUR')
    )

def truncate_col(col, maxlen=15):
    """Truncate column names for better table display."""
    return col if len(col) <= maxlen else col[:maxlen-3] + '...'

# --- PDF Visual Breaks & Callouts ---
def add_horizontal_line(pdf, color=(180, 180, 180)):
    y = pdf.get_y() + 2
    pdf.set_draw_color(*color)
    pdf.set_line_width(0.7)
    pdf.line(15, y, 195, y)
    pdf.ln(8)
    pdf.set_draw_color(0, 0, 0)
    pdf.set_line_width(0.2)

def add_shaded_block(pdf, text, color=(240, 240, 255)):
    pdf.ln(2)
    pdf.set_fill_color(*color)
    pdf.set_font(pdf.font_body, 'I', 11)
    pdf.multi_cell(0, 8, safe_text(text), fill=True)
    pdf.ln(2)
    pdf.set_fill_color(255, 255, 255)

def add_callout(pdf, text, icon='[TIP]', color=(220, 240, 220)):
    pdf.ln(2)
    pdf.set_fill_color(*color)
    pdf.set_font(pdf.font_body, 'B', 11)
    pdf.cell(10, 8, icon, fill=True, border=0)
    pdf.set_font(pdf.font_body, '', 11)
    pdf.multi_cell(0, 8, safe_text(text), fill=True)
    pdf.ln(2)
    pdf.set_fill_color(255, 255, 255)

# --- Table Highlighting Helper ---
def add_table(pdf, df, max_columns=6, highlight_col=None, highlight_icon=None, highlight_rows=None):
    if df.empty:
        pdf.set_font(pdf.font_body, 'I', 10)
        pdf.cell(0, 8, safe_text("No data available."), ln=True)
        return
    display_cols = df.columns[:max_columns]
    pdf.set_font(pdf.font_heading, 'B', 10)
    col_width = max(30, int(180 / len(display_cols)))
    # Header
    for col in display_cols:
        pdf.cell(col_width, 8, safe_text(str(col)), border=1, align='C', fill=True)
    pdf.ln()
    pdf.set_font(pdf.font_body, '', 10)
    fill = False
    for idx, row in df.iterrows():
        pdf.set_fill_color(240, 240, 240) if fill else pdf.set_fill_color(255, 255, 255)
        for col in display_cols:
            cell_val = safe_text(str(row[col]))
            style = ''
            if highlight_col and col == highlight_col:
                style = 'B'
            if highlight_rows and idx in highlight_rows:
                style = 'B'
            if style:
                pdf.set_font(pdf.font_body, style, 10)
            if highlight_icon and highlight_col and col == highlight_col and highlight_rows and idx in highlight_rows:
                cell_val = f"{highlight_icon} {cell_val}"
            pdf.cell(col_width, 8, cell_val, border=1, align='C', fill=fill)
            if style:
                pdf.set_font(pdf.font_body, '', 10)
        pdf.ln()
        fill = not fill
    if len(df.columns) > max_columns:
        pdf.set_font(pdf.font_body, 'I', 8)
        pdf.cell(0, 8, safe_text(f"Table truncated to first {max_columns} columns. Download full data as CSV."), ln=True)
    pdf.ln(2)
    pdf.set_fill_color(255, 255, 255)

# --- Chart Caption Helper (with visual break) ---
def add_chart(pdf, fig, caption=None):
    if isinstance(fig, go.Figure):
        img_bytes = fig.to_image(format="png", width=800, height=400, scale=2)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            tmpfile.write(img_bytes)
            tmpfile.flush()
            pdf.image(tmpfile.name, x=15, w=180)
        if caption:
            pdf.ln(2)
            pdf.set_font(pdf.font_body, 'I', 9)
            pdf.cell(0, 8, safe_text(caption), ln=True, align='C')
        add_horizontal_line(pdf)
    else:
        pdf.set_font(pdf.font_body, 'I', 10)
        pdf.cell(0, 8, safe_text("Chart not supported in PDF export (Altair/other)."), ln=True)
        if caption:
            pdf.set_font(pdf.font_body, 'I', 9)
            pdf.cell(0, 8, safe_text(caption), ln=True, align='C')
        add_horizontal_line(pdf)

# --- Risk/Scoring Explanation Helper ---
def add_risk_explanations(pdf, risk_terms):
    pdf.ln(2)
    pdf.set_font(pdf.font_body, 'I', 9)
    pdf.set_text_color(100, 100, 180)
    for term, explanation in risk_terms.items():
        pdf.cell(0, 7, f"{term}: {explanation}", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

# --- Update generate_professional_pdf_report to use best-practice defaults ---
def generate_professional_pdf_report(sections, developer_name="Aryan Zabihi", executive_summary=None, comparison_table=None):
    # First pass: Collect section information and page numbers
    toc_data = []
    current_page = 3  # Start after cover and TOC page
    
    if executive_summary:
        toc_data.append({
            'title': 'Executive Summary',
            'level': 1,
            'number': 1,
            'page': current_page
        })
        current_page += 1
    
    section_number = 2 if executive_summary else 1
    for section in sections:
        toc_data.append({
            'title': section['title'],
            'level': 1,
            'number': section_number,
            'page': current_page
        })
        current_page += 1
        
        subsection_number = 1
        for sub in section.get('subsections', []):
            toc_data.append({
                'title': sub['title'],
                'level': 2,
                'number': f"{section_number}.{subsection_number}",
                'page': current_page
            })
            current_page += 1
            subsection_number += 1
        section_number += 1
    
    # Second pass: Generate PDF with accurate TOC
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=20)
    
    # 1. Cover Page
    add_cover_page(pdf, "Smart Tender Evaluation Report", "AI-Powered Tender Analytics & Supplier Scoring", developer_name)
    
    # 2. Table of Contents with accurate page numbers
    # Clear any existing TOC data
    pdf.section_titles = []
    pdf.section_pages = []
    pdf.toc_links = []
    pdf.subsection_titles = []
    pdf.subsection_pages = []
    pdf.subsection_links = []
    
    # Add collected TOC data to PDF object
    for item in toc_data:
        if item['level'] == 1:
            pdf.section_titles.append(item['title'])
            pdf.section_pages.append(item['page'])
            pdf.toc_links.append(None)  # No link for now
        else:  # level == 2
            pdf.subsection_titles.append(item['title'])
            pdf.subsection_pages.append(item['page'])
            pdf.subsection_links.append(None)  # No link for now
    
    # Generate TOC using the improved function
    add_table_of_contents(pdf)
    
    # 3. Executive Summary
    if executive_summary:
        pdf.set_section_title("Executive Summary", level=1)
        pdf.add_page()
        pdf.add_numbered_heading("Executive Summary", level=1, number=1)
        for section in executive_summary:
            pdf.set_font(pdf.font_heading, 'B', 13)
            pdf.cell(0, 10, section['title'], ln=True)
            pdf.set_font(pdf.font_body, '', 12)
            pdf.multi_cell(0, 8, safe_text(section['content']))
            pdf.ln(2)
        pdf.ln(4)
        pdf.set_font(pdf.font_body, 'I', 10)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(0, 8, 'This summary provides a high-level overview of the tender evaluation.', ln=True, align='C')
        pdf.set_text_color(0, 0, 0)
        pdf.ln(4)
    
    # 4. Main Sections
    section_number = 2 if executive_summary else 1
    for section in sections:
        pdf.set_section_title(section['title'], level=1)
        pdf.add_page()
        pdf.add_numbered_heading(section['title'], level=1, number=section_number)
        
        subsection_number = 1
        for sub in section.get('subsections', []):
            pdf.set_section_title(sub['title'], level=2)
            pdf.add_numbered_heading(sub['title'], level=2, number=f"{section_number}.{subsection_number}")
            
            # --- Callout for best supplier or key finding ---
            if 'callout' in sub:
                add_callout(pdf, sub['callout'])
            # --- Summary ---
            if 'summary' in sub:
                pdf.set_font(pdf.font_body, '', 12)
                pdf.multi_cell(0, 8, safe_text(sub['summary']))
                pdf.ln(2)
            # --- Table with highlights ---
            if 'table' in sub:
                df = sub['table']
                highlight_col = None
                highlight_icon = None
                highlight_rows = None
                # Highlight logic: top performer (first row), red flag (risk > 0.7)
                if 'score' in df.columns:
                    highlight_col = 'score'
                    highlight_icon = '[BEST]'
                    highlight_rows = [df['score'].astype(float).idxmax()]
                elif 'total_risk' in df.columns:
                    highlight_col = 'total_risk'
                    highlight_icon = '[RISK]'
                    highlight_rows = df.index[df['total_risk'].astype(float) > 0.7].tolist()
                # Sort by impact if possible
                if 'score' in df.columns:
                    df = df.sort_values('score', ascending=False)
                elif 'total_risk' in df.columns:
                    df = df.sort_values('total_risk', ascending=False)
                add_table(pdf, df, highlight_col=highlight_col, highlight_icon=highlight_icon, highlight_rows=highlight_rows)
            # --- Chart with caption and break ---
            if 'chart' in sub:
                add_chart(pdf, sub['chart'], caption=sub.get('chart_caption'))
            # --- Risk/Scoring explanations ---
            if 'risk_explanations' in sub and sub['risk_explanations']:
                add_risk_explanations(pdf, RISK_EXPLANATIONS)
            # --- Inline commentary/callout ---
            if 'commentary' in sub:
                add_callout(pdf, sub['commentary'], icon='[NOTE]', color=(255, 255, 220))
            if 'export_note' in sub:
                pdf.set_font(pdf.font_body, 'I', 9)
                pdf.set_text_color(180, 80, 80)
                pdf.multi_cell(0, 8, safe_text(sub['export_note']))
                pdf.set_text_color(0, 0, 0)
            add_horizontal_line(pdf)
            subsection_number += 1
        section_number += 1
    
    return bytes(pdf.output(dest='S'))

# --- PDF Report Generation Logic ---

# --- Improved PDF Report Generation Logic ---
if generate_report:
    st.info("PDF generation started...")
    try:
        status_text.text("Preparing data for report... (10%)")
        progress_bar.progress(10)
        
        st.info("Preparing data for report...")
        selected_sections = [k for k, v in report_options.items() if v]
        sections = []
        if "Dashboard" in selected_sections:
            dashboard_subsections = []
            data = st.session_state.get('filtered_data', pd.DataFrame())
            # 1. Supplier Comparison (Price)
            if 'supplier' in data.columns and 'price' in data.columns:
                sorted_df = data.sort_values('price')
                fig_price = px.bar(
                    sorted_df,
                    x='supplier',
                    y='price',
                    color='price',
                    color_continuous_scale=CONTINUOUS_COLOR_SCALE,
                    title='Supplier vs Price',
                    labels={'price': 'Price', 'supplier': 'Supplier'},
                    hover_data=['supplier', 'price']
                )
                fig_price.update_traces(marker_line_width=2)
                fig_price.update_layout(yaxis=dict(showgrid=True), xaxis=dict(categoryorder='total ascending'))
                fig_price = apply_common_layout(fig_price)
                table_price = sorted_df[['supplier', 'price']].copy()
                table_price['price'] = table_price['price'].map(lambda x: f"{x:.1f}")
                table_price.columns = [truncate_col(c) for c in table_price.columns]
                dashboard_subsections.append({
                    'title': 'Supplier Comparison (Price)',
                    'summary': 'Bar chart and table of supplier prices (best deals first).',
                    'table': table_price,
                    'chart': fig_price,
                    'chart_caption': 'Supplier vs Price (Best Deals First)'
                })
            # 2. Supplier Comparison (Quality)
            if 'supplier' in data.columns and 'quality' in data.columns:
                fig_quality = px.bar(
                    data,
                    x='supplier',
                    y='quality',
                    color='quality',
                    color_continuous_scale=CONTINUOUS_COLOR_SCALE,
                    title='Supplier vs Quality',
                    labels={'quality': 'Quality', 'supplier': 'Supplier'},
                    hover_data=['supplier', 'quality']
                )
                fig_quality.update_traces(marker_line_width=2)
                fig_quality.update_layout(yaxis=dict(showgrid=True), xaxis=dict(categoryorder='total descending'))
                fig_quality = apply_common_layout(fig_quality)
                table_quality = data[['supplier', 'quality']].copy()
                table_quality['quality'] = table_quality['quality'].map(lambda x: f"{x:.1f}")
                table_quality.columns = [truncate_col(c) for c in table_quality.columns]
                dashboard_subsections.append({
                    'title': 'Supplier Comparison (Quality)',
                    'summary': 'Bar chart and table of supplier quality.',
                    'table': table_quality,
                    'chart': fig_quality,
                    'chart_caption': 'Supplier vs Quality'
                })
            # 3. Supplier Profile Radar Chart
            radar_cols = ['price', 'delivery_time_days', 'warranty_months', 'quality']
            if all(col in data.columns for col in radar_cols) and 'supplier' in data.columns:
                norm_df = data.copy()
                norm_df['price'] = normalize_column(norm_df['price'], minimize=True)
                norm_df['delivery_time_days'] = normalize_column(norm_df['delivery_time_days'], minimize=True)
                norm_df['warranty_months'] = normalize_column(norm_df['warranty_months'], minimize=False)
                norm_df['quality'] = normalize_column(norm_df['quality'], minimize=False)
                fig_radar = go.Figure()
                for i, row in norm_df.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[row[c] for c in radar_cols],
                        theta=radar_cols,
                        fill='toself',
                        name=row['supplier'],
                        line=dict(width=3)
                    ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    title="Supplier Radar Chart (Normalized)",
                )
                fig_radar = apply_common_layout(fig_radar)
                dashboard_subsections.append({
                    'title': 'Supplier Profile Radar Chart',
                    'summary': 'Radar chart of normalized supplier profiles.',
                    'chart': fig_radar,
                    'chart_caption': 'Supplier Profile Radar Chart (Normalized)'
                })
            # 4. Supplier Total Cost Table
            if all(col in data.columns for col in ['price', 'discount', 'shipping_cost', 'supplier']):
                df_cost = data.copy()
                df_cost['Total Cost'] = df_cost['price'] - df_cost['discount'] + df_cost['shipping_cost']
                table_cost = df_cost[['supplier', 'price', 'discount', 'shipping_cost', 'Total Cost']].copy()
                for col in ['price', 'discount', 'shipping_cost', 'Total Cost']:
                    table_cost[col] = table_cost[col].map(lambda x: f"{x:.1f}")
                table_cost.columns = [truncate_col(c) for c in table_cost.columns]
                dashboard_subsections.append({
                    'title': 'Supplier Total Cost Table',
                    'summary': 'Table of supplier total cost (price – discount + shipping cost).',
                    'table': table_cost
                })
            # 5. Delivery & Lead Time Insights (KPI and scatter)
            if 'lead_time_days' in data.columns and 'delivery_time_days' in data.columns and 'quantity' in data.columns:
                avg_lead = data['lead_time_days'].mean()
                min_delivery = data['delivery_time_days'].min()
                fig_scatter = px.scatter(
                    data,
                    x='lead_time_days',
                    y='delivery_time_days',
                    size='quantity',
                    color='quantity',
                    color_continuous_scale=CONTINUOUS_COLOR_SCALE,
                    hover_data=['supplier', 'lead_time_days', 'delivery_time_days', 'quantity'],
                    title='Lead Time vs Delivery Time',
                    labels={'lead_time_days': 'Lead Time (days)', 'delivery_time_days': 'Delivery Time (days)', 'quantity': 'Quantity'}
                )
                fig_scatter.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
                fig_scatter = apply_common_layout(fig_scatter)
                dashboard_subsections.append({
                    'title': 'Delivery & Lead Time Insights',
                    'summary': f'Average lead time: {avg_lead:.1f} days. Min delivery time: {min_delivery:.1f} days.',
                    'chart': fig_scatter,
                    'chart_caption': 'Lead Time vs Delivery Time (Bubble = Quantity)'
                })
            # 6. Supplier Lead Time (bar)
            if 'supplier' in data.columns and 'lead_time_days' in data.columns:
                fig_lead = px.bar(
                    data,
                    x='supplier',
                    y='lead_time_days',
                    color='lead_time_days',
                    color_continuous_scale=CONTINUOUS_COLOR_SCALE,
                    hover_data=['supplier', 'lead_time_days'],
                    title='Supplier vs Lead Time (days)',
                    labels={'supplier': 'Supplier', 'lead_time_days': 'Lead Time Days'}
                )
                fig_lead.update_traces(marker_line_width=2)
                fig_lead.update_layout(yaxis=dict(showgrid=True), xaxis=dict(categoryorder='total descending'))
                fig_lead = apply_common_layout(fig_lead)
                dashboard_subsections.append({
                    'title': 'Supplier Lead Time',
                    'summary': 'Bar chart of supplier lead time (days).',
                    'chart': fig_lead,
                    'chart_caption': 'Supplier vs Lead Time (days)'
                })
            # 7. Compliance Distribution (pie)
            if 'compliance' in data.columns:
                compliance_counts = data['compliance'].value_counts(dropna=False).reset_index()
                compliance_counts.columns = ['Compliance', 'Count']
                fig_pie = px.pie(
                    compliance_counts,
                    names='Compliance',
                    values='Count',
                    color='Compliance',
                    color_discrete_sequence=CATEGORICAL_COLOR_SEQUENCE,
                    title='Compliance (Yes/No)',
                    hole=0.4
                )
                fig_pie.update_traces(textinfo='percent+label', pull=[0.05 if c == compliance_counts['Compliance'].iloc[0] else 0 for c in compliance_counts['Compliance']])
                fig_pie = apply_common_layout(fig_pie)
                dashboard_subsections.append({
                    'title': 'Compliance Distribution',
                    'summary': 'Pie chart of supplier compliance (Yes/No).',
                    'chart': fig_pie,
                    'chart_caption': 'Compliance (Yes/No)'
                })
            # 8. Supplier Compliance & Certifications Overview (table)
            if all(col in data.columns for col in ['supplier', 'compliance', 'compliance_notes', 'certifications']):
                merged_table = data[['supplier', 'compliance', 'compliance_notes', 'certifications']].copy()
                merged_table.columns = [truncate_col(c) for c in merged_table.columns]
                dashboard_subsections.append({
                    'title': 'Supplier Compliance & Certifications Overview',
                    'summary': 'Table of supplier compliance notes and certifications.',
                    'table': merged_table
                })
            # 9. Payment & Terms (KPI and table)
            if 'payment_terms' in data.columns:
                most_common = data['payment_terms'].mode().iloc[0] if not data['payment_terms'].mode().empty else None
                most_common_count = data['payment_terms'].value_counts().iloc[0] if not data['payment_terms'].value_counts().empty else 0
                unique_terms = data['payment_terms'].nunique()
                summary = f"Most common payment term: {most_common} ({most_common_count} suppliers). Unique payment terms: {unique_terms}."
                table = data[['supplier', 'payment_terms']] if 'supplier' in data.columns else data[['payment_terms']]
                table.columns = [truncate_col(c) for c in table.columns]
                dashboard_subsections.append({
                    'title': 'Payment & Terms',
                    'summary': summary,
                    'table': table
                })
            # 10. Supplier Types (table and pie)
            if 'supplier_type' in data.columns:
                unique_types = data['supplier_type'].nunique()
                table_types = data[['supplier', 'supplier_type']] if not data.empty else pd.DataFrame()
                table_types.columns = [truncate_col(c) for c in table_types.columns]
                dashboard_subsections.append({
                    'title': 'Supplier Types',
                    'summary': f'Unique supplier types: {unique_types}.',
                    'table': table_types
                })
                type_counts = data['supplier_type'].value_counts().reset_index()
                type_counts.columns = ['supplier_type', 'count']
                fig_type = px.pie(
                    type_counts,
                    names='supplier_type',
                    values='count',
                    title='Supplier Type Distribution',
                    hole=0.4
                )
                fig_type.update_traces(textinfo='percent+label', pull=[0.05 if c == type_counts['supplier_type'].iloc[0] else 0 for c in type_counts['supplier_type']])
                fig_type = apply_common_layout(fig_type)
                dashboard_subsections.append({
                    'title': 'Supplier Type Distribution',
                    'summary': 'Pie chart of supplier type distribution.',
                    'chart': fig_type,
                    'chart_caption': 'Supplier Type Distribution'
                })
            # 11. Supplier Experience (bar)
            if 'supplier' in data.columns and 'experience_years' in data.columns:
                fig_bar = px.bar(
                    data,
                    x='supplier',
                    y='experience_years',
                    color='experience_years',
                    color_continuous_scale=CONTINUOUS_COLOR_SCALE,
                    title='Supplier Experience (Years)',
                    labels={'experience_years': 'Experience (Years)', 'supplier': 'Supplier'},
                    hover_data=['supplier', 'experience_years']
                )
                fig_bar.update_traces(marker_line_width=2)
                fig_bar.update_layout(yaxis=dict(showgrid=True), xaxis=dict(categoryorder='total descending'))
                fig_bar = apply_common_layout(fig_bar)
                dashboard_subsections.append({
                    'title': 'Supplier Experience',
                    'summary': 'Bar chart of supplier experience (years).',
                    'chart': fig_bar,
                    'chart_caption': 'Supplier Experience (Years)'
                })
            # 12. Experience Years vs Score (scatter)
            if 'experience_years' in data.columns and 'score' in data.columns:
                fig_scatter = px.scatter(
                    data,
                    x='experience_years',
                    y='score',
                    color='score',
                    color_continuous_scale=CONTINUOUS_COLOR_SCALE,
                    size='experience_years',
                    hover_data=['supplier', 'experience_years', 'score'],
                    title='Experience Years vs Score',
                    labels={'experience_years': 'Experience (Years)', 'score': 'Score'}
                )
                fig_scatter.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
                fig_scatter = apply_common_layout(fig_scatter)
                dashboard_subsections.append({
                    'title': 'Experience Years vs Score',
                    'summary': 'Scatter plot of experience years vs score.',
                    'chart': fig_scatter,
                    'chart_caption': 'Experience Years vs Score'
                })
            # Add Dashboard section with improved title
            sections.append({'title': 'Dashboard: Supplier Analysis Overview', 'subsections': dashboard_subsections})
            
            progress_bar.progress(30)
            status_text.text("Building Dashboard sections... (30%)")
        if "Scoring Model" in selected_sections:
            ranked = st.session_state.get('ranked', pd.DataFrame())
            variables = st.session_state.get('variables', [])
            scenario = st.session_state.get('scenario', None)
            weights = st.session_state.get('weights', None)
            custom_weights = st.session_state.get('custom_weights', None)
            scoring_subsections = []
            # Scenario Results
            if not ranked.empty and 'supplier' in ranked.columns and 'score' in ranked.columns:
                fig = px.bar(ranked, x='supplier', y='score', title='Supplier Scores', color='score', color_continuous_scale='Turbo')
                scoring_subsections.append({
                    'title': 'Scenario Results',
                    'summary': f"Top supplier: {ranked.iloc[0]['supplier']} (Score: {ranked.iloc[0]['score']:.1f})",
                    'chart': fig,
                    'chart_caption': 'Supplier Scores'
                })
            if scoring_subsections:
                sections.append({'title': 'Scoring Model: Supplier Ranking & Evaluation', 'subsections': scoring_subsections})
                
                progress_bar.progress(40)
                status_text.text("Building Scoring Model sections... (40%)")
        if "SSBI" in selected_sections:
            ranked = st.session_state.get('ranked', pd.DataFrame())
            scenario = st.session_state.get('scenario', None)
            weights = st.session_state.get('weights', None)
            custom_weights = st.session_state.get('custom_weights', None)
            variables = st.session_state.get('variables', [])
            filtered_data = st.session_state.get('filtered_data', pd.DataFrame())
            original_data = st.session_state.get('original_data', pd.DataFrame())
            ssbi_subsections = []
            # 1. Price Summary KPIs (summary only, no table)
            if 'price' in filtered_data.columns:
                price_min = filtered_data['price'].min()
                price_max = filtered_data['price'].max()
                price_mean = filtered_data['price'].mean()
                price_median = filtered_data['price'].median()
                price_std = filtered_data['price'].std()
                ssbi_subsections.append({
                    'title': 'Price Summary KPIs',
                    'summary': f"Min: {price_min:.1f}, Max: {price_max:.1f}, Mean: {price_mean:.1f}, Median: {price_median:.1f}, Std Dev: {price_std:.1f}"
                })
            # 2. Score Composition (only chart, no table)
            if not ranked.empty and variables:
                if scenario == 'custom' and custom_weights is not None:
                    contrib_df = pd.DataFrame({var: ranked[var] * custom_weights[var] for var in variables})
                else:
                    contrib_df = pd.DataFrame({var: ranked[var] * weights[var] for var in variables})
                contrib_df['supplier'] = ranked['supplier']
                contrib_df = contrib_df.set_index('supplier')
                fig_sc = px.bar(
                    contrib_df,
                    x=contrib_df.index,
                    y=variables,
                    title='Score Composition',
                    labels={'value': 'Score Contribution', 'supplier': 'Supplier'},
                    template='plotly_white',
                    width=1200,
                    height=500
                )
                fig_sc.update_layout(
                    barmode='stack',
                    xaxis_title='Supplier',
                    yaxis_title='Score',
                    xaxis_tickangle=-45,
                    legend_title_text='Variable',
                    margin=dict(l=40, r=40, t=60, b=120)
                )
                ssbi_subsections.append({
                    'title': 'Score Composition',
                    'summary': 'Stacked bar chart of score composition by variable.',
                    'chart': fig_sc,
                    'chart_caption': 'Score Composition (Stacked Bar)'
                })
            # 3. Price Distribution (table and chart)
            if 'price' in filtered_data.columns and 'supplier' in filtered_data.columns:
                table_pd = filtered_data[['supplier', 'price']].copy()
                table_pd['price'] = table_pd['price'].map(lambda x: f"{x:.1f}")
                table_pd.columns = [truncate_col(c) for c in table_pd.columns]
                fig_pd = go.Figure()
                fig_pd.add_trace(go.Box(
                    y=filtered_data['price'],
                    name='Price',
                    boxpoints=False,
                    marker_color='rgba(31,119,180,0.5)',
                    line_color='rgba(31,119,180,1)'
                ))
                suppliers = filtered_data['supplier'].unique()
                palette = px.colors.qualitative.Dark2
                color_map = {sup: palette[i % len(palette)] for i, sup in enumerate(suppliers)}
                for sup in suppliers:
                    sub = filtered_data[filtered_data['supplier'] == sup]
                    fig_pd.add_trace(go.Scatter(
                        y=sub['price'],
                        x=['']*len(sub),
                        mode='markers',
                        marker=dict(size=10, color=color_map[sup]),
                        text=sub['supplier'],
                        hovertemplate='Supplier: %{text}<br>Price: %{y}<extra></extra>',
                        name=sup,
                        showlegend=False
                    ))
                fig_pd.update_layout(
                    title='Price Distribution of Suppliers',
                    yaxis_title='Price',
                    showlegend=False
                )
                ssbi_subsections.append({
                    'title': 'Price Distribution',
                    'summary': 'Box plot and supplier overlay for price distribution.',
                    'table': table_pd,
                    'chart': fig_pd,
                    'chart_caption': 'Price Distribution (Box + Strip)'
                })
            # 4. Supplier Profiles (Radar, only chart)
            if not ranked.empty and len(variables) >= 3:
                fig_radar = go.Figure()
                for i, row in ranked.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=[row[var] for var in variables],
                        theta=variables,
                        fill='toself',
                        name=row['supplier']
                    ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    showlegend=True,
                    template='plotly_white',
                    title="Supplier Radar Chart (Normalized)",
                )
                ssbi_subsections.append({
                    'title': 'Supplier Profiles',
                    'summary': 'Radar chart of supplier profiles across variables.',
                    'chart': fig_radar,
                    'chart_caption': 'Supplier Profiles (Radar Chart)'
                })
            # 5. Price vs. Quality/Technical (table and chart)
            if not filtered_data.empty and 'price' in filtered_data.columns and ('quality' in filtered_data.columns or 'technical' in filtered_data.columns):
                y_var = 'quality' if 'quality' in filtered_data.columns else 'technical'
                table_pq = filtered_data[['supplier', 'price', y_var]].copy()
                for col in ['price', y_var]:
                    table_pq[col] = table_pq[col].map(lambda x: f"{x:.1f}")
                table_pq.columns = [truncate_col(c) for c in table_pq.columns]
                chart = alt.Chart(filtered_data).mark_circle(size=60).encode(
                    x='price', y=y_var, color='supplier', tooltip=['supplier', 'price', y_var]
                ).properties(title='Price vs. Quality/Technical')
                ssbi_subsections.append({
                    'title': 'Price vs. Quality/Technical',
                    'summary': f'Scatter plot of price vs. {y_var}.',
                    'table': table_pq,
                    'chart': chart,
                    'chart_caption': f'Price vs. {y_var.capitalize()} (Scatter)'
                })
            # 6. Supplier Comparison Table (table only)
            if 'supplier' in filtered_data.columns:
                table_sc = filtered_data.copy()
                for col in table_sc.select_dtypes(include='number').columns:
                    table_sc[col] = table_sc[col].map(lambda x: f"{x:.1f}")
                table_sc.columns = [truncate_col(c) for c in table_sc.columns]
                ssbi_subsections.append({
                    'title': 'Supplier Comparison',
                    'summary': 'Comparison table of all suppliers and variables.',
                    'table': table_sc
                })
            # 7. Correlation Heatmap (only chart)
            corr = filtered_data.select_dtypes(include='number').corr()
            if not corr.empty:
                labels = corr.columns.tolist()
                z = corr.values
                fig_corr = go.Figure(data=go.Heatmap(
                    z=z,
                    x=labels,
                    y=labels,
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1,
                    colorbar=dict(title='Correlation')
                ))
                if len(labels) <= 12:
                    for i in range(len(labels)):
                        for j in range(len(labels)):
                            fig_corr.add_annotation(
                                x=labels[j], y=labels[i],
                                text=f'{z[i][j]:.2f}',
                                showarrow=False,
                                font=dict(size=12, color='black' if abs(z[i][j]) < 0.7 else 'white')
                            )
                fig_corr.update_layout(
                    title='Correlation Heatmap',
                    xaxis=dict(tickangle=45, side='top'),
                    yaxis=dict(autorange='reversed'),
                    width=700, height=700,
                    margin=dict(l=100, b=100, t=50, r=50)
                )
                ssbi_subsections.append({
                    'title': 'Correlation Heatmap',
                    'summary': 'Correlation matrix of numeric variables.',
                    'chart': fig_corr,
                    'chart_caption': 'Correlation Heatmap'
                })
            # 8. Pareto Chart (table and chart)
            if 'price' in filtered_data.columns:
                pareto_df = filtered_data[['supplier', 'price']].copy()
                pareto_df = pareto_df.groupby('supplier', as_index=False).sum()
                pareto_df = pareto_df.sort_values('price', ascending=False)
                pareto_df['cum_pct'] = pareto_df['price'].cumsum() / pareto_df['price'].sum() * 100
                for col in ['price', 'cum_pct']:
                    pareto_df[col] = pareto_df[col].map(lambda x: f"{x:.1f}")
                pareto_df.columns = [truncate_col(c) for c in pareto_df.columns]
                fig_pareto = go.Figure()
                fig_pareto.add_bar(x=pareto_df['supplier'], y=pareto_df['price'], name='Price')
                fig_pareto.add_scatter(x=pareto_df['supplier'], y=pareto_df['cum_pct'], name='Cumulative %', yaxis='y2')
                fig_pareto.update_layout(
                    title='Pareto Chart: Supplier Price Contribution',
                    yaxis=dict(title='Price'),
                    yaxis2=dict(title='Cumulative %', overlaying='y', side='right', range=[0, 100]),
                    xaxis_title='Supplier',
                    legend=dict(orientation='h'),
                    template='plotly_white'
                )
                ssbi_subsections.append({
                    'title': 'Pareto Chart',
                    'summary': 'Pareto chart of supplier price contribution.',
                    'table': pareto_df,
                    'chart': fig_pareto,
                    'chart_caption': 'Pareto Chart: Supplier Price Contribution'
                })
            # 9. Filter Suppliers (Filtered Numeric Table)
            numeric_cols = [col for col in filtered_data.select_dtypes(include='number').columns if col != 'supplier']
            table_filt = filtered_data[['supplier'] + numeric_cols].copy() if numeric_cols else pd.DataFrame()
            for col in numeric_cols:
                table_filt[col] = table_filt[col].map(lambda x: f"{x:.1f}")
            table_filt.columns = [truncate_col(c) for c in table_filt.columns]
            if not table_filt.empty:
                ssbi_subsections.append({
                    'title': 'Filter Suppliers',
                    'summary': 'Filtered numeric variables table for suppliers.',
                    'table': table_filt
                })
            # Add SSBI section
            sections.append({'title': 'SSBI: Self-Service Business Intelligence', 'subsections': ssbi_subsections})
            
            progress_bar.progress(50)
            status_text.text("Building SSBI sections... (50%)")
        if "Advanced Analytics" in selected_sections:
            ranked = st.session_state.get('ranked', pd.DataFrame())
            variables = st.session_state.get('variables', [])
            filtered_data = st.session_state.get('filtered_data', pd.DataFrame())
            advanced_subsections = []
            # 1. Anomaly Detection
            if not filtered_data.empty and len(variables) >= 2:
                model = IsolationForest(contamination=0.1, random_state=42)
                anomalies = model.fit_predict(filtered_data[variables].fillna(0))
                ranked = ranked.copy()
                ranked['anomaly'] = anomalies
                outliers = ranked[ranked['anomaly'] == -1]
                cols = ['supplier', 'score'] + variables
                cols = list(dict.fromkeys(cols))
                table_anom = outliers[cols].copy() if not outliers.empty else pd.DataFrame()
                for col in table_anom.select_dtypes(include='number').columns:
                    table_anom[col] = table_anom[col].map(lambda x: f"{x:.1f}")
                table_anom.columns = [truncate_col(c) for c in table_anom.columns]
                plot_df = ranked.copy()
                plot_df['anomaly_label'] = plot_df['anomaly'].map({-1: 'Outlier', 1: 'Normal'})
                if 'price' in variables and ('quality' in variables or 'technical' in variables):
                    y_var = 'quality' if 'quality' in variables else 'technical'
                    fig_anom = px.scatter(plot_df, x='price', y=y_var, color='anomaly_label',
                                         hover_data=['supplier'], title='Isolation Forest Outliers: Price vs. Quality/Technical', symbol='anomaly_label')
                else:
                    fig_anom = px.scatter(plot_df, x=variables[0], y=variables[1], color='anomaly_label',
                                         hover_data=['supplier'], title=f'Isolation Forest Outliers: {variables[0]} vs. {variables[1]}', symbol='anomaly_label')
                advanced_subsections.append({
                    'title': 'Anomaly Detection',
                    'summary': safe_text('Outlier detection using Isolation Forest. Outliers are flagged as suspicious offers.'),
                    'table': table_anom,
                    'chart': fig_anom,
                    'chart_caption': 'Isolation Forest Outliers'
                })
            # 2. Principal Component Analysis (PCA)
            if not filtered_data.empty and len(variables) >= 2:
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(filtered_data[variables])
                pca = PCA(n_components=2)
                components = pca.fit_transform(scaled_data)
                pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
                pca_df['supplier'] = filtered_data['supplier'].values
                for col in ['PC1', 'PC2']:
                    pca_df[col] = pca_df[col].map(lambda x: f"{x:.2f}")
                pca_df.columns = [truncate_col(c) for c in pca_df.columns]
                import altair as alt
                chart = alt.Chart(pca_df).mark_circle(size=60).encode(
                    x='PC1', y='PC2', color='supplier', tooltip=['supplier', 'PC1', 'PC2']
                ).properties(title='PCA: Supplier Distribution')
                advanced_subsections.append({
                    'title': 'Principal Component Analysis (PCA)',
                    'summary': safe_text('PCA reduces dimensionality for visual interpretation and highlights key contributing variables.'),
                    'table': pca_df,
                    'chart': chart,
                    'chart_caption': 'PCA: Supplier Distribution'
                })
            # 3. Price Feature Importance
            if not ranked.empty and 'price' in ranked.columns and len(variables) > 1:
                from sklearn.ensemble import RandomForestRegressor
                feature_vars = [v for v in variables if v != 'price']
                X = ranked[feature_vars]
                y = ranked['price']
                model = RandomForestRegressor(random_state=42)
                model.fit(X, y)
                importances = model.feature_importances_
                imp_df = pd.DataFrame({'Variable': feature_vars, 'Importance': importances}).sort_values('Importance', ascending=False).reset_index(drop=True)
                imp_df.index = imp_df.index + 1
                imp_df.index.name = 'Rank'
                imp_df['Importance'] = imp_df['Importance'].map(lambda x: f"{x:.2f}")
                imp_df.columns = [truncate_col(c) for c in imp_df.columns]
                fig_imp = px.bar(
                    imp_df,
                    x='Variable',
                    y='Importance',
                    color='Importance',
                    color_continuous_scale='Viridis',
                    title='Feature Importance for Price'
                )
                fig_imp.update_coloraxes(showscale=True, colorbar_title='Importance')
                advanced_subsections.append({
                    'title': 'Price Feature Importance',
                    'summary': safe_text('Random Forest model shows which variables are most important in explaining supplier price.'),
                    'table': imp_df,
                    'chart': fig_imp,
                    'chart_caption': 'Feature Importance for Price'
                })
            # 4. Procurement Risk Scoring
            if not ranked.empty:
                risk_df = calculate_risk_scores(ranked)
                risk_table_cols = ['supplier', 'total_risk', 'lowball_risk', 'drip_pricing_risk', 'drip_pricing_flag', 'signaling_risk', 'cover_bid_risk', 'decoy_bid_risk', 'bid_similarity_risk']
                table_risk = risk_df[risk_table_cols].copy()
                for col in table_risk.select_dtypes(include='number').columns:
                    table_risk[col] = table_risk[col].map(lambda x: f"{x:.2f}")
                # Use short aliases for risk columns for PDF readability
                risk_col_aliases = {
                    'supplier': 'Supplier',
                    'total_risk': 'Total',
                    'lowball_risk': 'Lowball',
                    'drip_pricing_risk': 'Drip',
                    'drip_pricing_flag': 'DripFlag',
                    'signaling_risk': 'Signal',
                    'cover_bid_risk': 'Cover',
                    'decoy_bid_risk': 'Decoy',
                    'bid_similarity_risk': 'Similar'
                }
                table_risk.columns = [risk_col_aliases.get(c, c) for c in table_risk.columns]
                fig_risk = px.bar(risk_df, y='supplier', x='total_risk', color='total_risk', orientation='h',
                             title='Overall Supplier Risk Score',
                             labels={'total_risk': 'Risk Score', 'supplier': 'Supplier'}, template='plotly_white')
                fig_risk.update_layout(yaxis={'categoryorder':'total ascending'})
                advanced_subsections.append({
                    'title': 'Procurement Risk Scoring',
                    'summary': safe_text('Each supplier is scored on risk behaviors: Lowballing, Drip Pricing, Market Signaling, Cover Bidding, Decoy Bidding, and Bid Similarity. The total risk is the average of these scores.'),
                    'table': table_risk,
                    'chart': fig_risk,
                    'chart_caption': 'Overall Supplier Risk Score'
                })
            # Add Advanced Analytics section
            sections.append({'title': 'Advanced Analytics: Machine Learning Insights', 'subsections': advanced_subsections})
            
            progress_bar.progress(60)
            status_text.text("Building Advanced Analytics sections... (60%)")
        if "Negotiation Strategy" in selected_sections:
            negotiation_subsections = []
            import plotly.express as px
            import pandas as pd
            from sklearn.ensemble import RandomForestRegressor
            # Use main uploaded/filtered data from session state
            df = st.session_state.get('filtered_data', pd.DataFrame())
            if not df.empty:
                col_map = {c.lower(): c for c in df.columns}
                price_col = col_map.get('price')
                quality_col = col_map.get('quality')
                delivery_col = col_map.get('delivery') or col_map.get('delivery_time') or col_map.get('delivery_time_days')
                if price_col and quality_col and delivery_col:
                    # Calculate scores and rank
                    df = df.copy()
                    df['score'] = df[quality_col] / (df[price_col] * (1 + 0.01 * df[delivery_col]))
                    top_k = min(3, len(df))
                    df = df.sort_values('score', ascending=False).reset_index(drop=True)
                    df['winner'] = False
                    df.loc[:top_k-1, 'winner'] = True
                    threshold_score = df.loc[top_k-1, 'score'] if top_k > 0 else None
                    # 1. Variable Impact Analysis (feature importance)
                    X_cols = list(dict.fromkeys([price_col, quality_col, delivery_col]))
                    X = df[X_cols]
                    y = df['score']
                    mask = ~(X.isna().any(axis=1) | y.isna())
                    X_clean = X[mask]
                    y_clean = y[mask]
                    if len(X_clean) > 0:
                        rf = RandomForestRegressor(random_state=42)
                        rf.fit(X_clean, y_clean)
                        importances = pd.Series(rf.feature_importances_, index=X_clean.columns).sort_values(ascending=False)
                        imp_df = importances.reset_index()
                        imp_df.columns = ['Variable', 'Importance']
                        imp_df['Importance'] = imp_df['Importance'].map(lambda x: f"{x:.2f}")
                        imp_df.columns = [truncate_col(c) for c in imp_df.columns]
                        fig_imp = px.bar(imp_df, x='Variable', y='Importance', color='Importance', color_continuous_scale='Viridis', title='Feature Importances (Negotiation)')
                        negotiation_subsections.append({
                            'title': 'Variable Impact Analysis',
                            'summary': safe_text('Feature importances (impact on selection score) using Random Forest.'),
                            'table': imp_df,
                            'chart': fig_imp,
                            'chart_caption': 'Feature Importances (Negotiation)'
                        })
                    # 2. Trade-Off Matrix (3D scatter)
                    if all(v in df.columns for v in [price_col, quality_col, delivery_col, 'winner', 'supplier']):
                        fig_tradeoff = px.scatter_3d(
                            df,
                            x=price_col,
                            y=quality_col,
                            z=delivery_col,
                            color='winner',
                            text='supplier',
                            labels={v: v.capitalize() for v in [price_col, quality_col, delivery_col]},
                            title=f"Supplier Trade-Off Matrix: {price_col.capitalize()} vs. {quality_col.capitalize()} vs. {delivery_col.capitalize()}"
                        )
                        negotiation_subsections.append({
                            'title': 'Trade-Off Matrix',
                            'summary': safe_text('3D scatter plot of price, quality, and delivery for all suppliers. Winners are highlighted.'),
                            'chart': fig_tradeoff,
                            'chart_caption': 'Supplier Trade-Off Matrix'
                        })
                    # 3. Ranked Supplier Table
                    table_cols = ['supplier', price_col, quality_col, delivery_col, 'score', 'winner']
                    table_cols = [col for col in table_cols if col in df.columns]
                    df_table = df[table_cols].copy()
                    for col in df_table.select_dtypes(include='number').columns:
                        df_table[col] = df_table[col].map(lambda x: f"{x:.2f}")
                    df_table.columns = [truncate_col(c) for c in df_table.columns]
                    negotiation_subsections.append({
                        'title': 'Ranked Supplier Table',
                        'summary': safe_text('Table of suppliers ranked by negotiation score. Winners are marked.'),
                        'table': df_table
                    })
                    # 4. Intelligent Negotiation Advice
                    advice = []
                    winners = df[df['winner']].copy()
                    if not winners.empty:
                        min_price = winners[price_col].min()
                        max_quality = winners[quality_col].max()
                        min_delivery = winners[delivery_col].min()
                        for idx, row in winners.iterrows():
                            suggestions = []
                            if row[price_col] > min_price:
                                diff = row[price_col] - min_price
                                suggestions.append(f"reduce price by €{diff:.2f} to match the lowest winner's price")
                            if row[quality_col] < max_quality:
                                diff = max_quality - row[quality_col]
                                suggestions.append(f"increase quality by {diff:.2f} to match the highest winner's quality")
                            if row[delivery_col] > min_delivery:
                                diff = row[delivery_col] - min_delivery
                                suggestions.append(f"reduce delivery time by {diff:.2f} days to match the fastest winner's delivery")
                            if suggestions:
                                msg = f"{row['supplier']}: " + ", and ".join(suggestions) + "."
                            else:
                                msg = f"{row['supplier']}: Already matches the best terms among winners."
                            advice.append(msg)
                    advice_text = '\n'.join([safe_text(t) for t in advice]) if advice else 'No negotiation advice available.'
                    negotiation_subsections.append({
                        'title': 'Intelligent Negotiation Advice',
                        'summary': advice_text
                    })
            sections.append({'title': 'Negotiation Strategy: Strategic Guidance', 'subsections': negotiation_subsections})
            
            progress_bar.progress(70)
            status_text.text("Building Negotiation Strategy sections... (70%)")
        if "What-If Analysis" in selected_sections:
            whatif_subsections = []
            import pandas as pd
            from sklearn.ensemble import RandomForestRegressor
            data = st.session_state.get('filtered_data', pd.DataFrame())
            if not data.empty and 'supplier' in data.columns and 'score' in data.columns:
                suppliers = data['supplier'].unique().tolist()
                # Use selected supplier from session state if available, else use top-ranked
                selected_supplier = st.session_state.get('whatif_selected_supplier', None)
                if not selected_supplier or selected_supplier not in suppliers:
                    selected_supplier = data.sort_values('score', ascending=False)['supplier'].iloc[0]
                orig_row = data[data['supplier'] == selected_supplier].iloc[0]
                # 1. Impactful Variables for Score
                feature_vars = [col for col in data.select_dtypes(include='number').columns if col not in ['score']]
                X = data[feature_vars]
                y = data['score']
                rf = RandomForestRegressor(random_state=42)
                rf.fit(X, y)
                importances = rf.feature_importances_
                top_vars = [feature_vars[i] for i in importances.argsort()[::-1][:5]]  # Top 5 impactful variables
                imp_df = pd.DataFrame({'Variable': top_vars, 'Importance': importances[importances.argsort()[::-1][:5]]})
                imp_df['Importance'] = imp_df['Importance'].map(lambda x: f"{x:.2f}")
                imp_df.columns = [truncate_col(c) for c in imp_df.columns]
                whatif_subsections.append({
                    'title': 'Most Impactful Variables for Score',
                    'summary': safe_text(f"Most impactful variables for score: {', '.join(top_vars)}"),
                    'table': imp_df
                })
                # 2. What-If Simulation Table
                sim_data = data.copy()
                new_vals = {}
                for var in top_vars:
                    min_val = float(data[var].min())
                    max_val = float(data[var].max())
                    default_val = float(orig_row[var])
                    new_val = default_val  # For PDF, use original value (or could simulate a +10% change)
                    new_vals[var] = new_val
                    sim_data.loc[sim_data['supplier'] == selected_supplier, var] = new_val
                # Recalculate score and rank
                if all(v in ['price', 'quality', 'delivery_time_days'] for v in top_vars):
                    sim_data['score'] = sim_data['quality'] / (sim_data['price'] * (1 + 0.01 * sim_data['delivery_time_days']))
                else:
                    for var in top_vars:
                        sim_data[var + '_norm'] = (sim_data[var] - sim_data[var].min()) / (sim_data[var].max() - sim_data[var].min() + 1e-9)
                    sim_data['score'] = sim_data[[v + '_norm' for v in top_vars]].sum(axis=1)
                sim_data = sim_data.sort_values('score', ascending=False).reset_index(drop=True)
                sim_data['rank'] = sim_data['score'].rank(ascending=False, method='min').astype(int)
                new_row = sim_data[sim_data['supplier'] == selected_supplier].iloc[0]
                sim_table = sim_data[['supplier'] + top_vars + ['score', 'rank']].copy()
                for col in sim_table.select_dtypes(include='number').columns:
                    sim_table[col] = sim_table[col].map(lambda x: f"{x:.2f}")
                sim_table.columns = [truncate_col(c) for c in sim_table.columns]
                whatif_subsections.append({
                    'title': 'What-If Simulation Table',
                    'summary': safe_text(f"What-if simulation for supplier: {selected_supplier}"),
                    'table': sim_table
                })
                # 3. New Score and Rank for Selected Supplier
                whatif_subsections.append({
                    'title': 'New Score and Rank',
                    'summary': safe_text(f"New Score: {new_row['score']:.4f} | New Rank: {new_row['rank']} for supplier {selected_supplier}")
                })
            sections.append({'title': 'What-If Analysis: Scenario Modeling', 'subsections': whatif_subsections})
            
            progress_bar.progress(80)
            status_text.text("Building What-If Analysis sections... (80%)")
        # Debug print for sections and subsections
        st.info(f"Sections to be included in PDF: {[s['title'] for s in sections]}")
        for s in sections:
            st.info(f"Section: {s['title']}, Subsections: {[sub['title'] for sub in s.get('subsections',[])]}")
        
        progress_bar.progress(90)
        status_text.text("Generating PDF report... (90%)")
        
        st.info("Generating PDF bytes...")
        pdf_bytes = generate_professional_pdf_report(sections, developer_name="Aryan Zabihi")
        
        progress_bar.progress(100)
        status_text.text("PDF report generated successfully! (100%)")
        
        st.success("PDF report generated!")
        st.sidebar.download_button("Download PDF Report", data=pdf_bytes, file_name="tender_report.pdf", mime="application/pdf")
        
        # Clear progress indicators after a short delay
        time.sleep(2)
        progress_bar.empty()
        status_text.empty()
    except Exception as e:
        # Clear progress indicators on error
        progress_bar.empty()
        status_text.empty()
        st.error(f"Failed to generate PDF report: {e}")



