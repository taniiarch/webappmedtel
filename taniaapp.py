import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import google.generativeai as genai
import json
import io

# --- Gemini API Configuration ---
# Attempts to load API key from Streamlit secrets.toml.
# If not found, offers a text input in the sidebar for manual entry (for demo purposes).
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except KeyError:
    st.warning("Please add your GEMINI_API_KEY to secrets.toml (in the .streamlit folder) to enable insights. Alternatively, enter it in the sidebar.")
    with st.sidebar:
        st.session_state["GEMINI_API_KEY_ENV"] = st.text_input("Enter your Gemini API Key (Optional for Insights):", type="password")
        if st.session_state["GEMINI_API_KEY_ENV"]:
            genai.configure(api_key=st.session_state["GEMINI_API_KEY_ENV"])


# --- Data Cleaning and Chart/Insight Generation Functions ---

def generate_mock_data():
    """Generates a mock DataFrame for demonstration purposes."""
    data = {
        'Date': pd.to_datetime(['2023-01-01', '2023-01-05', '2023-01-10', '2023-01-15', '2023-01-20', '2023-01-25', '2023-01-30']),
        'Platform': ['Twitter', 'Facebook', 'Instagram', 'Twitter', 'LinkedIn', 'TikTok', 'Twitter'],
        'Sentiment': ['Positive', 'Negative', 'Neutral', 'Positive', 'Positive', 'Negative', 'Neutral'],
        'Location': ['New York', 'London', 'Paris', 'New York', 'Tokyo', 'Sydney', 'London'],
        'Engagements': [120, 80, 50, 150, None, 100, 70], # Simulate missing data
        'Media Type': ['Text', 'Image', 'Video', 'Text', 'Link', 'Video', 'Image']
    }
    df = pd.DataFrame(data)
    return df

def clean_data(df):
    """
    Performs data cleaning on the DataFrame:
    - Converts 'Date' to datetime.
    - Fills missing 'Engagements' with 0.
    - Normalizes column names (all lowercase, spaces replaced by underscores).
    """
    df['Date'] = pd.to_datetime(df['Date'])
    df['Engagements'] = df['Engagements'].fillna(0).astype(int)
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def generate_charts_and_insights(df):
    """
    Generates Plotly charts and insights using the Gemini API (if key is available).
    """
    dashboard_data = {}

    # --- 1. Pie chart: Sentiment Breakdown ---
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig_sentiment = px.pie(sentiment_counts, values='Count', names='Sentiment',
                           title='Sentiment Breakdown', hole=.4,
                           color_discrete_sequence=['#F472B6', '#EF4444', '#FCD34D']) # Pink, Red, Yellow
    fig_sentiment.update_layout(height=350, margin=dict(t=50, b=50, l=50, r=50),
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(family='Inter', color='#374151'), showlegend=True,
                                legend=dict(orientation='h', yanchor='bottom', y=-0.2))
    dashboard_data['sentiment_breakdown'] = {'title': 'Sentiment Breakdown', 'chart': fig_sentiment, 'insights': []}

    # --- 2. Line chart: Engagement Trend over time ---
    # Group by week for trend analysis
    engagement_trend = df.groupby(df['date'].dt.to_period('W'))['engagements'].sum().reset_index()
    engagement_trend['date'] = engagement_trend['date'].dt.to_timestamp() # Convert Period back to Timestamp for Plotly
    fig_engagement_trend = px.line(engagement_trend, x='date', y='engagements',
                                   title='Engagement Trend Over Time',
                                   line_shape='linear', markers=True,
                                   color_discrete_sequence=['#EC4899']) # Darker pink
    fig_engagement_trend.update_layout(height=350, margin=dict(t=50, b=50, l=50, r=50),
                                      xaxis_title='Date', yaxis_title='Engagements',
                                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                      font=dict(family='Inter', color='#374151'))
    dashboard_data['engagement_trend'] = {'title': 'Engagement Trend over Time', 'chart': fig_engagement_trend, 'insights': []}

    # --- 3. Bar chart: Platform Engagements ---
    platform_engagements = df.groupby('platform')['engagements'].sum().reset_index()
    fig_platform_engagements = px.bar(platform_engagements, x='platform', y='engagements',
                                      title='Platform Engagements',
                                      color='platform',
                                      color_discrete_map={
                                          'Twitter': '#DB2777',
                                          'Facebook': '#F0ABFC',
                                          'Instagram': '#FB7185',
                                          'LinkedIn': '#F87171',
                                          'TikTok': '#C084FC'
                                      }) # Pink color scheme
    fig_platform_engagements.update_layout(height=350, margin=dict(t=50, b=50, l=50, r=50),
                                          xaxis_title='Platform', yaxis_title='Total Engagements',
                                          paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                          font=dict(family='Inter', color='#374151'))
    dashboard_data['platform_engagements'] = {'title': 'Platform Engagements', 'chart': fig_platform_engagements, 'insights': []}

    # --- 4. Pie chart: Media Type Mix ---
    media_type_counts = df['media_type'].value_counts().reset_index()
    media_type_counts.columns = ['Media Type', 'Count']
    fig_media_type = px.pie(media_type_counts, values='Count', names='Media Type',
                           title='Media Type Mix', hole=.4,
                           color_discrete_sequence=['#F9A8D4', '#FBCFE8', '#FCE7F6', '#FDA4AF']) # Lighter pink shades
    fig_media_type.update_layout(height=350, margin=dict(t=50, b=50, l=50, r=50),
                                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                font=dict(family='Inter', color='#374151'), showlegend=True,
                                legend=dict(orientation='h', yanchor='bottom', y=-0.2))
    dashboard_data['media_type_mix'] = {'title': 'Media Type Mix', 'chart': fig_media_type, 'insights': []}

    # --- 5. Bar chart: Top 5 Locations ---
    location_engagements = df.groupby('location')['engagements'].sum().nlargest(5).reset_index()
    fig_top_locations = px.bar(location_engagements, x='location', y='engagements',
                               title='Top 5 Locations by Engagements',
                               color_discrete_sequence=['#FBCFE8']) # Pink color
    fig_top_locations.update_layout(height=350, margin=dict(t=50, b=50, l=50, r=50),
                                   xaxis_title='Location', yaxis_title='Total Engagements',
                                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                   font=dict(family='Inter', color='#374151'))
    dashboard_data['top_locations'] = {'title': 'Top 5 Locations by Engagements', 'chart': fig_top_locations, 'insights': []}

    # --- Generate insights using Gemini API (if key is available) ---
    api_key_configured = False
    try:
        # Check if genai is configured (this might raise an error if not configured)
        _ = genai.get_model('gemini-2.0-flash')
        api_key_configured = True
    except Exception:
        api_key_configured = False

    if api_key_configured:
        model = genai.GenerativeModel('gemini-2.0-flash')
        for key, value in dashboard_data.items():
            chart_data_desc = ""
            chart_title = value['title']

            # Prepare data description for the LLM prompt
            if key == 'sentiment_breakdown':
                chart_data_desc = f"Sentiment counts: {sentiment_counts.to_dict('records')}"
            elif key == 'engagement_trend':
                chart_data_desc = f"Engagement trend data: {engagement_trend.to_dict('records')}"
            elif key == 'platform_engagements':
                chart_data_desc = f"Platform engagements: {platform_engagements.to_dict('records')}"
            elif key == 'media_type_mix':
                chart_data_desc = f"Media type mix: {media_type_counts.to_dict('records')}"
            elif key == 'top_locations':
                chart_data_desc = f"Top locations by engagements: {location_engagements.to_dict('records')}"

            try:
                prompt = f"Based on the following data for \"{chart_title}\", provide the top 3 concise insights. Be specific and actionable:\n\n{chart_data_desc}"
                response = model.generate_content(
                    [{"role": "user", "parts": [{"text": prompt}]}],
                    generation_config={
                        "response_mime_type": "application/json",
                        "response_schema": {"type": "ARRAY", "items": {"type": "STRING"}}
                    }
                )
                insights = json.loads(response.text)
                dashboard_data[key]['insights'] = insights
            except Exception as e:
                st.error(f"Failed to generate insights for '{chart_title}': {e}. Ensure you have an internet connection and sufficient API quota.")
                dashboard_data[key]['insights'] = ["Failed to generate insights."]
    else:
        st.info("Gemini API key not found or invalid. Insights will not be generated.")

    return dashboard_data

# --- Streamlit Application ---

# Configure page layout
st.set_page_config(layout="wide", page_title="Interactive Media Intelligence Dashboard")

# --- Custom CSS for pink theme ---
st.markdown("""
    <style>
    /* Set the app background to a pink gradient */
    .stApp {
        background: linear-gradient(to bottom right, #FCE7F6, #FCE7F6, #FCE7F6); /* Lighter pink gradient */
    }
    /* Style for the main content container */
    .main {
        background-color: white;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 2rem;
    }
    /* Style for titles and subtitles */
    h1, h2, h3, h4, h5, h6 {
        color: #DB2777; /* Rose 700 */
        font-family: 'Inter', sans-serif;
    }
    /* Style for the file uploader label */
    .css-1d3z3hw > div > label { /* Specific target for Streamlit uploader label */
        color: white !important;
        background-color: #EC4899; /* Pink 600 */
        border-radius: 0.5rem;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        cursor: pointer;
    }
    .css-1d3z3hw > div > label:hover {
        background-color: #DB2777; /* Pink 700 */
    }
    /* Style for general Streamlit buttons */
    .stButton>button {
        background-color: #8B5CF6; /* Purple 600 */
        color: white;
        padding: 0.75rem 2rem;
        border-radius: 0.5rem;
        font-weight: bold;
        font-size: 1.125rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #7C3AED; /* Purple 700 */
        box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
    }
    /* Style for alerts (e.g., API key warning) */
    .stAlert {
        border-radius: 0.5rem;
    }
    /* Style for info text */
    .st-emotion-cache-16txt4v p { /* This might be a specific class for info text in Streamlit */
        color: #B91C1C; /* Red for error messages */
    }
    </style>
    """, unsafe_allow_html=True)

# --- App Header ---
st.title("Interactive Media Intelligence Dashboard")
st.subheader("by Tania Putri Rachmadani")
st.markdown("Upload your CSV to visualize and gain insights from your media data.")

st.markdown("---")

# --- CSV Upload Section ---
st.header("1. Upload Your CSV File")
st.markdown("Required columns: **Date, Platform, Sentiment, Location, Engagements, Media Type**")

uploaded_file = st.file_uploader("Choose CSV File", type="csv")

# Initialize session state to store dashboard output
if 'dashboard_output' not in st.session_state:
    st.session_state['dashboard_output'] = None
if 'cleaned_df' not in st.session_state:
    st.session_state['cleaned_df'] = None

if uploaded_file:
    st.success(f"Selected file: {uploaded_file.name}")
    try:
        # Read the CSV file
        df_uploaded = pd.read_csv(uploaded_file)

        # Basic validation for required columns
        required_cols = ['Date', 'Platform', 'Sentiment', 'Location', 'Engagements', 'Media Type']
        if not all(col in df_uploaded.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df_uploaded.columns]
            st.error(f"The uploaded CSV is missing required columns: {', '.join(missing_cols)}. Please check your file format.")
            st.session_state['dashboard_output'] = None # Reset dashboard if error
            st.stop() # Stop execution if columns are missing

        if st.button("Process Data"):
            with st.spinner('Processing Data and Generating Charts & Insights...'):
                st.session_state['cleaned_df'] = clean_data(df_uploaded.copy())
                st.session_state['dashboard_output'] = generate_charts_and_insights(st.session_state['cleaned_df'])
            st.success("Data successfully processed!")

    except Exception as e:
        st.error(f"Failed to process CSV: {e}. Please ensure the file format is correct.")
        st.session_state['dashboard_output'] = None
        st.session_state['cleaned_df'] = None # Reset cleaned data as well

else:
    # If no file is uploaded, offer to show an example dashboard
    st.info("Alternatively, you can view an example dashboard without uploading a CSV.")
    if st.button("Show Example Dashboard"):
        with st.spinner('Loading Example Dashboard...'):
            mock_df = generate_mock_data()
            st.session_state['cleaned_df'] = clean_data(mock_df.copy())
            st.session_state['dashboard_output'] = generate_charts_and_insights(st.session_state['cleaned_df'])
        st.success("Example Dashboard loaded!")

st.markdown("---")

# --- Interactive Dashboard Section ---
if st.session_state.get('dashboard_output'):
    st.header("Interactive Dashboard")
    for key, chart_info in st.session_state['dashboard_output'].items():
        st.subheader(f"{chart_info['title']}")
        st.plotly_chart(chart_info['chart'], use_container_width=True)
        st.markdown("#### Top 3 Insights:")
        # Display insights
        if chart_info['insights']:
            for i, insight in enumerate(chart_info['insights']):
                st.write(f"- {insight}")
        else:
            st.write("No insights generated.")
        st.markdown("---") # Separator between charts

    # --- Download Options ---
    st.markdown("### Download Report")
    st.info("Direct PDF download of the entire interactive dashboard (like in React) is more complex in Streamlit. You can download each chart as interactive HTML or the cleaned data as CSV.")

    # Download cleaned data as CSV
    if st.session_state['cleaned_df'] is not None:
        csv_data = st.session_state['cleaned_df'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Cleaned Data (CSV)",
            data=csv_data,
            file_name="cleaned_media_intelligence_data.csv",
            mime="text/csv",
        )

    # Download each chart as interactive HTML
    for key, chart_info in st.session_state['dashboard_output'].items():
        chart_html = chart_info['chart'].to_html(full_html=False, include_plotlyjs='cdn')
        st.download_button(
            label=f"Download '{chart_info['title']}' (Interactive HTML)",
            data=chart_html,
            file_name=f"{key}_chart.html",
            mime="text/html",
        )

# --- App Footer ---
st.markdown("---")
st.caption(f"© {pd.Timestamp.now().year} Media Intelligence Dashboard. All rights reserved.")
