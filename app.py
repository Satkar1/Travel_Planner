import os
import streamlit as st
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import pandas as pd
import io
import plotly.graph_objects as go

# ✅ Load API key from Streamlit Secrets (for Streamlit Cloud Deployment)
gemini_api_key = st.secrets["GEMINI_API_KEY"]

if not gemini_api_key:
    st.error("GEMINI_API_KEY is missing! Add it in Streamlit Secrets.")
    st.stop()

# ✅ Configure Google Gemini AI
genai.configure(api_key=gemini_api_key)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", google_api_key=gemini_api_key)

# ✅ Define Prompt for Travel Recommendations
prompt_template = PromptTemplate(
    input_variables=["source", "destination"],
    template="""
    You are a travel planning assistant. Provide travel options from {source} to {destination}. 
    Present the information in a structured table format with the following columns:

    | Travel Type | Price (Estimated) | Time (Estimated) | Description | Comfort Level (1-5, 5 being highest) | Directness (Direct/Indirect) |
    |-------------------|-------------------|-------------------|-------------|------------------------------------|-----------------------------|
    | Cab/Taxi          |                   |                   |             |                                    |                             |
    | Train             |                   |                   |             |                                    |                             |
    | Bus               |                   |                   |             |                                    |                             |
    | Flight            |                   |                   |             |                                    |                             |
    | Ola/Uber          |                   |                   |             |                                    |                             |

    Fill in the table with estimated prices, travel times, descriptions, comfort levels (1-5), and directness.
    If a mode of transport is unavailable, indicate it in the table.
    """
)

# ✅ Create LangChain LLMChain
travel_chain = LLMChain(llm=llm, prompt=prompt_template)

# ✅ Function to Fetch Travel Recommendations
def get_travel_recommendations(source, destination):
    try:
        response = travel_chain.run({"source": source, "destination": destination})
        return response if isinstance(response, str) else response["text"]
    except Exception as e:
        return f"An error occurred: {e}"

# ✅ Streamlit UI
st.title("🚀 AI-Powered Travel Planner")
st.write("Find the best travel options using AI!")

source = st.text_input("📍 Enter Source City:")
destination = st.text_input("📍 Enter Destination City:")

if st.button("🔍 Get Travel Options"):
    if source and destination:
        st.write(f"Generating travel options from **{source}** to **{destination}**...")
        recommendations = get_travel_recommendations(source, destination)
        st.write("### Travel Recommendations:")
        st.write(recommendations)

        # ✅ Process Data into a Table for Visualization
        try:
            table_data = recommendations.strip().split('\n')[2:-1]
            rows = [row.strip().split('|')[1:-1] for row in table_data]
            df = pd.DataFrame(rows, columns=["Travel Type", "Price (Estimated)", "Time (Estimated)", "Description", "Comfort Level", "Directness"])

            # Convert Price and Time to numeric values
            df["Price (Estimated)"] = pd.to_numeric(df["Price (Estimated)"].str.replace(r'[^\d\.]+', '', regex=True), errors='coerce')
            df["Time (Estimated)"] = pd.to_numeric(df["Time (Estimated)"].str.replace(r'[^\d\.]+', '', regex=True), errors='coerce')

            # ✅ Create Price Comparison Chart
            fig_price = go.Figure([go.Bar(x=df["Travel Type"], y=df["Price (Estimated)"])])
            fig_price.update_layout(title="💰 Price Comparison", xaxis_title="Travel Type", yaxis_title="Price (₹)")
            st.plotly_chart(fig_price)

            # ✅ Create Time Comparison Chart
            fig_time = go.Figure([go.Bar(x=df["Travel Type"], y=df["Time (Estimated)"])])
            fig_time.update_layout(title="⏳ Time Comparison", xaxis_title="Travel Type", yaxis_title="Time (Hours)")
            st.plotly_chart(fig_time)

            # ✅ CSV Download Button
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            st.download_button(label="📥 Download Travel Data as CSV", data=csv_buffer.getvalue(), file_name="travel_data.csv", mime="text/csv")

        except Exception as e:
            st.error(f"⚠️ Error processing data or creating charts: {e}")

    else:
        st.error("❌ Please enter both source and destination cities.")

# ✅ Sidebar Information
st.sidebar.header("ℹ️ About This App")
st.sidebar.write("""
This application utilizes **LangChain + Google Gemini AI** to provide travel recommendations.
Just enter your source and destination, and AI will generate a list of travel options! 🚀
""")

st.sidebar.subheader("🔧 Tech Stack")
st.sidebar.write("""
- **Python**
- **Streamlit**
- **LangChain**
- **Google Gemini AI**
- **pandas**
- **plotly**
""")

st.sidebar.subheader("🛠️ Deployment Instructions")
st.sidebar.write("""
1️⃣ Create a **GitHub repository** and upload `app.py` and `requirements.txt`.  
2️⃣ Go to **Streamlit Cloud** → **New App**.  
3️⃣ Select your GitHub repository.  
4️⃣ **Set API Key in Streamlit Secrets**:
   - Open **Settings > Secrets**.
   - Add:
     ```
     GEMINI_API_KEY = your-api-key-here
     ```
5️⃣ Click **Deploy** and your app will be live! 🎉
""")
