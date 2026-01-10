import pandas as pd
import gradio as gr
import requests

# ===============================
# FASTAPI BACKEND URL
# ===============================
API_URL = "http://127.0.0.1:8000/predict"


# ===============================
# PREDICT VIA API
# ===============================
def predict(
    Model,
    Year_of_Manufacture,
    Number_of_Engines,
    Engine_Type,
    Capacity,
    Fuel_Consumption_L_per_hour,
    Hourly_Maintenance_Cost_USD,
    Age,
    Sales_Region,
    Price_USD
):
    payload = {
        "Model": Model,
        "Year_of_Manufacture": int(Year_of_Manufacture),
        "Number_of_Engines": int(Number_of_Engines),
        "Engine_Type": Engine_Type,
        "Capacity": int(Capacity),
        "Fuel_Consumption_L_per_hour": float(Fuel_Consumption_L_per_hour),
        "Hourly_Maintenance_Cost_USD": float(Hourly_Maintenance_Cost_USD),
        "Age": int(Age),
        "Sales_Region": Sales_Region,
        "Price_USD": float(Price_USD)
    }

    response = requests.post(API_URL, json=payload, timeout=5)
    response.raise_for_status()
    result = response.json()

    return (
        result["prediction"],
        result["range_min"],
        result["range_max"]
    )


# ===============================
# CUSTOM CSS
# ===============================
CUSTOM_CSS = """
body {
    background: radial-gradient(circle at top, #0f172a, #020617);
}

.gr-box {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(12px);
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.08);
}

h1 {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
}

.gradient-text {
    background: linear-gradient(to right, #38bdf8, #22c55e);
    -webkit-background-clip: text;
    color: transparent;
}

.emoji {
    color: initial !important;
    margin-right: 8px;
}

button {
    background: linear-gradient(135deg, #22c55e, #38bdf8) !important;
    color: black !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    border-radius: 14px !important;
    height: 60px !important;
}

/* remove label background */
label span,
.gr-label span {
    background: transparent !important;
    box-shadow: none !important;
    padding: 0 !important;
}
"""


# ===============================
# UI
# ===============================
with gr.Blocks(css=CUSTOM_CSS, theme=gr.themes.Soft()) as demo:

    gr.Markdown("""
    <h1>
        <span class="emoji">✈️</span>
        <span class="gradient-text">Airplane Range Predictor</span>
    </h1>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 🛩 Aircraft Info")
            Model = gr.Textbox(label="Model")
            Engine_Type = gr.Textbox(label="Engine Type")
            Sales_Region = gr.Textbox(label="Sales Region")

        with gr.Column(scale=2):
            gr.Markdown("### 🏭 Manufacturing")
            Year_of_Manufacture = gr.Number(label="Year of Manufacture")
            Age = gr.Number(label="Age")
            Number_of_Engines = gr.Number(label="Number of Engines")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📐 Performance")
            Capacity = gr.Number(label="Capacity")
            Fuel_Consumption_L_per_hour = gr.Number(label="Fuel Consumption (L/hour)")

        with gr.Column(scale=2):
            gr.Markdown("### 💰 Costs & Price")
            Hourly_Maintenance_Cost_USD = gr.Number(label="Hourly Maintenance Cost (USD)")
            Price_USD = gr.Number(label="Price (USD)")

    predict_btn = gr.Button("🚀 Predict Range")

    with gr.Row():
        prediction = gr.Number(label="Predicted Value")
        range_min = gr.Number(label="Range Min")
        range_max = gr.Number(label="Range Max")

    predict_btn.click(
        predict,
        inputs=[
            Model,
            Year_of_Manufacture,
            Number_of_Engines,
            Engine_Type,
            Capacity,
            Fuel_Consumption_L_per_hour,
            Hourly_Maintenance_Cost_USD,
            Age,
            Sales_Region,
            Price_USD
        ],
        outputs=[
            prediction,
            range_min,
            range_max
        ]
    )


if __name__ == "__main__":
    demo.launch()