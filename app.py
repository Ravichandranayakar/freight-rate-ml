import gradio as gr
import pandas as pd
import numpy as np
import joblib

# Load model 
try:
    model = joblib.load("freight_rate_model.joblib")
    model_status = " Model loaded successfully"
except Exception as e:
    model = None
    model_status = f" Warning: Could not load model ({e})"

# Inference Logic & Feature Engineering
DOW_MAP = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6
}

def predict_freight_rate(
    distance,
    cargo_weight,
    equipment_type,
    origin_lat,
    origin_lng,
    dest_lat,
    dest_lng,
    market_index,
    month,
    day_of_week
):
    if model is None:
        return (
            "Model file 'freight_rate_model.joblib' not found. Please ensure the model artifact is saved.",
            "$0.00 / mi",
            "Model Offline"
        )

    # 1. Feature Engineering
    dow_num = DOW_MAP.get(day_of_week, 2)
    is_short_haul = 1 if distance < 150 else 0
    est_cost = distance * (cargo_weight / 1000.0) * 0.05
    market_pressure = market_index * 1.15

    # 2. Building inference dataframe with exact column schema 
    input_df = pd.DataFrame([{
        'distance': float(distance),
        'cargo_weight': float(cargo_weight),
        'equipment_type': str(equipment_type),
        'origin_lat': float(origin_lat),
        'origin_lng': float(origin_lng),
        'dest_lat': float(dest_lat),
        'dest_lng': float(dest_lng),
        'market_index': float(market_index),
        'month': int(month),
        'day_of_week': int(dow_num),
        'is_short_haul': int(is_short_haul),
        'est_cost': float(est_cost),
        'market_pressure': float(market_pressure)
    }])

    # 3. Predict
    try:
        pred_rate = float(model.predict(input_df)[0])
        pred_rate = max(0.0, pred_rate)
        per_mile = pred_rate / max(distance, 1.0)

        rate_display = f"${pred_rate:,.2f}"
        rpm_display = f"${per_mile:.2f} / mile"
        summary = (
            f"**Lane:** ({origin_lat:.2f}, {origin_lng:.2f}) ➔ ({dest_lat:.2f}, {dest_lng:.2f})\n\n"
            f"- **Distance:** {distance:,} miles\n"
            f"- **Equipment:** {equipment_type}\n"
            f"- **Cargo Weight:** {cargo_weight:,} lbs\n"
            f"- **Estimated Base Cost Proxy:** ${est_cost:,.2f}\n"
            f"- **Model Validation Accuracy:** MAE $114.36 (Tuned LightGBM L1)"
        )

        return rate_display, rpm_display, summary

    except Exception as err:
        return f"Error during prediction: {err}", "N/A", "Error"

# Modern Gradio UI

theme = gr.themes.Soft(
    primary_hue="teal",
    neutral_hue="slate"
)

with gr.Blocks(theme=theme, title="US Freight Rate Prediction Engine") as demo:
    gr.Markdown(
        """
        #  US Freight Rate Prediction Engine
        Predict commercial spot market shipping rates across US logistics lanes using an optimized **LightGBM** regressor (MAE: $114.36).
        """
    )
    
    with gr.Row():
        # Column 1: Shipment Characteristics
        with gr.Column(scale=1):
            gr.Markdown("###  Shipment Characteristics")
            distance = gr.Slider(
                label="Trip Distance (miles)", minimum=10, maximum=3000, value=715, step=5
            )
            cargo_weight = gr.Number(
                label="Cargo Weight (lbs)", value=34500, minimum=500, maximum=50000, step=500
            )
            equipment_type = gr.Dropdown(
                label="Equipment Type", choices=["Dry Van", "Reefer", "Flatbed"], value="Dry Van"
            )

            gr.Markdown("###  Lane Coordinates")
            with gr.Row():
                origin_lat = gr.Number(label="Origin Lat", value=33.7490)
                origin_lng = gr.Number(label="Origin Lng", value=-84.3880)
            with gr.Row():
                dest_lat = gr.Number(label="Dest Lat", value=41.8781)
                dest_lng = gr.Number(label="Dest Lng", value=-87.6298)

        # Column 2: Market Signals & Results
        with gr.Column(scale=1):
            gr.Markdown("###  Market & Calendar Signals")
            market_index = gr.Slider(
                label="Market Pressure Index", minimum=0.0, maximum=5.0, value=1.2, step=0.05
            )
            with gr.Row():
                month = gr.Dropdown(
                    label="Shipment Month", choices=list(range(1, 13)), value=10
                )
                day_of_week = gr.Dropdown(
                    label="Day of Week",
                    choices=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                    value="Wednesday"
                )

            predict_btn = gr.Button(" Predict Freight Rate", variant="primary", size="lg")

            gr.Markdown("###  Prediction Output")
            with gr.Row():
                out_rate = gr.Textbox(label="Estimated Posted Rate (USD)", elem_id="rate_box")
                out_rpm = gr.Textbox(label="Rate Per Mile")
            
            out_summary = gr.Markdown(label="Shipment Summary")

    # 1-Click Example Presets for recruiters/viewers
    gr.Markdown("###  Click an Example to Pre-fill and Test")
    gr.Examples(
        examples=[
            [715, 34500, "Dry Van", 33.7490, -84.3880, 41.8781, -87.6298, 1.20, 10, "Wednesday"],   # Atlanta -> Chicago
            [240, 42000, "Flatbed", 29.7604, -95.3698, 32.7767, -96.7970, 0.95, 8, "Tuesday"],      # Houston -> Dallas
            [1130, 28000, "Reefer", 34.0522, -118.2437, 47.6062, -122.3321, 1.85, 11, "Friday"],   # LA -> Seattle
            [95, 15000, "Dry Van", 40.7128, -74.0060, 39.9526, -75.1652, 1.10, 5, "Monday"],       # NYC -> Philly (Short haul)
        ],
        inputs=[
            distance, cargo_weight, equipment_type,
            origin_lat, origin_lng, dest_lat, dest_lng,
            market_index, month, day_of_week
        ],
        outputs=[out_rate, out_rpm, out_summary],
        fn=predict_freight_rate,
        cache_examples=False
    )

    predict_btn.click(
        fn=predict_freight_rate,
        inputs=[
            distance, cargo_weight, equipment_type,
            origin_lat, origin_lng, dest_lat, dest_lng,
            market_index, month, day_of_week
        ],
        outputs=[out_rate, out_rpm, out_summary]
    )

if __name__ == "__main__":
    demo.launch()
