import gradio as gr
import pandas as pd
import numpy as np
import joblib
import math
from datetime import datetime

try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(func):
            return func

# ---------------------------------------------------------
# Load Model
# ---------------------------------------------------------
try:
    model = joblib.load("freight_rate_model.joblib")
except Exception:
    model = None

# ---------------------------------------------------------
# City Coordinates Database (Top US Freight Hubs)
# ---------------------------------------------------------
US_CITIES = {
    "Atlanta, GA": (33.7490, -84.3880),
    "Chicago, IL": (41.8781, -87.6298),
    "Dallas, TX": (32.7767, -96.7970),
    "Houston, TX": (29.7604, -95.3698),
    "Los Angeles, CA": (34.0522, -118.2437),
    "New York, NY": (40.7128, -74.0060),
    "Philadelphia, PA": (39.9526, -75.1652),
    "Seattle, WA": (47.6062, -122.3321),
    "Miami, FL": (25.7617, -80.1918),
    "Denver, CO": (39.7392, -104.9903),
    "Kansas City, MO": (39.0997, -94.5786),
    "Memphis, TN": (35.1495, -90.0490),
    "Phoenix, AZ": (33.4484, -112.0740),
    "Indianapolis, IN": (39.7684, -86.1581)
}

MARKET_PRESETS = {
    "Normal Demand (Standard Rates)": 1.0,
    "Tight Capacity / Peak Season (Surge)": 1.35,
    "Soft Market / Low Demand (Discounted)": 0.85
}

def haversine_miles(lat1, lon1, lat2, lon2):
    r = 3958.8
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    return max(40, round(r * c * 1.18))

def update_lane_distance(origin, dest):
    c1 = US_CITIES.get(origin, (33.7490, -84.3880))
    c2 = US_CITIES.get(dest, (41.8781, -87.6298))
    return haversine_miles(c1[0], c1[1], c2[0], c2[1])

# ---------------------------------------------------------
# Prediction Function 
# ---------------------------------------------------------
@spaces.GPU
def predict_freight(origin_city, dest_city, equipment, weight, market_desc, custom_dist):
    if model is None:
        return "Model not loaded. Ensure freight_rate_model.joblib is present.", "N/A", "Offline"

    # Coordinates
    orig_lat, orig_lng = US_CITIES.get(origin_city, (33.7490, -84.3880))
    dest_lat, dest_lng = US_CITIES.get(dest_city, (41.8781, -87.6298))

    distance = custom_dist if custom_dist > 0 else haversine_miles(orig_lat, orig_lng, dest_lat, dest_lng)
    market_index = MARKET_PRESETS.get(market_desc, 1.0)
    quote_signal = 2.06

    # Feature Engineering formulas 
    est_cost = distance * quote_signal
    market_presssure = market_index * quote_signal
    is_short_haul = 1 if distance < 250 else 0

    # Dummy variables (To match  pd.get_dummies drop_first=True)
    eq_flatbed = 1 if equipment == "Flatbed" else 0
    eq_reefer = 1 if equipment == "Reefer" else 0

    now = datetime.now()
    month = now.month
    day_of_week = now.weekday()

    # Exact 15 features matching model.booster_.feature_name() in order
    input_df = pd.DataFrame([{
        'pickup_lat': float(orig_lat),
        'pickup_lon': float(orig_lng),
        'delivery_lat': float(dest_lat),
        'delivery_lon': float(dest_lng),
        'distance': float(distance),
        'weight': float(weight),
        'market_index': float(market_index),
        'quote_signal': float(quote_signal),
        'month': int(month),
        'day_of_week': int(day_of_week),
        'est_cost': float(est_cost),
        'market_presssure': float(market_presssure),
        'is_short_haul': int(is_short_haul),
        'equipment_Flatbed': int(eq_flatbed),
        'equipment_Reefer': int(eq_reefer)
    }])

    pred_rate = float(model.predict(input_df)[0])
    pred_rate = max(0.0, pred_rate)
    rpm = pred_rate / max(distance, 1.0)

    rate_text = f"${pred_rate:,.2f}"
    rpm_text = f"${rpm:.2f} / mi"

    details = f"""
    ### Lane Breakdown:
    * **Route:** {origin_city} -> {dest_city} ({distance:,} miles)
    * **Equipment:** {equipment} | **Weight:** {weight:,} lbs
    * **Market Condition:** {market_desc}
    * **Estimated Base Cost:** ${est_cost:,.2f}
    * **Model Architecture:** LightGBM L1 Regression (Holdout MAE: $114.36)
    """

    return rate_text, rpm_text, details

# ---------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------
city_list = list(US_CITIES.keys())

with gr.Blocks(title="US Freight Rate Prediction Engine") as demo:
    gr.Markdown(
        """
        # US Freight Rate Estimator
        Predict commercial truckload spot rates across major US shipping lanes using an optimized LightGBM regressor.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Select Origin & Destination")
            with gr.Row():
                origin_city = gr.Dropdown(label="Origin Hub", choices=city_list, value="Atlanta, GA")
                dest_city = gr.Dropdown(label="Destination Hub", choices=city_list, value="Chicago, IL")

            gr.Markdown("### Load Specifications")
            equipment = gr.Radio(label="Trailer Equipment", choices=["Dry Van", "Reefer", "Flatbed"], value="Dry Van")
            weight = gr.Slider(label="Cargo Weight (lbs)", minimum=1000, maximum=48000, value=34000, step=1000)
            market_desc = gr.Dropdown(
                label="Market Capacity Condition",
                choices=list(MARKET_PRESETS.keys()),
                value="Normal Demand (Standard Rates)"
            )

            distance = gr.Slider(label="Estimated Highway Distance (miles)", minimum=40, maximum=3000, value=715, step=5)

            origin_city.change(fn=update_lane_distance, inputs=[origin_city, dest_city], outputs=distance)
            dest_city.change(fn=update_lane_distance, inputs=[origin_city, dest_city], outputs=distance)

            predict_btn = gr.Button("Calculate Rate", variant="primary", size="lg")

        with gr.Column(scale=1):
            gr.Markdown("### Rate Prediction")
            with gr.Row():
                out_rate = gr.Textbox(label="Estimated Posted Rate", elem_id="rate-box")
                out_rpm = gr.Textbox(label="Rate Per Mile (RPM)")

            out_details = gr.Markdown()

    gr.Markdown("### Quick Examples (Click to Test)")
    gr.Examples(
        examples=[
            ["Atlanta, GA", "Chicago, IL", "Dry Van", 34000, "Normal Demand (Standard Rates)", 715],
            ["Houston, TX", "Dallas, TX", "Flatbed", 42000, "Normal Demand (Standard Rates)", 240],
            ["Los Angeles, CA", "Seattle, WA", "Reefer", 28000, "Tight Capacity / Peak Season (Surge)", 1135],
            ["New York, NY", "Philadelphia, PA", "Dry Van", 15000, "Normal Demand (Standard Rates)", 95]
        ],
        inputs=[origin_city, dest_city, equipment, weight, market_desc, distance],
        outputs=[out_rate, out_rpm, out_details],
        fn=predict_freight,
        cache_examples=False
    )

    predict_btn.click(
        fn=predict_freight,
        inputs=[origin_city, dest_city, equipment, weight, market_desc, distance],
        outputs=[out_rate, out_rpm, out_details]
    )

demo.launch(ssr_mode=False)
