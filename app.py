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
# 64 Cities Database 
# ---------------------------------------------------------
US_CITIES = {
    'Albany': (43.9697, -71.0584), 'Albuquerque': (35.3992, -109.5979), 'Amarillo': (33.1714, -104.3679), 
    'Atlanta': (34.8493, -86.2894), 'Austin': (29.3955, -99.6978), 'Bakersfield': (33.2883, -116.6793), 
    'Baltimore': (38.1691, -72.7456), 'Baton Rouge': (30.5013, -92.6537), 'Birmingham': (33.4579, -86.7279), 
    'Boston': (39.2746, -69.5), 'Buffalo': (41.6768, -78.567), 'Charleston': (34.1533, -79.7447), 
    'Chattanooga': (35.2979, -87.1865), 'Cincinnati': (37.1571, -86.775), 'Columbia': (34.6357, -83.2851), 
    'Corpus Christi': (28.6137, -95.8957), 'Dallas': (31.8302, -94.3834), 'Dayton': (39.8721, -85.6293), 
    'Detroit': (43.6192, -84.0452), 'El Paso': (33.3107, -110.3123), 'Fort Wayne': (41.3156, -85.3621), 
    'Fresno': (33.0792, -119.7953), 'Grand Rapids': (43.438, -86.2917), 'Green Bay': (44.303, -87.5287), 
    'Greensboro': (35.9819, -82.7838), 'Harrisburg': (39.7169, -78.1248), 'Hartford': (39.5533, -72.1805), 
    'Houston': (30.7116, -94.7655), 'Indianapolis': (38.5257, -88.8595), 'Jacksonville': (31.9869, -84.489), 
    'Kansas City': (39.411, -91.1245), 'Las Vegas': (33.2235, -115.2499), 'Lexington': (36.9915, -84.9988), 
    'Little Rock': (34.2752, -92.0621), 'Los Angeles': (28.5662, -116.7025), 'Louisville': (37.3659, -87.4715), 
    'Lubbock': (29.8753, -99.8546), 'Madison': (41.7024, -89.8559), 'Memphis': (35.568, -89.5287), 
    'Milwaukee': (39.6218, -88.4801), 'Mobile': (31.8044, -88.2519), 'Montgomery': (31.376, -86.1074), 
    'Nashville': (35.2948, -88.0892), 'New Orleans': (32.5327, -92.1952), 'New York': (40.0067, -73.7984), 
    'Oklahoma City': (31.8829, -98.581), 'Philadelphia': (39.2232, -72.9671), 'Phoenix': (28.7567, -115.8773), 
    'Providence': (37.3729, -69.5), 'Raleigh': (35.3738, -77.4761), 'Reno': (39.5484, -118.6484), 
    'Richmond': (38.0912, -76.7891), 'Salt Lake City': (35.0349, -113.189), 'San Antonio': (29.5097, -98.4006), 
    'San Francisco': (35.1967, -121.6985), 'Savannah': (33.6097, -82.2999), 'Shreveport': (32.5463, -92.3189), 
    'St. Louis': (38.382, -89.8882), 'Syracuse': (42.8092, -76.8997), 'Tampa': (28.3576, -86.9478), 
    'Toledo': (41.9766, -84.459), 'Tucson': (30.9273, -113.2304), 'Tulsa': (33.4589, -94.7856), 
    'Washington': (38.8758, -77.0509)
}

# Equipment Options 
EQUIPMENT_CHOICES = [
    "Dry Van (Standard Enclosed Box)",
    "Reefer (Refrigerated Temperature-Controlled)",
    "Flatbed (Open Deck for Heavy Cargo)"
]

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
    c1 = US_CITIES.get(origin, (34.8493, -86.2894))
    c2 = US_CITIES.get(dest, (41.8781, -87.6298))
    return haversine_miles(c1[0], c1[1], c2[0], c2[1])

# ---------------------------------------------------------
# Prediction Function 
# ---------------------------------------------------------
@spaces.GPU
def predict_freight(origin_city, dest_city, equipment_label, weight, market_desc, custom_dist):
    if model is None:
        return "Model not loaded. Ensure freight_rate_model.joblib is present.", "N/A", "Offline"

    orig_lat, orig_lng = US_CITIES.get(origin_city, (34.8493, -86.2894))
    dest_lat, dest_lng = US_CITIES.get(dest_city, (41.8781, -87.6298))

    distance = custom_dist if custom_dist > 0 else haversine_miles(orig_lat, orig_lng, dest_lat, dest_lng)
    market_index = MARKET_PRESETS.get(market_desc, 1.0)
    quote_signal = 2.06

    # Feature Engineering 
    est_cost = distance * quote_signal
    market_presssure = market_index * quote_signal
    is_short_haul = 1 if distance < 250 else 0

    # Dummy variables matching pd.get_dummies (drop_first=True, Dry Van = 0, 0)
    eq_flatbed = 1 if equipment == "Flatbed" else 0
    eq_reefer = 1 if equipment == "Reefer" else 0

    now = datetime.now()
    month = now.month
    day_of_week = now.weekday()

    # 15 features  in order
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
    * **Equipment:** {equipment_label}
    * **Cargo Weight:** {weight:,} lbs
    * **Market Condition:** {market_desc}
    * **Estimated Operating Base Cost:** ${est_cost:,.2f}
    * **Model Architecture:** LightGBM L1 Regression (Holdout MAE: $114.36)
    """

    return rate_text, rpm_text, details

# ---------------------------------------------------------
# Clean Gradio UI
# ---------------------------------------------------------
city_list = sorted(list(US_CITIES.keys()))
# Reddish-Blue Theme & Styling
custom_theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="red",
    neutral_hue="slate"
)

custom_css = """
button.primary {
    background: linear-gradient(135deg, #1e40af 0%, #b91c1c 100%) !important;
    color: white !important;
    border: none !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}
button.primary:hover {
    background: linear-gradient(135deg, #2563eb 0%, #dc2626 100%) !important;
    transform: translateY(-1px);
}
"""

with gr.Blocks(theme=custom_theme, css=custom_css, title="US Freight Rate Prediction Engine") as demo:

    gr.Markdown(
        """
        # US Freight Rate Estimator
        Predict commercial truckload spot rates across all 64 major US shipping hubs using an optimized LightGBM regressor.
        """
    )

    # Model Performance Stats Bar
    gr.HTML(
        """
        <div style="background: rgba(30, 41, 59, 0.6); padding: 12px 18px; border-radius: 8px; border: 1px solid rgba(148, 163, 184, 0.2); margin-bottom: 16px; display: flex; flex-wrap: wrap; gap: 24px; justify-content: space-between;">
            <div><span style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">Model</span><br><strong style="color: #f8fafc; font-size: 1.05rem;">LightGBM (L1 Loss)</strong></div>
            <div><span style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">Holdout Accuracy</span><br><strong style="color: #38bdf8; font-size: 1.05rem;">$114.36 MAE</strong></div>
            <div><span style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">R² Score</span><br><strong style="color: #34d399; font-size: 1.05rem;">0.826</strong></div>
            <div><span style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">Optimization Gain</span><br><strong style="color: #f87171; font-size: 1.05rem;">23.8% Error Reduction</strong></div>
            <div><span style="color: #94a3b8; font-size: 0.8rem; text-transform: uppercase;">Validation</span><br><strong style="color: #f8fafc; font-size: 1.05rem;">Chronological (48k Loads)</strong></div>
        </div>
        """
    )


    with gr.Row():
        # Column 1: Inputs
        with gr.Column(scale=1):
            gr.Markdown("### Select Origin & Destination (64 Hubs Available)")
            with gr.Row():
                origin_city = gr.Dropdown(label="Origin Hub (Pickup)", choices=city_list, value="Atlanta")
                dest_city = gr.Dropdown(label="Destination Hub (Delivery)", choices=city_list, value="Chicago")

            gr.Markdown("### Load & Equipment Specifications")
            equipment = gr.Radio(
                label="Trailer Equipment Type",
                choices=["Dry Van", "Reefer", "Flatbed"],
                value="Dry Van",
                info="Dry Van: Standard enclosed box | Reefer: Refrigerated unit | Flatbed: Open deck"
            )

            weight = gr.Slider(label="Cargo Weight (lbs)", minimum=1000, maximum=48000, value=34000, step=1000)
            market_desc = gr.Dropdown(
                label="Market Capacity Condition",
                choices=list(MARKET_PRESETS.keys()),
                value="Normal Demand (Standard Rates)"
            )

            distance = gr.Slider(label="Estimated Highway Distance (miles)", minimum=40, maximum=3500, value=715, step=5)

            origin_city.change(fn=update_lane_distance, inputs=[origin_city, dest_city], outputs=distance)
            dest_city.change(fn=update_lane_distance, inputs=[origin_city, dest_city], outputs=distance)

            predict_btn = gr.Button("Calculate Rate", variant="primary", size="lg")

        # Column 2: Results
        with gr.Column(scale=1):
            gr.Markdown("### Rate Prediction")
            with gr.Row():
                out_rate = gr.Textbox(label="Estimated Posted Rate", elem_id="rate-box")
                out_rpm = gr.Textbox(label="Rate Per Mile (RPM)")

            out_details = gr.Markdown()

    # Full Width Examples at the Bottom
    gr.Markdown("### Quick Examples (Click to Test)")
    gr.Examples(
        examples=[
            ["Atlanta", "Chicago", "Dry Van", 34000, "Normal Demand (Standard Rates)", 715],
            ["Houston", "Dallas", "Flatbed", 42000, "Normal Demand (Standard Rates)", 240],
            ["Los Angeles", "San Francisco", "Reefer", 28000, "Tight Capacity / Peak Season (Surge)", 380],
            ["New York", "Philadelphia", "Dry Van", 15000, "Normal Demand (Standard Rates)", 95]
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
