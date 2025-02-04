import os
import logging
from typing import Optional, Dict, List
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from flask import Flask, redirect, render_template, request, url_for
from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    current_user,
    login_required
)
import joblib
from models.user_model import db, User

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Application Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models/outbreak_model.pkl")
DB_PATH = os.path.join(BASE_DIR, 'models/database.db')
TEMPLATE_DIR = os.path.join(BASE_DIR, "../templates")
STATIC_DIR = os.path.join(BASE_DIR, "../static")

# Initialize Flask
server = Flask(__name__, template_folder=TEMPLATE_DIR)
server.config.update(
    SECRET_KEY=os.getenv("SECRET_KEY", "your-secret-key-here"),
    SQLALCHEMY_DATABASE_URI=f"sqlite:///{DB_PATH}",
    SQLALCHEMY_TRACK_MODIFICATIONS=False
)

# Initialize Dash
app = dash.Dash(
    __name__,
    server=server,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.SOLAR],
    assets_folder=STATIC_DIR
)

# Initialize extensions
db.init_app(server)
login_manager = LoginManager()
login_manager.init_app(server)
login_manager.login_view = "login"

# Load ML model
try:
    model = joblib.load(MODEL_PATH)
    logger.info("ML model loaded successfully")
except FileNotFoundError:
    logger.error(f"Model file not found at: {MODEL_PATH}")
    raise

def create_risk_gauge(risk_probability: float) -> dcc.Graph:
    """Create a gauge chart for risk probability visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': 'darkblue'},
            'steps': [
                {'range': [0, 30], 'color': 'lightgreen'},
                {'range': [30, 70], 'color': 'yellow'},
                {'range': [70, 100], 'color': 'salmon'}
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=30, b=20),
        font={'size': 16}
    )
    
    return dcc.Graph(figure=fig)

def create_trend_chart(historical_data: pd.DataFrame) -> dcc.Graph:
    """Create a line chart for historical trend visualization."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=historical_data['date'],
            y=historical_data['risk_level'],
            name="Risk Level",
            line=dict(color='red', width=2)
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=historical_data['date'],
            y=historical_data['new_cases'],
            name="New Cases",
            line=dict(color='blue', width=2)
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title='Historical Trend Analysis',
        xaxis_title='Date',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return dcc.Graph(figure=fig)

def create_input_field(id_name: str, label: str, placeholder: str = "") -> html.Div:
    """Create a standardized input field with label."""
    return html.Div([
        html.Label(label, style={"color": "#333"}),
        dcc.Input(
            id=id_name,
            type="number",
            placeholder=placeholder,
            style={
                "marginBottom": "10px",
                "width": "100%",
                "padding": "8px",
                "borderRadius": "4px",
                "border": "1px solid #ddd"
            }
        )
    ])

def create_prediction_card(prediction_details: Dict) -> dbc.Card:
    """Create a detailed prediction card with multiple metrics."""
    return dbc.Card([
        dbc.CardHeader("Detailed Prediction Analysis"),
        dbc.CardBody([
            html.Div([
                html.H4(f"Risk Level: {prediction_details['risk_level']}", 
                       className="text-primary"),
                html.P(f"Confidence Score: {prediction_details['confidence']:.2f}%"),
                html.P(f"Key Factors:"),
                html.Ul([
                    html.Li(f"{factor}: {value}")
                    for factor, value in prediction_details['key_factors'].items()
                ]),
                html.P(f"Recommendation: {prediction_details['recommendation']}", 
                      className="mt-3 font-weight-bold")
            ])
        ])
    ])

def predict_outbreak(new_cases: float, humidity: float, population_density: float,
                    temperature: float, rainfall: float) -> str:
    """Make prediction using the loaded model."""
    try:
        prediction = model.predict([[
            new_cases,
            humidity,
            population_density,
            temperature,
            rainfall
        ]])[0]
        return "High" if prediction == 1 else "Low"
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise

def get_recommendation(risk_level: str, probability: float) -> str:
    """Generate recommendations based on risk level and probability."""
    if risk_level == "High" and probability > 0.7:
        return "Immediate action required. Implement strict containment measures."
    elif risk_level == "High":
        return "Increased monitoring recommended. Prepare containment measures."
    elif risk_level == "Low" and probability < 0.3:
        return "Continue normal monitoring procedures."
    else:
        return "Maintain vigilance and regular monitoring."

# Flask-Login user loader
@login_manager.user_loader
def load_user(user_id: str) -> Optional[User]:
    return User.query.get(int(user_id))

# Flask Routes
@server.route("/")
def home():
    return redirect("/login")

@server.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect("/dashboard")
        
    if request.method == "POST":
        user = User.query.filter_by(username=request.form["username"]).first()
        if user and user.check_password(request.form["password"]):
            login_user(user)
            return redirect("/dashboard")
        return "Invalid username or password"
    
    return render_template("login.html")

@server.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect("/dashboard")
        
    if request.method == "POST":
        if User.query.filter_by(username=request.form["username"]).first():
            return "Username already exists"
        
        new_user = User(
            username=request.form["username"],
            email=request.form["email"]
        )
        new_user.set_password(request.form["password"])
        
        db.session.add(new_user)
        db.session.commit()
        return redirect("/login")
    
    return render_template("register.html")

@server.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/login")

# Dash Layout
# Add these CSS styles at the top of your Dash layout
app.layout = html.Div([
    dcc.Location(id="url"),
    html.Div(id="page-content", style={
        "padding": "20px",
        "backgroundColor": "#1a1a1a",
        "minHeight": "100vh",
        "display": "flex",
        "justifyContent": "center",
        "alignItems": "flex-start"
    }),
    html.Link(
        rel="stylesheet",
        href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"
    )
])

def create_dashboard():
    return html.Div([
        dbc.Container([
            # Header Row with 3D Effect
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H1(
                            [
                                html.I(className="fas fa-virus mr-2"),
                                f"Welcome, {current_user.username}!"
                            ],
                            className="text-center mb-4 text-white",
                            style={
                                "textShadow": "2px 2px 4px rgba(0,0,0,0.5)",
                                "transform": "rotateX(20deg)",
                                "fontFamily": "'Arial Black', sans-serif"
                            }
                        ),
                        html.P(
                            "Disease Outbreak Detection System",
                            className="text-center text-light mb-4",
                            style={"fontSize": "1.2rem"}
                        )
                    ], className="neumorphic-box", style={
                        "padding": "20px",
                        "background": "linear-gradient(145deg, #1e1e1e, #2a2a2a)",
                        "borderRadius": "15px",
                        "boxShadow": "5px 5px 10px #0d0d0d, -5px -5px 10px #272727"
                    })
                ], width=12)
            ], className="mb-5"),

            # Input Form Row with Floating Labels
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader(
                            [
                                html.I(className="fas fa-chart-line mr-2"),
                                "Enter Outbreak Parameters"
                            ],
                            className="h4 text-light",
                            style={
                                "background": "linear-gradient(45deg, #2c3e50, #3498db)",
                                "borderRadius": "15px 15px 0 0"
                            }
                        ),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col(create_input_field("new-cases", "New Cases", "e.g., 150"), md=6),
                                dbc.Col(create_input_field("humidity", "Humidity (%)", "e.g., 65"), md=6)
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Col(create_input_field("population-density", "Population Density", "e.g., 200"), md=6),
                                dbc.Col(create_input_field("temperature", "Temperature (°C)", "e.g., 28"), md=6)
                            ], className="mb-3"),
                            dbc.Row([
                                dbc.Col(create_input_field("rainfall", "Rainfall (mm)", "e.g., 120"), md=6),
                            ]),
                            dbc.Button(
                                [
                                    html.I(className="fas fa-biohazard mr-2"),
                                    "Analyze Risk"
                                ],
                                id="predict-button",
                                color="danger",
                                className="w-100 mt-4 py-3",
                                style={
                                    "fontSize": "1.2rem",
                                    "boxShadow": "3px 3px 8px rgba(0,0,0,0.3)",
                                    "transition": "all 0.3s ease"
                                }
                            )
                        ], style={"background": "#ffffff"})
                    ], className="border-0 shadow-lg", style={"borderRadius": "15px"})
                ], md=8, sm=12)
            ], className="mb-5 justify-content-center"),

            # Prediction Output Row with Animated Cards
            dbc.Row([
                dbc.Col([
                    html.Div(
                        id="prediction-output",
                        className="mb-5",
                        style={"minHeight": "400px"}
                    )
                ], width=12)
            ], className="justify-content-center"),

            # Historical Trend Row with 3D Chart
            dbc.Row([
                dbc.Col([
                    html.Div(
                        id="historical-trend",
                        className="mb-5 p-4",
                        style={
                            "background": "#2a2a2a",
                            "borderRadius": "15px",
                            "boxShadow": "5px 5px 15px rgba(0,0,0,0.3)"
                        }
                    )
                ], width=12)
            ], className="justify-content-center"),

            # Footer with Glowing Effect
            dbc.Row([
                dbc.Col([
                    html.Hr(style={"borderColor": "#3498db"}),
                    html.Div([
                        html.A(
                            [
                                html.I(className="fas fa-sign-out-alt mr-2"),
                                "Logout"
                            ],
                            href="/logout",
                            className="btn btn-outline-danger btn-lg d-block mx-auto",
                            style={
                                "width": "200px",
                                "transition": "all 0.3s ease",
                                "boxShadow": "0 0 10px rgba(255,0,0,0.3)"
                            }
                        )
                    ], className="text-center")
                ], width=12)
            ], className="mt-5")
        ], fluid=True, className="px-4", style={"maxWidth": "1400px"})
    ], style={"backgroundColor": "#1a1a1a"})

# Update the create_risk_gauge function with 3D effect
def create_risk_gauge(risk_probability: float) -> dcc.Graph:
    """Create a 3D-style gauge chart for risk probability visualization."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': 'darkred', 'thickness': 0.3},
            'bgcolor': "black",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#2ecc71'},
                {'range': [30, 70], 'color': '#f1c40f'},
                {'range': [70, 100], 'color': '#e74c3c'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.8,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        font={'color': "white", 'size': 16},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title={
            'text': "Outbreak Risk Probability",
            'font': {'size': 24, 'color': "white"},
            'y': 0.9
        }
    )
    
    return dcc.Graph(figure=fig)

# Update the create_prediction_card function with modern styling
def create_prediction_card(prediction_details: Dict) -> dbc.Card:
    """Create a modern prediction card with animated elements."""
    risk_color = "#e74c3c" if prediction_details['risk_level'] == "High" else "#2ecc71"
    return dbc.Card([
        dbc.CardHeader(
            [
                html.I(className="fas fa-exclamation-triangle mr-2"),
                "Risk Analysis Report"
            ],
            style={
                "background": f"linear-gradient(45deg, {risk_color}, #2c3e50)",
                "color": "white",
                "fontSize": "1.25rem"
            }
        ),
        dbc.CardBody([
            html.Div([
                html.H4(
                    f"Risk Level: {prediction_details['risk_level']}",
                    className="mb-4",
                    style={"color": risk_color}
                ),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.P("Confidence Score:", className="text-muted"),
                            html.H3(
                                f"{prediction_details['confidence']:.1f}%",
                                style={"color": "#3498db"}
                            )
                        ], className="text-center p-3 neumorphic-box")
                    ], md=4),
                    dbc.Col([
                        html.Div([
                            html.P("Key Risk Factors:", className="text-muted"),
                            html.Ul([
                                html.Li(
                                    [
                                        html.I(className="fas fa-arrow-circle-up mr-2"),
                                        f"{factor}: {value}"
                                    ],
                                    className="my-2"
                                ) for factor, value in prediction_details['key_factors'].items()
                            ], className="list-unstyled")
                        ], className="p-3")
                    ], md=8)
                ]),
                html.Div([
                    html.P(
                        "Recommendation:",
                        className="text-warning font-weight-bold mb-2"
                    ),
                    html.Div(
                        prediction_details['recommendation'],
                        className="p-3 alert alert-danger",
                        role="alert",
                        style={"borderRadius": "10px"}
                    )
                ], className="mt-4")
            ])
        ], style={"backgroundColor": "#2a2a2a", "color": "white"})
    ], className="shadow-lg", style={"borderRadius": "15px", "border": "none"})

# Update the create_trend_chart function with dark theme
def create_trend_chart(historical_data: pd.DataFrame) -> dcc.Graph:
    """Create a dark-themed line chart for historical trend visualization."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=historical_data['date'],
            y=historical_data['risk_level'],
            name="Risk Level",
            line=dict(color='#e74c3c', width=3, shape='spline'),
            fill='tozeroy',
            fillcolor='rgba(231,76,60,0.1)'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Bar(
            x=historical_data['date'],
            y=historical_data['new_cases'],
            name="New Cases",
            marker_color='#3498db',
            opacity=0.7
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title={
            'text': 'Historical Trend Analysis',
            'font': {'color': 'white', 'size': 24}
        },
        xaxis={
            'title': 'Date',
            'gridcolor': '#444',
            'color': 'white'
        },
        yaxis={
            'title': 'Risk Level',
            'gridcolor': '#444',
            'color': 'white'
        },
        yaxis2={
            'title': 'New Cases',
            'gridcolor': '#444',
            'color': 'white'
        },
        height=500,
        margin=dict(l=20, r=20, t=80, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="white")
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x unified"
    )
    
    return dcc.Graph(figure=fig)

# Add hover effects and transitions to buttons in callback
@app.callback(
    Output("predict-button", "style"),
    [Input("predict-button", "n_clicks")]
)
def animate_button(n_clicks):
    if n_clicks:
        return {"transform": "scale(0.95)", "boxShadow": "1px 1px 5px rgba(0,0,0,0.3)"}
    return {"boxShadow": "3px 3px 8px rgba(0,0,0,0.3)"}
@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def display_page(pathname):
    if pathname == "/dashboard":
        if not current_user.is_authenticated:
            return dcc.Location(id='redirect', href="/login")
        return create_dashboard()
    return dcc.Location(id='redirect', href="/login")

@app.callback(
    [
        Output("prediction-output", "children"),
        Output("historical-trend", "children")
    ],
    [Input("predict-button", "n_clicks")],
    [
        State("new-cases", "value"),
        State("humidity", "value"),
        State("population-density", "value"),
        State("temperature", "value"),
        State("rainfall", "value")
    ]
)
def update_dashboard(n_clicks, new_cases, humidity, population_density, temperature, rainfall):
    if not n_clicks:
        return "", ""
    
    try:
        # Validate inputs
        parameters = {
            'New Cases': new_cases,
            'Humidity': humidity,
            'Population Density': population_density,
            'Temperature': temperature,
            'Rainfall': rainfall
        }
        
        if any(v is None for v in parameters.values()):
            missing_fields = [k for k, v in parameters.items() if v is None]
            return dbc.Alert(
                f"Please enter values for: {', '.join(missing_fields)}",
                color="warning"
            ), ""
        
        # Make prediction
        risk_level = predict_outbreak(new_cases, humidity, population_density, temperature, rainfall)
        
        # Calculate confidence and risk probability (example calculation)
        confidence = np.random.uniform(75, 95)  # Replace with actual confidence calculation
        risk_probability = 0.8 if risk_level == "High" else 0.2  # Replace with actual probability
        
        # Generate prediction details
        prediction_details = {
            'risk_level': risk_level,
            'confidence': confidence,
            'key_factors': {
                'Population Density Impact': f"{population_density:.1f} people/km²",
                'Environmental Risk': f"{(humidity * temperature / 100):.1f}",
                'Current Spread Rate': f"{(new_cases / population_density):.2f}"
            },
            'recommendation': get_recommendation(risk_level, risk_probability)
        }
        
        # Generate sample historical data (replace with actual data)
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        historical_data = pd.DataFrame({
            'date': dates,
            'risk_level': np.random.uniform(0, 1, 30),
            'new_cases': np.random.normal(new_cases, new_cases*0.1, 30)
        })
        
        # Create prediction output
        prediction_output = dbc.Container([
            dbc.Row([
                dbc.Col([
                    create_risk_gauge(risk_probability)
                ], md=6, sm=12),
                dbc.Col([
                    create_prediction_card(prediction_details)
                ], md=6, sm=12)
            ]),
            dbc.Row([
                dbc.Col([
                    html.A(
                        dbc.Button(
                            "Return to Dashboard",
                            color="primary",
                            className="mt-3"
                        ),
                        href="/dashboard"
                    )
                ])
            ])
        ])
        
        # Create historical trend
        trend_output = create_trend_chart(historical_data)
        
        return prediction_output, trend_output
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return dbc.Alert(
            "An error occurred during prediction. Please try again.",
            color="danger"
        ), ""

if __name__ == "__main__":
    with server.app_context():
        db.create_all()
    
    logger.info("Starting the server...")
    server.run(debug=False, host="0.0.0.0", port=8050)