import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from utils_fn import embarked_imputation, age_imputation, feature_engineer, drop_features, as_category, full_preprocess


# ===================================
# PAGE CONFIG
# ===================================
st.set_page_config(
    page_title="Titanic Dashboard",
    page_icon="🚢",
    layout="wide",
)
# ===================================
# CUSTOM CSS
# ===================================
def load_css(path: str):
    css_path = Path(__file__).parent / path
    with open(css_path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")

# ===================================
# DATA & MODEL LOADERS (cached)
# ===================================
BASE_DIR = Path(__file__).parent

@st.cache_data
def load_train():
    return pd.read_csv(BASE_DIR / "data" / "train.csv")

@st.cache_resource
def load_model():
    with open(BASE_DIR / "gradient_model.pkl", "rb") as f:
        return pickle.load(f)
    
# ===================================
# LOAD
# ===================================
try:
    train = load_train()
    DATA_OK = True
except Exception as e:
    st.warning(f"⚠️ The train.csv file was not found. Some sections will not be available. ({e})")
    DATA_OK = False
    train = pd.DataFrame()
 
try:
    model = load_model()
    MODEL_OK = True
except Exception as e:
    st.warning(f"⚠️ gradient_model.pkl was not found. The prediction will not be available. ({e})")
    MODEL_OK = False
    model = None

# ===================================
# Head
# ===================================
h1, h2 = st.columns([2, 1])
with h1:
    st.title("Titanic Survival Dashboard")
    st.markdown("""
    This app performs simple exploratory analysis and survival prediction using the classic Kaggle dataset.
    * **Python libraries:** plotly, pandas, numpy, streamlit
    * **Data source:** [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic/).
    """)

with h2:
    st.space()
    st.image(
        BASE_DIR / "gm_generated_image-titanic.png",
        use_container_width=False,
        caption="RMS Titanic, 1912",
        width=400
    )

# ===================================
# SideBar
# ===================================
df = load_train()
with st.sidebar:
    # st.markdown("## 🚢 Titanic Dashboard")
    # st.markdown("Análisis y modelo predictivo basado en el dataset de Kaggle.")
    # st.divider()
 
    if DATA_OK:
        st.markdown("### Global filters")
        clase_sel = st.multiselect(
            "Class (Pclass)", options=[1, 2, 3], default=[1, 2, 3]
        )
        sexo_sel = st.multiselect(
            "Sex", options=["male", "female"], default=["male", "female"]
        )
        edad_sel = st.slider(
            "Age Rate", min_value=0, max_value=80, value=(0, 80)
        )
        # st.divider()
 
    # st.markdown(
    #     "<small style='opacity:.6'>Portfolio project · Kaggle Titanic</small>",
    #     unsafe_allow_html=True,
    # )

# ===================================
#  Section 1 · KPIs
# ===================================
if DATA_OK:
    st.header("General Metrics")
    # st.divider()
 
    # Apply sidebar filters
    mask = (
        train['Pclass'].isin(clase_sel)
        & train['Sex'].isin(sexo_sel)
        & train['Age'].between(edad_sel[0], edad_sel[1], inclusive='both')
    )
    df_f = train[mask]
 
    total      = len(df_f)
    survived   = int(df_f['Survived'].sum())
    not_surv   = total - survived
    surv_rate  = df_f['Survived'].mean() if total > 0 else 0
    avg_age    = df_f['Age'].mean()
    avg_fare   = df_f['Fare'].mean()
 
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("**Passengers**",        f"{total:,}")
    k2.metric("**Survived**",          f"{survived:,}")
    k3.metric("**No Survived**",       f"{not_surv:,}")
    k4.metric("**Survived Rate**",     f"{surv_rate:.1%}")
    k5.metric("**Average age**",       f"{avg_age:.1f}" if not np.isnan(avg_age) else "N/A")

    st.write('Data Dimension: ' + str(df_f.shape[0]) + ' rows and ' + str(df_f.shape[1]) + ' columns.')
    st.dataframe(df_f)
    # st.divider()


    # ===================================
    # Charts row 1 | Distributions
    # ===================================
    st.header("Distributions")
    c1, c2 = st.columns(2)
    no_color = '#4B7CA3'
    yes_color = '#9BC4E5'

    # Survival by Sex
    surv_sex = (
        df_f.groupby(['Sex', 'Survived']).size()
        .reset_index(name='count')
    )
    surv_sex['Survived'] = surv_sex['Survived'].map({0: 'No', 1: 'Yes'})
    fig1 = px.bar(
        surv_sex, x='Sex', y='count', color='Survived',
        barmode='group', color_discrete_map={'No': no_color, 'Yes': yes_color},
        labels={'count': 'Passengers'},
        title='Survival by Sex',
    )
    # fig1.update_layout(plot_bgcolor='#f8f9fb', paper_bgcolor='#f8f9fb')
    c1.plotly_chart(fig1, use_container_width=True)
 
    # Survival by Pclass
    surv_class = (
        df_f.groupby(['Pclass', 'Survived']).size()
        .reset_index(name='count')
    )
    surv_class['Survived'] = surv_class['Survived'].map({0: 'No', 1: 'Yes'})
    fig2 = px.bar(
        surv_class, x='Pclass', y='count', color='Survived',
        barmode='group', color_discrete_map={'No': no_color, 'Yes': yes_color},
        labels={'count': 'Passengers', 'Pclass': 'Class'},
        title='Survival by Class',
    )
    # fig2.update_layout(plot_bgcolor='#f8f9fb', paper_bgcolor='#f8f9fb')
    c2.plotly_chart(fig2, use_container_width=True)

    # ===================================
    # Charts row 2
    # ===================================
    c3, c4 = st.columns(2)
 
    # Age distribution by survival
    df_age = df_f.dropna(subset=['Age'])
    df_age = df_age.copy()
    df_age['Survived'] = df_age['Survived'].map({0: 'No', 1: 'Yes'})
    fig3 = px.histogram(
        df_age, x='Age', color='Survived', nbins=30, barmode='overlay',
        opacity=0.7,
        color_discrete_map={'No': no_color, 'Yes': yes_color},
        title='Age distribution by survival',
        # labels={'count': 'Passengers'},
    )
    # fig3.update_layout(plot_bgcolor='#f8f9fb', paper_bgcolor='#f8f9fb')
    c3.plotly_chart(fig3, use_container_width=True)
 
    # Survival by Family Size
    df_fs = df_f.copy()
    df_fs['familySize'] = df_fs['SibSp'] + df_fs['Parch'] + 1
    fs_rate = (
        df_fs.groupby('familySize')['Survived'].mean().reset_index()
    )
    fig4 = px.bar(
        fs_rate, x='familySize', y='Survived',
        color='Survived',
                                           # orange
        color_continuous_scale=[no_color, '#1A4C78', yes_color],
        labels={'familySize': 'Family Size', 'Survived': 'Survived Rate'},
        title='Survival by Family Size',
    )
    # fig4.update_layout(plot_bgcolor='#f8f9fb', paper_bgcolor='#f8f9fb', coloraxis_showscale=False)
    c4.plotly_chart(fig4, use_container_width=True)
 
    # st.divider()
 
# ===================================
#  Section 2 | Model results
# ===================================
st.header("Model results")
 
col_met, col_chart = st.columns(([1, 2]))

 
with col_met:
    st.space()
    st.markdown("**Evaluation Metrics** (CV × 10 folds)")
    st.metric("Accuracy",  "82.7%")
    st.metric("F1 Macro",  "80.9%")
    st.metric("Error rate","17.3%")
    st.caption("Model: GradientBoostingClassifier · lr=0.05 · n=100")
 
with col_chart:
    # Feature importance proxy - hardcoded from notebook results
    feat_imp = pd.DataFrame({
        'Feature':    ['Sex', 'Fare', 'Age', 'Pclass', 'familySize', 'Embarked'],
        'Importance': [ 0.31,   0.22,  0.18,     0.16,         0.08,       0.05],
    }).sort_values('Importance', ascending=True)
 
    fig_fi = px.bar(
        feat_imp, x='Importance', y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale=['#aed6f1', '#1a5276'],
        title='Importance of variables (estimated)',
    )
    fig_fi.update_layout(
        # plot_bgcolor='#f8f9fb', paper_bgcolor='#f8f9fb',
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_fi, use_container_width=True)
 
st.divider()
 
 
# ==============================================
# Section 3 in SIDEBAR | Interactive prediction 
# ==============================================
with st.sidebar:
    st.divider()
    st.markdown("### Interactive prediction")
    st.markdown("Fill in the passenger's details and the model will predict whether they would have survived.")

    pclass_in   = st.selectbox("Class", [1, 2, 3], index=2)
    sex_in      = st.radio("Sex", ["male", "female"], horizontal=False)
    age_in      = st.slider("Age", min_value=1, max_value=80, value=30)
    fare_in     = st.slider("Fare ($)", min_value=0.0, max_value=512.3, value=14.45, step=0.5)
    # fare_in     = st.number_input("Fare ($)", min_value=0.0, value=14.45, step=0.5)
    embarked_in = st.selectbox("Embarked", ["S", "C", "Q"], help="S=Southampton · C=Cherbourg · Q=Queenstown")
    sibsp_in    = st.number_input("Siblings / Spouse on board", min_value=0, max_value=8, value=0)
    parch_in    = st.number_input("Parents / Childs on board", min_value=0, max_value=6, value=0)
 
    predecir_btn = st.button("Predict survival &#8674;", use_container_width=True)


st.header("Interactive prediction")
st.markdown("Fill in the passenger's details and the model will predict whether they would have survived.")
 
# -------------------------
# Button response
# -------------------------

if predecir_btn:
    if not MODEL_OK:
        st.error("The model is unavailable. Make sure `gradient_model.pkl` is in your project folder.")
    else:
        # Build raw input — same columns as original dataset
        input_raw = pd.DataFrame({
            'Pclass':    [pclass_in],
            'Sex':       [sex_in],
            'Age':       [float(age_in)],
            'Fare':      [fare_in],
            'Embarked':  [embarked_in],
            'SibSp':     [sibsp_in],
            'Parch':     [parch_in],
        })

        # Apply same preprocessing as in the notebook
        input_proc = feature_engineer(input_raw)      # adds familySize
        input_proc = drop_features(input_proc)         # drops SibSp, Parch
        input_proc = as_category(input_proc, ['Sex', 'Embarked'])

        pred = model.predict(input_proc)[0]
        prob = model.predict_proba(input_proc)[0][1]

        # Result card
        if pred == 1:
            st.success("✅ **Would have survived!**")
            color = "#2ecc71"
        else:
            st.error("❌ **Would not have survived!**")
            color = "#e74c3c"

        st.metric("Probability of survival", f"{prob:.1%}")

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={'suffix': '%', 'font': {'size': 28}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 40],   'color': '#fde8e8'},
                    {'range': [40, 60],  'color': '#fef9e7'},
                    {'range': [60, 100], 'color': '#e8f8f1'},
                ],
                'threshold': {
                    'line': {'color': '#2c3e50', 'width': 3},
                    'thickness': 0.8,
                    'value': 50,
                },
            },
            title={'text': "Probability of survival"},
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=50, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

else:
    st.info("&#8672; Fill out the form and click on **Predict** to see the result.")

st.space()
st.space()
st.space()
# st.markdown(
# # You can see how this works in the Quickview Jupyter Notebook or see the code.

#     "<small style='opacity:.6'> You can see how this works in the Quickview Jupyter Notebook or see the code.[Titanic](https://www.kaggle.com/competitions/titanic/)."
#     "By Natalia Bonilla V.</small>",
#     unsafe_allow_html=True,
# )

st.header("About this app")
st.markdown("This app displays data from Kaggle at https://www.kaggle.com/competitions/titanic.")
st.markdown("You can see how this works in the quick view of the [Jupyter Notebook](https://github.com/nbonillav/kaggle-competitions/blob/main/titanic-projectv2.ipynb)")
