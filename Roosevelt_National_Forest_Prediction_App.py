import streamlit as st
import pickle
import pandas as pd
import numpy as np
import os

# Page configuration
st.set_page_config(
    page_title="Roosevelt Forest Cover Prediction",
    page_icon="🌲",
    layout="wide"
)

# Title and header
st.markdown("""
<div style="background-color:#2E8B57;padding:15px;border-radius:10px;margin-bottom:20px">
<h1 style="color:white;text-align:center;margin:0">🌲 Roosevelt National Forest Prediction App 🌲</h1>
</div>""", unsafe_allow_html=True)

# About This Website section
with st.expander("ℹ️ About This Website"):
    st.markdown("""
    This project helps identify which types of trees grow in different areas of Roosevelt National Forest using a machine learning algorithm called CatBoost. Using topographic data such as elevation, slope, proximity to water sources, past fires, and roadways, as well as light availability and wilderness area designations, the model predicts which of 7 forest types is found in any given 30m x 30m patch of land. The problem is ecologically important because different tree species create distinct habitats that support diverse wildlife communities, making forest type prediction crucial for conservation and management efforts.
    
    The project serves dual purposes: scientific and educational. Scientifically, the model achieves strong performance (83% F1-macro score) with acceptable generalization capabilities (10% train-test gap), making it potentially suitable for ecological research applications. Additionally, it reduced the feature set from 54 to 11 and eliminated the need for the 40 soil type features in the original dataset. With the full dataset, Logistic Regression, SVM, Random Forest, XGBoost, LightGBM and CatBoost were tried. CatBoost outperformed every other model besides LightGBM, which had significant overfitting issues even after regularization efforts so it wasn't used for this project. When comparing simplified models, two versions of CatBoost were tested: the first achieved 80.5% F1-macro score with a 6.8% performance difference between training and testing, while the second achieved 83.0% F1-macro with a 10% difference. The second model was chosen for this website because it performed better on new data for every class even if it had a larger test/performance gap, but both versions are available in the GitHub repository. Educationally, the interactive web application allows users to explore machine learning predictions in real-time, visualize relationships between environmental variables and forest distributions, and gain insights into forest ecology through hands-on interaction with the model.
    
    The model and dataset are specifically based on Roosevelt National Forest in northern Colorado. Therefore, the predictions are tailored to this study area's unique environmental conditions and may not generalize to other geographic regions with different climatic patterns or topography. This regional specificity makes the model particularly valuable for local forest management decisions while serving as an educational case study for understanding how environmental factors influence forest cover types in Roosevelt National Forest.
    """)
    st.markdown("- Project Repository: https://github.com/hiker-drew/roosevelt-national-forest-prediction-app")

# Information sections
with st.expander("📚 Data Source and Source Materials"):
    st.write("""
    **Primary Data Source:**
    
    Forest Cover Type Dataset (US Forest Service & US Geological Survey) - Roosevelt National Forest, Northern Colorado - [https://www.kaggle.com/competitions/forest-cover-type-prediction/data?select=test.csv](https://www.kaggle.com/competitions/forest-cover-type-prediction/data?select=test.csv)
    
    **Forest Type Descriptions:**
    
    1. **Spruce/Fir** - [https://conps.org/project/subalpine-mesic-meadow/](https://conps.org/project/subalpine-mesic-meadow/)
    
    2. **Lodgepole Pine** - [https://www.nps.gov/articles/wildland-fire-lodgepole-pine.htm](https://www.nps.gov/articles/wildland-fire-lodgepole-pine.htm)
    
    3. **Ponderosa Pine** - [https://ucanr.edu/sites/forestry/Ecology/Identification/Ponderosa_Pine_Pinus_ponderosa/](https://ucanr.edu/sites/forestry/Ecology/Identification/Ponderosa_Pine_Pinus_ponderosa/)
    
    4. **Cottonwood/Willow** - [https://www.deschuteslandtrust.org/news/blog/2019-blog-posts/cottonwood-benefits](https://www.deschuteslandtrust.org/news/blog/2019-blog-posts/cottonwood-benefits) and [https://www.ndsu.edu/pubweb/chiwonlee/plsc368/student/papers98/willows.htm](https://www.ndsu.edu/pubweb/chiwonlee/plsc368/student/papers98/willows.htm)
    
    5. **Aspen** - [https://www.nationalforests.org/blog/tree-profile-aspen-so-much-more-than-a-tree](https://www.nationalforests.org/blog/tree-profile-aspen-so-much-more-than-a-tree)
    
    6. **Douglas-fir** - [https://ucanr.edu/sites/forestry/Ecology/Identification/Douglas-fir/](https://ucanr.edu/sites/forestry/Ecology/Identification/Douglas-fir/) and [https://www.fs.usda.gov/database/feis/plants/tree/psemeng/all.html](https://www.fs.usda.gov/database/feis/plants/tree/psemeng/all.html)
    
    7. **Krummholz** - [https://conps.org/project/subalpine-mesic-meadow/](https://conps.org/project/subalpine-mesic-meadow/)
    """)

st.markdown("---")

st.markdown("## 🌲 Forest Types:")

# Forest type descriptions - moved to main page
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **1. Spruce/Fir**  
    🏔️ Elevation: 2,525-3,675m  
    Typically found at higher elevations in cool, moist environments. These subalpine areas experience cold temperatures year-round with heavy snowfall that often remains on the ground well into summer.
    
    **2. Lodgepole Pine**  
    🏔️ Elevation: 2,169-3,413m  
    Their serotinous cones are adapted so that their sealing resin melts during a fire. These seeds are then in ideal conditions for germination with a bright open canopy and clear forest floor.
    
    **3. Ponderosa Pine**  
    🏔️ Elevation: 1,903-2,850m  
    Mature trees have thick, fire-resistant bark, open crowns, and self-pruning limbs that reduce fire damage.
    
    **4. Cottonwood/Willow**  
    🏔️ Elevation: 1,989-2,526m  
    Both species are found along rivers and streams where broken branches can go downriver and form the roots of new trees.
    """)

with col2:
    st.markdown("""
    **5. Aspen**  
    🏔️ Elevation: 2,482-3,007m  
    Deciduous trees known for their vibrant fall colors grow in large clonal colonies connected by shared root systems, making each grove a single organism. A pioneer species that quickly colonizes burned areas, but also requires abundant sunshine.
    
    **6. Douglas-fir**  
    🏔️ Elevation: 1,863-2,883m  
    Adaptable to diverse climates from coastal maritime conditions to harsh mountain environments. Important timber tree and is grown as a Christmas tree. Douglas-fir adapts to different slope aspects based on sunlight and moisture availability. Though the Rocky Mountain Douglas-fir are also versatile they are only found within the Rocky Mountains.
    
    **7. Krummholz**  
    🏔️ Elevation: 2,870-3,849m  
    At alpine elevations, harsh conditions including strong winds and ice cause trees to grow in twisted, wind-sculpted forms known as krummholz.
    """)

# Define the exact features used in your model
SELECTED_FEATURES = [
    'elevation',
    'horizontal_distance_to_roadways',
    'horizontal_distance_to_fire_points',
    'horizontal_distance_to_hydrology',
    'wilderness_area4',
    'hillshade_9am',
    'vertical_distance_to_hydrology',
    'wilderness_area1',
    'hillshade_noon',
    'wilderness_area3',
    'aspect'
]

# Feature ranges for UI sliders (min, max, default)
FEATURE_RANGES = {
    'elevation': (1863, 3849, 2752),
    'horizontal_distance_to_roadways': (0, 6890, 1316),
    'horizontal_distance_to_fire_points': (0, 6993, 1256),
    'horizontal_distance_to_hydrology': (0, 1343, 180),
    'wilderness_area4': (0, 1, 0),
    'hillshade_9am': (58, 254, 220),
    'vertical_distance_to_hydrology': (-146, 554, 32),
    'wilderness_area1': (0, 1, 0),
    'hillshade_noon': (99, 254, 223),
    'wilderness_area3': (0, 1, 0),
    'aspect': (0, 360, 126),
}

# Forest type mapping
FOREST_TYPES = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine", 
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

# Load model (no scaler needed anymore)
@st.cache_resource
def load_model():
    """Load the saved model"""
    model = None
    
    try:
        with open("catboost_forest_classifier.pkl", "rb") as f:
            model = pickle.load(f)
        st.success("✅ Model loaded successfully")
    except FileNotFoundError:
        st.error("❌ catboost_forest_classifier.pkl not found")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
    
    return model

# Create sidebar for inputs
st.sidebar.title('🌲 Enter Forest Characteristics:')

# Create input fields for all features
input_values = {}

# Continuous features
st.sidebar.subheader("📏 Terrain Measurements:")
st.sidebar.markdown("*(Measurements in meters and degrees)*")

# Elevation
feature = 'elevation'
min_val, max_val, default = FEATURE_RANGES[feature]
input_values[feature] = st.sidebar.slider(
    "Elevation - Height Above Sea Level",
    min_val, max_val, default
)

# Distance features
for feature in ['horizontal_distance_to_roadways', 
                'horizontal_distance_to_fire_points', 
                'horizontal_distance_to_hydrology',
                'vertical_distance_to_hydrology']:
    min_val, max_val, default = FEATURE_RANGES[feature]
    
    # Create readable labels
    if feature == 'horizontal_distance_to_roadways':
        label = "Distance To Roadways"
    elif feature == 'horizontal_distance_to_fire_points':
        label = "Distance To Previous Fires"
    elif feature == 'horizontal_distance_to_hydrology':
        label = "Horizontal Distance To Water Sources"
    elif feature == 'vertical_distance_to_hydrology':
        label = "Vertical Distance To Water Sources"
    
    input_values[feature] = st.sidebar.slider(
        label, min_val, max_val, default
    )

# Aspect (compass direction)
feature = 'aspect'
min_val, max_val, default = FEATURE_RANGES[feature]
input_values[feature] = st.sidebar.slider(
    "Aspect - Slope Compass Direction",
    min_val, max_val, default
)

# Hillshade features
st.sidebar.subheader("☀️ Amount of Sunlight:")
for feature in ['hillshade_9am', 'hillshade_noon']:
    min_val, max_val, default = FEATURE_RANGES[feature]
    label = "Sunlight at 9am" if feature == 'hillshade_9am' else "Sunlight at Noon"
    input_values[feature] = st.sidebar.slider(
        label, min_val, max_val, default
    )

# Wilderness areas (binary features)
st.sidebar.subheader("🏞️ Wilderness Areas:")
wilderness_selection = st.sidebar.radio(
    "",
    ["None", "Rawah Wilderness Area", "Comanche Peak Wilderness Area", "Cache la Poudre Wilderness Area"]
)

# Set binary values based on selection
input_values['wilderness_area1'] = 1 if wilderness_selection == "Rawah Wilderness Area" else 0
input_values['wilderness_area3'] = 1 if wilderness_selection == "Comanche Peak Wilderness Area" else 0
input_values['wilderness_area4'] = 1 if wilderness_selection == "Cache la Poudre Wilderness Area" else 0

st.markdown("---")

# Display current configuration
st.markdown("## ⚗️ Current Forest Configuration:")

# Create a more readable display
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🏔️ Terrain Features:")
    terrain_data = {
        "Elevation": f"{input_values['elevation']} m",
        "Aspect": f"{input_values['aspect']}°",
        "Distance to Roadways": f"{input_values['horizontal_distance_to_roadways']} m",
        "Distance to Previous Fires": f"{input_values['horizontal_distance_to_fire_points']} m",
        "Horizontal Distance To Water Sources": f"{input_values['horizontal_distance_to_hydrology']} m",
        "Vertical Distance To Water Sources": f"{input_values['vertical_distance_to_hydrology']} m"
    }
    for key, value in terrain_data.items():
        st.write(f"**{key}:** {value}")

with col2:
    st.markdown("#### ☀️ Light & Area Features:")
    st.write(f"**Sunlight at 9am:** {input_values['hillshade_9am']}")
    st.write(f"**Sunlight at Noon:** {input_values['hillshade_noon']}")
    st.write(f"**Wilderness Area:** {wilderness_selection}")

# Prediction section
st.markdown("### ✨ Press predict to classify forest:")

if st.button("Predict Forest Type", type="primary", use_container_width=True):
    model = load_model()
    
    if model is not None:
        try:
            # Create feature vector in the exact order specified
            feature_vector = [input_values[feature] for feature in SELECTED_FEATURES]
            
            # Create DataFrame with the exact feature names
            input_df = pd.DataFrame([feature_vector], columns=SELECTED_FEATURES)
            
            # Debug: Show what we're sending to the model
            with st.expander("🔍 Debug: Input Data"):
                st.write("**Raw Feature Values:**")
                for i, feature in enumerate(SELECTED_FEATURES):
                    st.write(f"{feature}: {feature_vector[i]}")
                st.write("**Input DataFrame Shape:**", input_df.shape)
                st.dataframe(input_df)
            
            # Make prediction directly without scaling
            prediction_raw = model.predict(input_df)
            
            # Extract the actual prediction value
            if isinstance(prediction_raw, np.ndarray):
                if prediction_raw.ndim > 1:
                    prediction = prediction_raw[0, 0]  # For 2D arrays
                else:
                    prediction = prediction_raw[0]  # For 1D arrays
            else:
                prediction = prediction_raw
                
            # Convert to Python int and add 1 to convert from 0-6 to 1-7
            prediction = int(prediction) + 1
            
            # Create columns for centered display
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                # Display the main prediction
                st.success(f"### 🌲 Predicted Forest Type:")
                st.success(f"### **{prediction}. {FOREST_TYPES.get(prediction, f'Type {prediction}')}**")
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(input_df)[0]
                
                # Find confidence
                max_prob = max(probabilities)
                confidence = max_prob * 100
                
                # Display confidence
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.info(f"**Confidence Level:** {confidence:.1f}%")
                
                # Create probability chart
                st.subheader("📊 Prediction Probabilities:")
                
                # Create probability data
                prob_data = []
                for i, prob in enumerate(probabilities):
                    # Model outputs 0-6, so we add 1 to get 1-7
                    forest_type_id = i + 1
                    
                    prob_data.append({
                        'Forest Type': FOREST_TYPES.get(forest_type_id, f'Type {forest_type_id}'),
                        'Probability (%)': float(prob) * 100
                    })
                
                prob_df = pd.DataFrame(prob_data)
                prob_df = prob_df.sort_values('Probability (%)', ascending=True)
                
                # Create horizontal bar chart
                st.bar_chart(prob_df.set_index('Forest Type')['Probability (%)'])
                
                # Show top 3 predictions
                st.subheader("🏆 Top 3 Most Likely Forest Types:")
                top_3 = prob_df.nlargest(3, 'Probability (%)')
                for idx, row in top_3.iterrows():
                    st.write(f"{row['Forest Type']}: {row['Probability (%)']:.1f}%")
                
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            
            # Show debugging information
            with st.expander("🐛 Debugging Information"):
                st.write("**Error details:**")
                st.code(str(e))
                st.write(f"**Feature vector length:** {len(feature_vector) if 'feature_vector' in locals() else 'N/A'}")
                st.write(f"**Expected features:** {len(SELECTED_FEATURES)}")
                st.write(f"**Input shape:** {input_df.shape if 'input_df' in locals() else 'N/A'}")
                if model and hasattr(model, 'n_features_in_'):
                    st.write(f"**Model expects:** {model.n_features_in_} features")
    else:
        st.error("❌ Model not loaded. Please ensure 'catboost_forest_classifier.pkl' is in the same directory as this app.")

# Footer
st.markdown("---")
st.markdown("🌲 Roosevelt National Forest Prediction App")
st.markdown("- Project Repository: https://github.com/hiker-drew/roosevelt-national-forest-prediction-app")