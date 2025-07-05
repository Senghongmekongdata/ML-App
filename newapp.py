import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression

# Classification models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

# Regression models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

# Clustering
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

# Metrics
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           mean_squared_error, mean_absolute_error, r2_score, roc_auc_score,
                           precision_recall_curve, roc_curve, silhouette_score)

# Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

# Data analysis
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Advanced ML Model Training & Analysis App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .stButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Create directories
for directory in ['models', 'model_metadata', 'experiments', 'data_profiles']:
    if not os.path.exists(directory):
        os.makedirs(directory)

class AdvancedMLApp:
    def __init__(self):
        self.models_dir = 'models'
        self.metadata_dir = 'model_metadata'
        self.experiments_dir = 'experiments'
        self.profiles_dir = 'data_profiles'

        # Define available models with more options
        self.classification_models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'Extra Trees': ExtraTreesClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Ridge Classifier': RidgeClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'KNN': KNeighborsClassifier(),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Naive Bayes': GaussianNB(),
            'Neural Network': MLPClassifier(random_state=42, max_iter=500)
        }

        self.regression_models = {
            'Random Forest': RandomForestRegressor(random_state=42),
            'Extra Trees': ExtraTreesRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'AdaBoost': AdaBoostRegressor(random_state=42),
            'Linear Regression': LinearRegression(),
            'Ridge': Ridge(random_state=42),
            'Lasso': Lasso(random_state=42),
            'ElasticNet': ElasticNet(random_state=42),
            'SVR': SVR(),
            'KNN': KNeighborsRegressor(),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'Neural Network': MLPRegressor(random_state=42, max_iter=500)
        }

        self.clustering_models = {
            'K-Means': KMeans(random_state=42, n_init='auto'),
            'DBSCAN': DBSCAN(),
            'Agglomerative': AgglomerativeClustering(),
            'Gaussian Mixture': GaussianMixture(random_state=42)
        }

        self.scalers = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'None': None
        }

    def generate_data_profile(self, df, profile_name):
        """Generate comprehensive data profile"""
        profile = {
            'name': profile_name,
            'created_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'numeric_summary': {},
            'categorical_summary': {},
            'correlations': {},
            'outliers': {}
        }

        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            profile['numeric_summary'][col] = {
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis()
            }

            # Outlier detection using IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)][col]
            profile['outliers'][col] = len(outliers)

        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            profile['categorical_summary'][col] = {
                'unique_values': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'value_counts': df[col].value_counts().head(10).to_dict()
            }

        # Correlation matrix for numeric columns
        if len(numeric_cols) > 1:
            profile['correlations'] = df[numeric_cols].corr().to_dict()

        # Save profile
        profile_path = os.path.join(self.profiles_dir, f"{profile_name}.json")
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=4, default=str)

        return profile

    def auto_feature_selection(self, X, y, problem_type, k=10):
        """Automatic feature selection"""
        if problem_type == 'Classification':
            selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        else:
            selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))

        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        feature_scores = selector.scores_

        return X_selected, selected_features, feature_scores

    def hyperparameter_tuning(self, model, X, y, param_grid, cv=5):
        """Perform hyperparameter tuning"""
        grid_search = GridSearchCV(
            model, param_grid, cv=cv, scoring='accuracy' if hasattr(model, 'predict_proba') else 'r2',
            n_jobs=-1, verbose=0
        )
        grid_search.fit(X, y)
        return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

    def model_comparison(self, X, y, problem_type, cv=5):
        """Compare multiple models"""
        if problem_type == 'Classification':
            models = self.classification_models
            scoring = 'accuracy'
        else:
            models = self.regression_models
            scoring = 'r2'

        results = {}
        for name, model in models.items():
            try:
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
                results[name] = {
                    'mean_score': scores.mean(),
                    'std_score': scores.std(),
                    'scores': scores.tolist()
                }
            except Exception as e:
                results[name] = {
                    'mean_score': 0,
                    'std_score': 0,
                    'scores': [],
                    'error': str(e)
                }

        return results

    def plot_learning_curves(self, model, X, y, title="Learning Curves"):
        """Plot learning curves"""
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )

        fig, ax = plt.subplots(figsize=(10, 6))

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        val_scores_mean = np.mean(val_scores, axis=1)
        val_scores_std = np.std(val_scores, axis=1)

        ax.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training score')
        ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                       train_scores_mean + train_scores_std, alpha=0.1, color='blue')

        ax.plot(train_sizes, val_scores_mean, 'o-', color='red', label='Cross-validation score')
        ax.fill_between(train_sizes, val_scores_mean - val_scores_std,
                       val_scores_mean + val_scores_std, alpha=0.1, color='red')

        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Score')
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True)

        return fig

    def create_advanced_visualizations(self, df):
        """Create advanced data visualizations"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        visualizations = []

        # Correlation heatmap
        if len(numeric_cols) > 1:
            fig_corr = px.imshow(
                df[numeric_cols].corr(),
                title="Correlation Heatmap",
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            visualizations.append(('Correlation Heatmap', fig_corr))

        # Distribution plots
        if len(numeric_cols) > 0:
            fig_dist = make_subplots(
                rows=min(3, len(numeric_cols)),
                cols=min(3, (len(numeric_cols) + 2) // 3),
                subplot_titles=numeric_cols[:9]
            )

            for i, col in enumerate(numeric_cols[:9]):
                row = i // 3 + 1
                col_idx = i % 3 + 1
                fig_dist.add_trace(
                    go.Histogram(x=df[col], name=col, showlegend=False),
                    row=row, col=col_idx
                )

            fig_dist.update_layout(title_text="Distribution Plots", height=600)
            visualizations.append(('Distributions', fig_dist))

        return visualizations

    def dimensionality_reduction(self, X, method='PCA', n_components=2):
        """Perform dimensionality reduction"""
        if method == 'PCA':
            reducer = PCA(n_components=n_components)
        elif method == 'TSNE':
            reducer = TSNE(n_components=n_components, random_state=42)
        elif method == 'UMAP':
            reducer = umap.UMAP(n_components=n_components, random_state=42)

        X_reduced = reducer.fit_transform(X)
        return X_reduced, reducer

    def save_experiment(self, experiment_data):
        """Save experiment results"""
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiment_path = os.path.join(self.experiments_dir, f"{experiment_id}.json")

        with open(experiment_path, 'w') as f:
            json.dump(experiment_data, f, indent=4, default=str)

        return experiment_id

    def load_experiment(self, experiment_id):
        """Load experiment results"""
        experiment_path = os.path.join(self.experiments_dir, f"{experiment_id}.json")

        try:
            with open(experiment_path, 'r') as f:
                return json.load(f)
        except:
            return None

    def get_experiments(self):
        """Get list of saved experiments"""
        try:
            experiments = []
            for file in os.listdir(self.experiments_dir):
                if file.endswith('.json'):
                    experiments.append(file[:-5])  # Remove .json extension
            return sorted(experiments, reverse=True)
        except:
            return []

    # Include all the original methods from the base class
    def save_model(self, model, model_name, metadata):
        """Save model and its metadata"""
        try:
            model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)

            metadata_path = os.path.join(self.metadata_dir, f"{model_name}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4, default=str)

            return True
        except Exception as e:
            st.error(f"Error saving model: {str(e)}")
            return False

    def load_model(self, model_name):
        """Load model and its metadata"""
        try:
            model_path = os.path.join(self.models_dir, f"{model_name}.pkl")
            metadata_path = os.path.join(self.metadata_dir, f"{model_name}.json")

            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            return model, metadata
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None, None

    def get_saved_models(self):
        """Get list of saved models"""
        try:
            models = []
            for file in os.listdir(self.models_dir):
                if file.endswith('.pkl'):
                    model_name = file[:-4]
                    models.append(model_name)
            return models
        except:
            return []

    def preprocess_data(self, X, y=None, is_training=True, preprocessor=None):
        """Preprocess the data"""
        if is_training:
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
            numerical_features = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
            boolean_features = X.select_dtypes(include=['bool']).columns.tolist()
            categorical_features.extend(boolean_features)

            transformers = []

            if numerical_features:
                transformers.append(('num', StandardScaler(), numerical_features))

            if categorical_features:
                transformers.append(('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_features))

            if transformers:
                preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
            else:
                from sklearn.preprocessing import FunctionTransformer
                preprocessor = FunctionTransformer(lambda x: x)

            X_processed = preprocessor.fit_transform(X)
            return X_processed, preprocessor
        else:
            X_processed = preprocessor.transform(X)
            return X_processed

def main():
    app = AdvancedMLApp()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Advanced ML Model Training & Analysis Platform</h1>
        <p>Comprehensive Machine Learning Workflow Management</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("🔧 Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "📤 Data Upload & Profiling",
        "📊 Exploratory Data Analysis",
        "🔍 Feature Engineering",
        "🏋️ Model Training",
        "⚖️ Model Comparison",
        "🎯 Hyperparameter Tuning",
        "🔮 Model Usage",
        "📚 Model Library",
        "🧪 Experiments",
        "🔬 Clustering Analysis"
    ])

    if page == "📤 Data Upload & Profiling":
        st.header("📤 Data Upload & Comprehensive Profiling")

        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state['data'] = df

                st.success(f"✅ Data uploaded successfully! Shape: {df.shape}")

                # Generate data profile
                profile_name = st.text_input("Enter profile name:", value=f"profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

                if st.button("🔍 Generate Data Profile"):
                    with st.spinner("Generating comprehensive data profile..."):
                        profile = app.generate_data_profile(df, profile_name)
                        st.session_state['current_profile'] = profile
                        st.success("✅ Data profile generated!")

                # Display profile if available
                if 'current_profile' in st.session_state:
                    profile = st.session_state['current_profile']

                    # Overview metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Rows", profile['shape'][0])
                    with col2:
                        st.metric("Columns", profile['shape'][1])
                    with col3:
                        st.metric("Missing Values", sum(profile['missing_values'].values()))
                    with col4:
                        st.metric("Numeric Columns", len(profile['numeric_summary']))

                    # Data quality issues
                    st.subheader("🚨 Data Quality Issues")
                    missing_cols = [col for col, pct in profile['missing_percentage'].items() if pct > 0]
                    if missing_cols:
                        st.warning(f"Columns with missing values: {', '.join(missing_cols)}")

                    outlier_cols = [col for col, count in profile['outliers'].items() if count > 0]
                    if outlier_cols:
                        st.warning(f"Columns with outliers: {', '.join(outlier_cols)}")

                    # Detailed statistics
                    if st.expander("📊 Detailed Statistics"):
                        st.json(profile['numeric_summary'])

                # Basic data preview
                st.subheader("🔍 Data Preview")
                st.dataframe(df.head(10))

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

    elif page == "📊 Exploratory Data Analysis":
        st.header("📊 Exploratory Data Analysis")

        if 'data' not in st.session_state:
            st.warning("⚠️ Please upload data first.")
            return

        df = st.session_state['data']

        # Create advanced visualizations
        with st.spinner("Creating visualizations..."):
            visualizations = app.create_advanced_visualizations(df)

        for title, fig in visualizations:
            st.subheader(title)
            st.plotly_chart(fig, use_container_width=True)

        # Statistical tests
        st.subheader("🔬 Statistical Analysis")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                var1 = st.selectbox("Select first variable", numeric_cols)
            with col2:
                var2 = st.selectbox("Select second variable", numeric_cols, index=1)

            if st.button("🔍 Perform Correlation Test"):
                correlation, p_value = stats.pearsonr(df[var1].dropna(), df[var2].dropna())
                st.metric("Correlation Coefficient", f"{correlation:.4f}")
                st.metric("P-value", f"{p_value:.4f}")

                if p_value < 0.05:
                    st.success("✅ Statistically significant correlation")
                else:
                    st.warning("⚠️ No significant correlation")

    elif page == "🔍 Feature Engineering":
        st.header("🔍 Feature Engineering")

        if 'data' not in st.session_state:
            st.warning("⚠️ Please upload data first.")
            return

        df = st.session_state['data']

        # Feature selection
        st.subheader("🎯 Automatic Feature Selection")

        features = st.multiselect("Select Features", df.columns.tolist())
        target = st.selectbox("Select Target Variable", df.columns.tolist())
        problem_type = st.selectbox("Problem Type", ["Classification", "Regression"])

        if features and target:
            k_features = st.slider("Number of features to select", 1, min(10, len(features)), 5)

            if st.button("🔍 Perform Feature Selection"):
                try:
                    X = df[features].copy()
                    y = df[target].copy()

                    # Handle missing values
                    X = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])
                    y = y.fillna(y.median() if pd.api.types.is_numeric_dtype(y) else y.mode().iloc[0])

                    # Encode categorical variables
                    for col in X.columns:
                        if X[col].dtype == 'object':
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col].astype(str))

                    if problem_type == "Classification" and y.dtype == 'object':
                        le = LabelEncoder()
                        y = le.fit_transform(y)

                    # Perform feature selection
                    X_selected, selected_features, feature_scores = app.auto_feature_selection(
                        X, y, problem_type, k_features
                    )

                    # Display results
                    st.success(f"✅ Selected {len(selected_features)} features")

                    # Feature importance
                    importance_df = pd.DataFrame({
                        'Feature': features,
                        'Score': feature_scores,
                        'Selected': [f in selected_features for f in features]
                    }).sort_values('Score', ascending=False)

                    st.dataframe(importance_df)

                    # Store selected features
                    st.session_state['selected_features'] = selected_features

                except Exception as e:
                    st.error(f"Error in feature selection: {str(e)}")

        # Dimensionality reduction
        st.subheader("📉 Dimensionality Reduction")

        if features:
            reduction_method = st.selectbox("Select Method", ["PCA", "TSNE", "UMAP"])
            n_components = st.slider("Number of Components", 2, min(5, len(features)), 2)

            if st.button("🔄 Apply Dimensionality Reduction"):
                try:
                    X = df[features].select_dtypes(include=[np.number])
                    if X.empty:
                        st.error("No numeric features selected for dimensionality reduction")
                        return

                    X = X.fillna(X.median())

                    X_reduced, reducer = app.dimensionality_reduction(X, reduction_method, n_components)

                    # Plot results
                    if n_components == 2:
                        fig = px.scatter(
                            x=X_reduced[:, 0], y=X_reduced[:, 1],
                            title=f"{reduction_method} Visualization",
                            labels={'x': f'{reduction_method}1', 'y': f'{reduction_method}2'}
                        )
                        st.plotly_chart(fig, use_container_width=True)

                    # Show explained variance for PCA
                    if reduction_method == "PCA":
                        st.metric("Explained Variance Ratio", f"{reducer.explained_variance_ratio_.sum():.4f}")

                except Exception as e:
                    st.error(f"Error in dimensionality reduction: {str(e)}")

    elif page == "⚖️ Model Comparison":
        st.header("⚖️ Model Comparison")

        if 'data' not in st.session_state:
            st.warning("⚠️ Please upload data first.")
            return

        df = st.session_state['data']

        # Model comparison setup
        features = st.multiselect("Select Features", df.columns.tolist())
        target = st.selectbox("Select Target Variable", df.columns.tolist())
        problem_type = st.selectbox("Problem Type", ["Classification", "Regression"])

        if features and target:
            cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)

            if st.button("🏆 Compare Models"):
                try:
                    # Prepare data
                    X = df[features].copy()
                    y = df[target].copy()

                    # Handle missing values
                    for col in X.columns:
                        if X[col].dtype in ['object', 'category']:
                            X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 'Unknown')
                        else:
                            X[col] = X[col].fillna(X[col].median())

                    # Encode categorical variables
                    for col in X.columns:
                        if X[col].dtype == 'object':
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col].astype(str))

                    if problem_type == "Classification" and y.dtype == 'object':
                        le = LabelEncoder()
                        y = le.fit_transform(y)
                    elif problem_type == "Regression":
                        y = y.fillna(y.median())

                    # Compare models
                    with st.spinner("Comparing models..."):
                        results = app.model_comparison(X, y, problem_type, cv_folds)

                    # Display results
                    results_df = pd.DataFrame.from_dict(results, orient='index')
                    results_df = results_df.sort_values('mean_score', ascending=False)

                    st.subheader("🏆 Model Comparison Results")
                    st.dataframe(results_df[['mean_score', 'std_score']])

                    # Visualization
                    fig = px.bar(
                        x=results_df.index,
                        y=results_df['mean_score'],
                        error_y=results_df['std_score'],
                        title="Model Performance Comparison"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Save experiment
                    experiment_data = {
                        'type': 'model_comparison',
                        'problem_type': problem_type,
                        'features': features,
                        'target': target,
                        'cv_folds': cv_folds,
                        'results': results,
                        'best_model': results_df.index[0]
                    }

                    experiment_id = app.save_experiment(experiment_data)
                    st.success(f"✅ Experiment saved as: {experiment_id}")

                except Exception as e:
                    st.error(f"Error in model comparison: {str(e)}")

    elif page == "🎯 Hyperparameter Tuning":
        st.header("🎯 Hyperparameter Tuning")

        if 'data' not in st.session_state:
            st.warning("⚠️ Please upload data first.")
            return

        df = st.session_state['data']

        # Hyperparameter tuning setup
        features = st.multiselect("Select Features", df.columns.tolist())
        target = st.selectbox("Select Target Variable", df.columns.tolist())
        problem_type = st.selectbox("Problem Type", ["Classification", "Regression"])

        if features and target:
            if problem_type == "Classification":
                model_name = st.selectbox("Select Model", list(app.classification_models.keys()))
                base_model = app.classification_models[model_name]
            else:
                model_name = st.selectbox("Select Model", list(app.regression_models.keys()))
                base_model = app.regression_models[model_name]

            # Define parameter grids
            param_grids = {
                'Random Forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                },
                'Gradient Boosting': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                },
                'SVM': {
                    'C': [0.1, 1, 10],
                    'kernel': ['rbf', 'linear'],
                    'gamma': ['scale', 'auto']
                },
                'KNN': {
                    'n_neighbors': [3, 5, 7, 9],
                    'weights': ['uniform', 'distance']
                }
            }

            # Get parameter grid for selected model
            if model_name in param_grids:
                param_grid = param_grids[model_name]

                st.subheader("🔧 Parameter Grid")
                st.json(param_grid)

                cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)

                if st.button("🚀 Start Hyperparameter Tuning"):
                    try:
                        # Prepare data
                        X = df[features].copy()
                        y = df[target].copy()

                        # Handle missing values and encoding
                        for col in X.columns:
                            if X[col].dtype in ['object', 'category']:
                                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 'Unknown')
                            else:
                                X[col] = X[col].fillna(X[col].median())

                        for col in X.columns:
                            if X[col].dtype == 'object':
                                le = LabelEncoder()
                                X[col] = le.fit_transform(X[col].astype(str))

                        if problem_type == "Classification" and y.dtype == 'object':
                            le = LabelEncoder()
                            y = le.fit_transform(y)
                        elif problem_type == "Regression":
                            y = y.fillna(y.median())

                        # Perform hyperparameter tuning
                        with st.spinner("Tuning hyperparameters... This may take a while."):
                            best_model, best_params, best_score = app.hyperparameter_tuning(
                                base_model, X, y, param_grid, cv_folds
                            )

                        # Display results
                        st.success(f"✅ Hyperparameter tuning completed!")
                        st.metric("Best Score", f"{best_score:.4f}")

                        st.subheader("🏆 Best Parameters")
                        st.json(best_params)

                        # Store the best model
                        st.session_state['tuned_model'] = best_model
                        st.session_state['tuned_model_info'] = {
                            'model_name': model_name,
                            'best_params': best_params,
                            'best_score': best_score,
                            'features': features,
                            'target': target,
                            'problem_type': problem_type
                        }

                        # Learning curves
                        st.subheader("📈 Learning Curves")
                        fig = app.plot_learning_curves(best_model, X, y, f"{model_name} Learning Curves")
                        st.pyplot(fig)

                    except Exception as e:
                        st.error(f"Error in hyperparameter tuning: {str(e)}")
            else:
                st.info("Hyperparameter tuning not available for this model.")

    elif page == "🏋️ Model Training":
        st.header("🏋️ Advanced Model Training")

        if 'data' not in st.session_state:
            st.warning("⚠️ Please upload data first.")
            return

        df = st.session_state['data']

        # Advanced training options
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 Training Configuration")
            problem_type = st.selectbox("Problem Type", ["Classification", "Regression"])
            features = st.multiselect("Select Features", df.columns.tolist())
            target = st.selectbox("Select Target Variable", df.columns.tolist())

            # Use selected features from feature engineering if available
            if 'selected_features' in st.session_state:
                if st.checkbox("Use Auto-Selected Features"):
                    features = st.session_state['selected_features']
                    st.info(f"Using {len(features)} auto-selected features")

        with col2:
            st.subheader("🔧 Advanced Options")
            scaler_type = st.selectbox("Select Scaler", list(app.scalers.keys()))
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
            cross_validation = st.checkbox("Enable Cross-Validation", value=True)
            if cross_validation:
                cv_folds = st.slider("CV Folds", 3, 10, 5)

        if features and target:
            # Model selection
            if problem_type == "Classification":
                model_name = st.selectbox("Select Model", list(app.classification_models.keys()))
                selected_model = app.classification_models[model_name]
            else:
                model_name = st.selectbox("Select Model", list(app.regression_models.keys()))
                selected_model = app.regression_models[model_name]

            if st.button("🚀 Train Advanced Model"):
                try:
                    # Prepare data
                    X = df[features].copy()
                    y = df[target].copy()

                    # Handle missing values
                    for col in X.columns:
                        if X[col].dtype in ['object', 'category']:
                            X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 'Unknown')
                        else:
                            X[col] = X[col].fillna(X[col].median())

                    if y.dtype in ['object', 'category']:
                        y = y.fillna(y.mode().iloc[0] if not y.mode().empty else 'Unknown')
                    else:
                        y = y.fillna(y.median())

                    # Encoding
                    label_encoder = None
                    if problem_type == "Classification" and y.dtype == 'object':
                        label_encoder = LabelEncoder()
                        y = label_encoder.fit_transform(y)

                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42,
                        stratify=y if problem_type == "Classification" else None
                    )

                    # Create preprocessing pipeline
                    preprocessor = None
                    if scaler_type != 'None':
                        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
                        numerical_features = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()

                        transformers = []
                        if numerical_features:
                            transformers.append(('num', app.scalers[scaler_type], numerical_features))
                        if categorical_features:
                            transformers.append(('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_features))

                        if transformers:
                            preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
                    else:
                        # If no scaler, still need a preprocessor for one-hot encoding if categorical features exist
                        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
                        numerical_features = X.select_dtypes(include=[np.number]).columns.tolist() # Numeric columns for passthrough

                        transformers = []
                        if categorical_features:
                            transformers.append(('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_features))

                        if transformers:
                            preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')


                    # Create and train pipeline
                    if preprocessor:
                        pipeline = Pipeline([
                            ('preprocessor', preprocessor),
                            ('model', selected_model)
                        ])
                    else:
                        pipeline = Pipeline([('model', selected_model)])

                    # Training with progress
                    with st.spinner("Training model..."):
                        pipeline.fit(X_train, y_train)

                    # Predictions
                    y_pred = pipeline.predict(X_test)

                    # Evaluation
                    if problem_type == "Classification":
                        if label_encoder:
                            y_test_orig = label_encoder.inverse_transform(y_test)
                            y_pred_orig = label_encoder.inverse_transform(y_pred)
                            score = accuracy_score(y_test_orig, y_pred_orig)

                            # Additional classification metrics
                            st.subheader("📊 Classification Results")
                            col1, col2 = st.columns(2)

                            with col1:
                                st.metric("Accuracy", f"{score:.4f}")

                                # Classification report
                                report = classification_report(y_test_orig, y_pred_orig, output_dict=True)
                                st.dataframe(pd.DataFrame(report).transpose())

                            with col2:
                                # Confusion matrix
                                cm = confusion_matrix(y_test_orig, y_pred_orig)
                                fig, ax = plt.subplots(figsize=(8, 6))
                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                                ax.set_title('Confusion Matrix')
                                st.pyplot(fig)
                        else:
                            score = accuracy_score(y_test, y_pred)
                    else:
                        # Regression metrics
                        score = r2_score(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        mae = mean_absolute_error(y_test, y_pred)

                        st.subheader("📊 Regression Results")
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("R² Score", f"{score:.4f}")
                            st.metric("MSE", f"{mse:.4f}")
                            st.metric("MAE", f"{mae:.4f}")

                        with col2:
                            # Actual vs Predicted plot
                            fig, ax = plt.subplots(figsize=(8, 6))
                            ax.scatter(y_test, y_pred, alpha=0.6)
                            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                            ax.set_xlabel('Actual Values')
                            ax.set_ylabel('Predicted Values')
                            ax.set_title('Actual vs Predicted')
                            st.pyplot(fig)

                    # Cross-validation if enabled
                    if cross_validation:
                        st.subheader("🔄 Cross-Validation Results")
                        cv_scores = cross_val_score(
                            pipeline, X, y, cv=cv_folds,
                            scoring='accuracy' if problem_type == "Classification" else 'r2'
                        )

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("CV Mean", f"{cv_scores.mean():.4f}")
                        with col2:
                            st.metric("CV Std", f"{cv_scores.std():.4f}")
                        with col3:
                            st.metric("CV Range", f"{cv_scores.max() - cv_scores.min():.4f}")

                    # Learning curves
                    st.subheader("📈 Learning Curves")
                    fig = app.plot_learning_curves(pipeline, X, y, f"{model_name} Learning Curves")
                    st.pyplot(fig)

                    # Store model for saving
                    st.session_state['trained_model'] = pipeline
                    st.session_state['model_info'] = {
                        'model_name': model_name,
                        'problem_type': problem_type,
                        'features': features,
                        'target': target,
                        'score': score,
                        'test_size': test_size,
                        'scaler_type': scaler_type,
                        'label_encoder': label_encoder
                    }

                    st.success("✅ Model trained successfully!")

                except Exception as e:
                    st.error(f"Error training model: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())

            # Save model section
            if 'trained_model' in st.session_state:
                st.markdown("---")
                st.subheader("💾 Save Model")

                col1, col2 = st.columns(2)
                with col1:
                    save_name = st.text_input("Enter model name to save:")
                with col2:
                    model_description = st.text_area("Model description (optional):")

                if st.button("💾 Save Model") and save_name:
                    try:
                        model_info = st.session_state['model_info']

                        metadata = {
                            'model_name': save_name,
                            'description': model_description,
                            'original_model': model_info['model_name'],
                            'problem_type': model_info['problem_type'],
                            'features': model_info['features'],
                            'target': model_info['target'],
                            'score': model_info['score'],
                            'test_size': model_info['test_size'],
                            'scaler_type': model_info['scaler_type'],
                            'created_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'data_shape': df.shape
                        }

                        if app.save_model(st.session_state['trained_model'], save_name, metadata):
                            st.success(f"✅ Model '{save_name}' saved successfully!")
                        else:
                            st.error("❌ Failed to save model.")
                    except Exception as e:
                        st.error(f"Error saving model: {str(e)}")

    elif page == "🔮 Model Usage":
        st.header("🔮 Advanced Model Usage")

        saved_models = app.get_saved_models()

        if not saved_models:
            st.warning("⚠️ No saved models found. Please train and save a model first.")
            return

        selected_model_name = st.selectbox("Select a saved model", saved_models)

        if selected_model_name:
            model, metadata = app.load_model(selected_model_name)

            if model and metadata:
                # Enhanced model info display
                st.subheader("ℹ️ Model Information")

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"**Model:** {metadata['original_model']}")
                    st.info(f"**Problem:** {metadata['problem_type']}")
                    st.info(f"**Score:** {metadata['score']:.4f}")

                with col2:
                    st.info(f"**Features:** {len(metadata['features'])}")
                    st.info(f"**Target:** {metadata['target']}")
                    st.info(f"**Created:** {metadata['created_date']}")

                with col3:
                    if 'description' in metadata and metadata['description']:
                        st.info(f"**Description:** {metadata['description']}")
                    st.info(f"**Data Shape:** {metadata['data_shape']}")

                # Feature importance (if available)
                if hasattr(model.named_steps.get('model', model), 'feature_importances_'):
                    st.subheader("📊 Feature Importance")
                    try:
                        # Handle pipeline vs direct model
                        if 'preprocessor' in model.named_steps:
                            # Get feature names after one-hot encoding if applicable
                            preprocessor = model.named_steps['preprocessor']
                            ohe_cols = []
                            if hasattr(preprocessor, 'transformers_'):
                                for name, transformer, cols in preprocessor.transformers_:
                                    if isinstance(transformer, OneHotEncoder):
                                        ohe_cols.extend(transformer.get_feature_names_out(cols))
                                    elif name == 'num':
                                        ohe_cols.extend(cols) # numerical features just pass through
                            feature_names = ohe_cols
                        else:
                            feature_names = metadata['features']

                        importances = model.named_steps.get('model', model).feature_importances_

                        # Ensure importances and feature_names match in length or handle appropriately
                        if len(importances) == len(feature_names):
                            importance_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': importances
                            }).sort_values('Importance', ascending=False)

                            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                                       title="Feature Importance")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.warning("Could not display feature importance due to mismatch in feature names and importance array length.")
                    except Exception as e:
                        st.warning(f"Could not display feature importance: {e}")


                # Prediction interface
                st.markdown("---")
                prediction_type = st.radio("Choose prediction type",
                                         ["Single Prediction", "Batch Prediction", "Interactive Prediction"])

                if prediction_type == "Single Prediction":
                    st.subheader("🎯 Single Prediction")

                    # Dynamic input fields
                    input_data = {}
                    num_cols = min(3, len(metadata['features']))
                    cols = st.columns(num_cols)

                    for i, feature in enumerate(metadata['features']):
                        with cols[i % num_cols]:
                            # Attempt to pre-fill based on data type, though this is heuristic
                            # For more robust solution, store data types/ranges in metadata
                            if df[feature].dtype in ['int64', 'float64']:
                                # Try to get min/max from original data for sensible numeric input
                                input_data[feature] = st.number_input(f"Enter {feature} (numeric):", value=float(df[feature].mean()) if feature in df.columns else 0.0)
                            elif df[feature].dtype == 'object' or df[feature].dtype == 'category':
                                unique_vals = df[feature].dropna().unique().tolist()
                                if len(unique_vals) <= 20: # Use selectbox for fewer unique values
                                    input_data[feature] = st.selectbox(f"Select {feature}:", unique_vals)
                                else:
                                    input_data[feature] = st.text_input(f"Enter {feature} (categorical):")
                            else:
                                input_data[feature] = st.text_input(f"Enter {feature}:")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("🔮 Make Prediction"):
                            try:
                                # Create a DataFrame from the input data, ensuring correct order and types
                                input_df_raw = pd.DataFrame([input_data])
                                # Reindex to ensure feature order matches training
                                input_df_raw = input_df_raw.reindex(columns=metadata['features'])

                                prediction = model.predict(input_df_raw) # Model pipeline handles preprocessing
                                st.success(f"🎯 Prediction: **{prediction[0]}**")

                                # Prediction confidence (if available)
                                if hasattr(model, 'predict_proba') and metadata['problem_type'] == 'Classification':
                                    try:
                                        proba = model.predict_proba(input_df_raw)
                                        confidence = np.max(proba) * 100
                                        st.metric("Confidence", f"{confidence:.1f}%")
                                    except Exception as prob_e:
                                        st.warning(f"Could not get prediction probabilities: {prob_e}")

                            except Exception as e:
                                st.error(f"Error making prediction: {str(e)}")
                                import traceback
                                st.error(traceback.format_exc())

                    with col2:
                        if st.button("📊 Show Prediction Details"):
                            if metadata['problem_type'] == 'Classification' and hasattr(model, 'predict_proba'):
                                try:
                                    input_df_raw = pd.DataFrame([input_data])
                                    input_df_raw = input_df_raw.reindex(columns=metadata['features'])

                                    proba = model.predict_proba(input_df_raw)
                                    classes = model.classes_ if hasattr(model, 'classes_') else range(proba.shape[1])

                                    prob_df = pd.DataFrame({
                                        'Class': classes,
                                        'Probability': proba[0]
                                    }).sort_values('Probability', ascending=False)

                                    fig = px.bar(prob_df, x='Class', y='Probability',
                                               title="Prediction Probabilities")
                                    st.plotly_chart(fig, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error showing details: {str(e)}")

                elif prediction_type == "Interactive Prediction":
                    st.subheader("🎮 Interactive Prediction")

                    # Create sliders/inputs for numeric features
                    input_data = {}

                    for feature in metadata['features']:
                        # Attempt to pre-fill based on data type, though this is heuristic
                        # For more robust solution, store data types/ranges in metadata
                        if df[feature].dtype in ['int64', 'float64']:
                            min_val = float(df[feature].min()) if feature in df.columns and not df[feature].empty else 0.0
                            max_val = float(df[feature].max()) if feature in df.columns and not df[feature].empty else 100.0
                            default_val = float(df[feature].mean()) if feature in df.columns and not df[feature].empty else 0.0
                            if max_val == min_val: # Avoid division by zero for slider step
                                default_val = min_val
                                step = 1.0
                            else:
                                step = (max_val - min_val) / 20.0 # 20 steps
                                if step == 0: step = 1.0
                            input_data[feature] = st.slider(f"{feature} (numeric):", min_value=min_val, max_value=max_val, value=default_val, step=step)
                        elif df[feature].dtype == 'object' or df[feature].dtype == 'category':
                            unique_vals = df[feature].dropna().unique().tolist()
                            if len(unique_vals) > 0:
                                input_data[feature] = st.selectbox(f"Select {feature} (categorical):", unique_vals)
                            else:
                                input_data[feature] = st.text_input(f"Enter {feature} (categorical):")
                        else:
                            input_data[feature] = st.text_input(f"Enter {feature}:")

                    # Real-time prediction
                    try:
                        input_df_raw = pd.DataFrame([input_data])
                        input_df_raw = input_df_raw.reindex(columns=metadata['features'])

                        prediction = model.predict(input_df_raw)

                        st.markdown("### 🎯 Live Prediction")
                        st.metric("Prediction", f"{prediction[0]}")

                        # Live probability chart for classification
                        if metadata['problem_type'] == 'Classification' and hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(input_df_raw)
                            classes = model.classes_ if hasattr(model, 'classes_') else range(proba.shape[1])

                            prob_df = pd.DataFrame({
                                'Class': classes,
                                'Probability': proba[0]
                            })

                            fig = px.bar(prob_df, x='Class', y='Probability',
                                       title="Live Prediction Probabilities")
                            st.plotly_chart(fig, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error in interactive prediction: {str(e)}")

                else:  # Batch Prediction
                    st.subheader("📊 Enhanced Batch Prediction")

                    uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type="csv")

                    if uploaded_file is not None:
                        try:
                            batch_df = pd.read_csv(uploaded_file)

                            # Feature validation
                            missing_features = set(metadata['features']) - set(batch_df.columns)
                            # extra_features = set(batch_df.columns) - set(metadata['features']) # This is okay, will be ignored by model

                            if missing_features:
                                st.error(f"❌ Missing required features in uploaded file: {missing_features}")
                                return

                            batch_X = batch_df[metadata['features']].copy() # Ensure we only use model's expected features

                            st.subheader("🔍 Batch Data Preview")
                            st.dataframe(batch_X.head())

                            # Batch prediction options
                            col1, col2 = st.columns(2)
                            with col1:
                                include_probabilities = st.checkbox("Include Prediction Probabilities",
                                                                   value=metadata['problem_type']=='Classification')
                            with col2:
                                include_original = st.checkbox("Include Original Data", value=True)

                            if st.button("🚀 Make Batch Predictions"):
                                try:
                                    with st.spinner("Making batch predictions..."):
                                        predictions = model.predict(batch_X)

                                    # Create results dataframe
                                    if include_original:
                                        results_df = batch_df.copy()
                                    else:
                                        results_df = batch_X.copy()

                                    # For classification, convert predictions back if label encoder was used
                                    if metadata['problem_type'] == 'Classification' and metadata.get('label_encoder'):
                                        # This requires saving/loading the label encoder, which is not currently done
                                        # For simplicity here, we'll assume the model's direct output is desired or a pre-trained encoder exists
                                        # A more robust solution would pickle the label_encoder along with the model's metadata
                                        st.warning("Label decoding for classification predictions requires the original LabelEncoder object, which is not currently persisted with the model. Displaying raw predictions.")
                                        results_df[f'Predicted_{metadata["target"]}'] = predictions
                                    else:
                                        results_df[f'Predicted_{metadata["target"]}'] = predictions

                                    # Add probabilities for classification
                                    if include_probabilities and metadata['problem_type'] == 'Classification':
                                        if hasattr(model, 'predict_proba'):
                                            probabilities = model.predict_proba(batch_X)
                                            classes = model.classes_ if hasattr(model, 'classes_') else range(probabilities.shape[1])

                                            for i, class_name in enumerate(classes):
                                                results_df[f'Prob_{class_name}'] = probabilities[:, i]
                                        else:
                                            st.warning("Selected model does not support prediction probabilities.")

                                    st.subheader("📋 Batch Prediction Results")
                                    st.dataframe(results_df)

                                    # Summary statistics
                                    st.subheader("📊 Prediction Summary")
                                    if metadata['problem_type'] == 'Classification':
                                        pred_counts = pd.Series(predictions).value_counts()
                                        fig = px.pie(values=pred_counts.values, names=pred_counts.index,
                                                   title="Prediction Distribution")
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        fig = px.histogram(x=predictions, title="Prediction Distribution")
                                        st.plotly_chart(fig, use_container_width=True)

                                    # Download results
                                    csv = results_df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download Results CSV",
                                        data=csv,
                                        file_name=f"batch_predictions_{selected_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                        mime="text/csv"
                                    )

                                except Exception as e:
                                    st.error(f"Error making batch predictions: {str(e)}")
                                    import traceback
                                    st.error(traceback.format_exc())

                        except Exception as e:
                            st.error(f"Error reading file: {str(e)}")

    elif page == "📚 Model Library":
        st.header("📚 Model Library")
        st.write("Browse and manage your saved models and experiments.")

        # Display saved models
        st.subheader("🗃️ Saved Models")
        saved_models = app.get_saved_models()
        if saved_models:
            for model_name in saved_models:
                col1, col2 = st.columns([3,1])
                with col1:
                    st.write(f"- {model_name}")
                with col2:
                    if st.button(f"Details {model_name}", key=f"details_{model_name}"):
                        model, metadata = app.load_model(model_name)
                        if model and metadata:
                            st.json(metadata)
        else:
            st.info("No models saved yet.")

        # Display saved experiments
        st.subheader("🧪 Saved Experiments")
        saved_experiments = app.get_experiments()
        if saved_experiments:
            for exp_id in saved_experiments:
                col1, col2 = st.columns([3,1])
                with col1:
                    st.write(f"- {exp_id}")
                with col2:
                    if st.button(f"View {exp_id}", key=f"view_{exp_id}"):
                        experiment_data = app.load_experiment(exp_id)
                        if experiment_data:
                            st.json(experiment_data)
        else:
            st.info("No experiments saved yet.")


    elif page == "🧪 Experiments":
        st.header("🧪 Experiment Tracking & Analysis")
        st.write("Track and analyze the results of your model training and comparison experiments.")

        saved_experiments = app.get_experiments()

        if not saved_experiments:
            st.warning("⚠️ No experiments found. Please run Model Comparison or Hyperparameter Tuning to save experiments.")
            return

        selected_experiment_id = st.selectbox("Select an Experiment", saved_experiments)

        if selected_experiment_id:
            experiment_data = app.load_experiment(selected_experiment_id)

            if experiment_data:
                st.subheader(f"📈 Experiment Details: {selected_experiment_id}")
                st.json(experiment_data) # Display raw JSON for now

                if experiment_data.get('type') == 'model_comparison':
                    st.subheader("📊 Model Comparison Summary")
                    results_df = pd.DataFrame.from_dict(experiment_data['results'], orient='index')
                    results_df = results_df.sort_values('mean_score', ascending=False)
                    st.dataframe(results_df[['mean_score', 'std_score']])

                    fig = px.bar(
                        x=results_df.index,
                        y=results_df['mean_score'],
                        error_y=results_df['std_score'],
                        title="Model Performance Comparison"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                elif experiment_data.get('type') == 'hyperparameter_tuning':
                    st.subheader("🎯 Hyperparameter Tuning Summary")
                    st.write(f"**Model:** {experiment_data['model_name']}")
                    st.write(f"**Best Score:** {experiment_data['best_score']:.4f}")
                    st.write("**Best Parameters:**")
                    st.json(experiment_data['best_params'])

            else:
                st.error("Could not load experiment details.")


    elif page == "🔬 Clustering Analysis":
        st.header("🔬 Clustering Analysis")

        if 'data' not in st.session_state:
            st.warning("⚠️ Please upload data first.")
            return

        df = st.session_state['data']

        # Clustering setup
        st.subheader("⚙️ Clustering Configuration")

        col1, col2 = st.columns(2)

        with col1:
            numeric_df = df.select_dtypes(include=[np.number])
            features = st.multiselect("Select Features for Clustering",
                                    numeric_df.columns.tolist())
            clustering_method = st.selectbox("Select Clustering Method", list(app.clustering_models.keys()))

        with col2:
            clusterer_params = {}
            if clustering_method == 'K-Means':
                n_clusters = st.slider("Number of Clusters", 2, 10, 3)
                clusterer_params = {'n_clusters': n_clusters}
            elif clustering_method == 'DBSCAN':
                eps = st.slider("Epsilon", 0.1, 2.0, 0.5, 0.1)
                min_samples = st.slider("Min Samples", 2, 10, 5)
                clusterer_params = {'eps': eps, 'min_samples': min_samples}
            elif clustering_method == 'Agglomerative':
                n_clusters_agg = st.slider("Number of Clusters", 2, 10, 3)
                clusterer_params = {'n_clusters': n_clusters_agg}
            elif clustering_method == 'Gaussian Mixture':
                n_components = st.slider("Number of Components", 2, 10, 3)
                clusterer_params = {'n_components': n_components}


        if features and len(features) >= 1:
            if st.button("🔬 Perform Clustering"):
                try:
                    # Prepare data
                    X = df[features].copy()
                    X = X.fillna(X.median()) # Handle NaNs with median imputation for numeric data

                    # Scale the data
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    # Configure clustering model
                    if clustering_method == 'K-Means':
                        clusterer = KMeans(random_state=42, **clusterer_params)
                    elif clustering_method == 'DBSCAN':
                        clusterer = DBSCAN(**clusterer_params)
                    elif clustering_method == 'Agglomerative':
                        clusterer = AgglomerativeClustering(**clusterer_params)
                    elif clustering_method == 'Gaussian Mixture':
                        clusterer = GaussianMixture(random_state=42, **clusterer_params)

                    # Perform clustering
                    with st.spinner(f"Performing {clustering_method} clustering..."):
                        if clustering_method == 'Gaussian Mixture':
                            cluster_labels = clusterer.fit_predict(X_scaled)
                        else:
                            cluster_labels = clusterer.fit_predict(X_scaled)

                    # Calculate silhouette score (only if more than 1 cluster and no noise points)
                    if len(set(cluster_labels)) > 1 and -1 not in cluster_labels: # -1 indicates noise for DBSCAN
                        try:
                            silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                            st.metric("Silhouette Score", f"{silhouette_avg:.4f}")
                        except Exception as sil_e:
                            st.warning(f"Could not calculate Silhouette Score: {sil_e}")
                    else:
                        st.info("Silhouette Score requires at least 2 non-noise clusters.")

                    # Add cluster labels to dataframe
                    df_clustered = df[features].copy()
                    df_clustered['Cluster'] = cluster_labels

                    # Display cluster summary
                    st.subheader("📊 Cluster Summary")
                    cluster_counts = df_clustered['Cluster'].value_counts().sort_index()
                    st.dataframe(cluster_counts.rename("Count"))
                    st.dataframe(df_clustered.groupby('Cluster')[features].mean())

                    # Visualizations
                    st.subheader("📈 Cluster Visualizations")
                    if len(features) >= 2:
                        # 2D plot
                        if len(features) == 2:
                            fig = px.scatter(df_clustered, x=features[0], y=features[1], color='Cluster',
                                             title=f"{clustering_method} Clusters (2D)",
                                             hover_data=features)
                            st.plotly_chart(fig, use_container_width=True)
                        # 3D plot
                        elif len(features) >= 3:
                            fig = px.scatter_3d(df_clustered, x=features[0], y=features[1], z=features[2],
                                                color='Cluster', title=f"{clustering_method} Clusters (3D)",
                                                hover_data=features)
                            st.plotly_chart(fig, use_container_width=True)
                        # For more than 3 features, use dimensionality reduction for visualization
                        else:
                            st.info("For more than 3 features, dimensionality reduction will be applied for visualization.")
                            dr_method = st.selectbox("Select DR method for visualization", ["PCA", "TSNE", "UMAP"])
                            n_comp_vis = 2 # Always visualize in 2D or 3D for clarity

                            X_reduced_vis, _ = app.dimensionality_reduction(X_scaled, dr_method, n_comp_vis)
                            df_reduced = pd.DataFrame(X_reduced_vis, columns=[f'{dr_method}1', f'{dr_method}2'])
                            df_reduced['Cluster'] = cluster_labels

                            fig = px.scatter(df_reduced, x=f'{dr_method}1', y=f'{dr_method}2', color='Cluster',
                                             title=f"{clustering_method} Clusters with {dr_method} (2D)",
                                             hover_data=['Cluster'])
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("Please select at least two features for effective visualization.")

                    # Download clustered data
                    st.markdown("---")
                    st.subheader("📥 Download Clustered Data")
                    csv = df_clustered.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Clustered Data CSV",
                        data=csv,
                        file_name=f"clustered_data_{clustering_method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                except Exception as e:
                    st.error(f"Error performing clustering: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
        else:
            st.info("Please select features to perform clustering.")

if __name__ == "__main__":
    main()