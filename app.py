import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import base64
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Product Recommendation Engine",
    page_icon="🛒",
    layout="wide"
)

# Define CSS styles
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #42A5F5;
        margin-bottom: 1rem;
    }
    .info-text {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("<h1 class='main-header'>Product Recommendation Engine</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='info-text'>Upload your CSV data to generate personalized product recommendations using machine learning!</div>",
    unsafe_allow_html=True)

# Navigation sidebar
st.sidebar.image("https://www.svgrepo.com/show/374994/recommendation.svg", width=100)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Upload", "Recommendation Settings", "Results", "About"])

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None
if 'user_item_matrix' not in st.session_state:
    st.session_state.user_item_matrix = None
if 'similarity_matrix' not in st.session_state:
    st.session_state.similarity_matrix = None
if 'method' not in st.session_state:
    st.session_state.method = "Collaborative Filtering"
if 'top_n' not in st.session_state:
    st.session_state.top_n = 5
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'user_column' not in st.session_state:
    st.session_state.user_column = None
if 'item_column' not in st.session_state:
    st.session_state.item_column = None
if 'rating_column' not in st.session_state:
    st.session_state.rating_column = None


# Functions for recommendation algorithms
def create_user_item_matrix(df, user_col, item_col, rating_col=None):
    """Creates a user-item matrix from transaction data."""
    if rating_col:
        user_item = df.pivot_table(
            index=user_col,
            columns=item_col,
            values=rating_col,
            aggfunc='mean',
            fill_value=0
        )
    else:
        # If no rating column, use count of interactions
        user_item = df.groupby([user_col, item_col]).size().unstack(fill_value=0)

    return user_item


def collaborative_filtering(user_item_matrix, target_user=None, k=5):
    """
    Implémente le filtrage collaboratif basé sur les items.

    Args:
        user_item_matrix: Matrice utilisateur-item des interactions
        target_user: Utilisateur spécifique pour lequel générer des recommandations (tous les utilisateurs si None)
        k: Nombre de recommandations à générer par utilisateur

    Returns:
        Dictionnaire des recommandations par utilisateur
    """
    # Calcul de la similarité entre items
    item_similarity = cosine_similarity(user_item_matrix.T)
    item_similarity_df = pd.DataFrame(
        item_similarity,
        index=user_item_matrix.columns,
        columns=user_item_matrix.columns
    )

    # Sauvegarde de la matrice de similarité dans l'état de la session
    st.session_state.similarity_matrix = item_similarity_df

    # Déterminer quels utilisateurs traiter
    users_to_process = [target_user] if target_user else user_item_matrix.index
    recommendations = {}

    for user in users_to_process:
        if user not in user_item_matrix.index:
            continue

        # Récupérer les items avec lesquels l'utilisateur a interagi
        user_items = user_item_matrix.loc[user]
        interacted_items = user_items[user_items > 0].index.tolist()

        # Trouver les items avec lesquels l'utilisateur n'a pas interagi
        non_interacted_items = user_item_matrix.columns.difference(interacted_items)
        recommendations_for_user = {}

        for item in non_interacted_items:
            # Obtenir les items similaires avec lesquels l'utilisateur a interagi
            similar_items = item_similarity_df[item].loc[interacted_items]
            item_scores = similar_items.values

            # Récupérer les évaluations de l'utilisateur pour les items avec lesquels il a interagi
            user_ratings = user_item_matrix.loc[user, interacted_items].values

            # Calculer la note prédite
            if len(item_scores) > 0:
                # Éviter la division par zéro
                sum_abs_scores = np.sum(np.abs(item_scores))

                if sum_abs_scores > 0:
                    # Calcul pondéré basé sur la similarité
                    predicted_rating = np.sum(item_scores * user_ratings) / sum_abs_scores
                    recommendations_for_user[item] = predicted_rating
                else:
                    # Si la somme est nulle, utiliser la moyenne des évaluations de l'utilisateur
                    avg_rating = np.mean(user_ratings) if len(user_ratings) > 0 else 0.0
                    recommendations_for_user[item] = avg_rating

        # Trier et obtenir les k meilleures recommandations
        sorted_recommendations = sorted(recommendations_for_user.items(), key=lambda x: x[1], reverse=True)
        recommendations[user] = sorted_recommendations[:k]

    return recommendations


def content_based_filtering(df, target_column, item_col, user_col, k=5):
    """
    Implémente le filtrage basé sur le contenu en utilisant les caractéristiques des items.

    Args:
        df: DataFrame contenant les données utilisateur-item
        target_column: Colonne contenant les caractéristiques des items
        item_col: Nom de la colonne identifiant les items
        user_col: Nom de la colonne identifiant les utilisateurs
        k: Nombre de recommandations à générer par utilisateur

    Returns:
        Dictionnaire des recommandations par utilisateur
    """
    # Créer une matrice de caractéristiques pour les items
    target_dummies = pd.get_dummies(df[target_column])

    # Joindre les caractéristiques des items
    item_features = df[[item_col]].join(target_dummies)

    # Agréger par item (pour gérer les doublons)
    item_features = item_features.groupby(item_col).mean()

    # Normaliser les caractéristiques
    scaler = MinMaxScaler()
    item_features_scaled = pd.DataFrame(
        scaler.fit_transform(item_features),
        index=item_features.index,
        columns=item_features.columns
    )

    # Calculer la similarité entre items
    item_similarity = cosine_similarity(item_features_scaled)
    item_similarity_df = pd.DataFrame(
        item_similarity,
        index=item_features_scaled.index,
        columns=item_features_scaled.index
    )

    # Sauvegarder la matrice de similarité
    st.session_state.similarity_matrix = item_similarity_df

    # Obtenir l'historique des interactions utilisateur-item
    user_items = df.groupby(user_col)[item_col].apply(list).to_dict()

    # Générer des recommandations pour chaque utilisateur
    recommendations = {}

    for user, items in user_items.items():
        # Obtenir les items uniques avec lesquels l'utilisateur a interagi
        unique_items = list(set(items))

        if not unique_items:
            continue

        # Trouver les items avec lesquels l'utilisateur n'a pas interagi
        all_items = item_features_scaled.index.tolist()
        non_interacted_items = list(set(all_items) - set(unique_items))

        # Calculer les scores pour les items non consultés
        item_scores = {}

        for non_item in non_interacted_items:
            if non_item not in item_similarity_df.index:
                continue

            # Calculer la similarité moyenne avec les items de l'utilisateur
            similarities = []
            for user_item in unique_items:
                if user_item in item_similarity_df.columns:
                    similarities.append(item_similarity_df.loc[non_item, user_item])

            # Attribuer un score basé sur la similarité moyenne
            if similarities:
                # Utiliser une moyenne pondérée pour favoriser les fortes similarités
                weights = np.array(
                    similarities) ** 2  # Mettre au carré pour donner plus de poids aux fortes similarités
                weighted_sum = np.sum(weights * np.array(similarities))
                sum_weights = np.sum(weights)

                if sum_weights > 0:
                    item_scores[non_item] = weighted_sum / sum_weights
                else:
                    item_scores[non_item] = np.mean(similarities)

        # Trier et obtenir les k meilleures recommandations
        sorted_scores = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        recommendations[user] = sorted_scores[:k]

    return recommendations

def matrix_factorization(user_item_matrix, k=5, components=10):
    """Implements matrix factorization for recommendations."""
    # Apply SVD for matrix factorization
    svd = TruncatedSVD(n_components=min(components, min(user_item_matrix.shape) - 1))
    latent_matrix = svd.fit_transform(user_item_matrix)

    # Reconstruct the matrix
    reconstructed = latent_matrix @ svd.components_
    reconstructed_df = pd.DataFrame(
        reconstructed,
        index=user_item_matrix.index,
        columns=user_item_matrix.columns
    )

    # Generate recommendations for each user
    recommendations = {}

    for user in user_item_matrix.index:
        # Get items the user has already interacted with
        user_items = user_item_matrix.loc[user]
        user_items = user_items[user_items > 0].index.tolist()

        # Get predicted ratings for items the user hasn't interacted with
        non_interacted_items = user_item_matrix.columns.difference(user_items)
        predicted_ratings = {item: reconstructed_df.loc[user, item] for item in non_interacted_items}

        # Sort and get top k recommendations
        sorted_ratings = sorted(predicted_ratings.items(), key=lambda x: x[1], reverse=True)
        recommendations[user] = sorted_ratings[:k]

    return recommendations


def visualize_recommendations(recommendations, user=None):
    """Generate visualization for recommendations."""
    if user:
        if user in recommendations:
            user_recs = recommendations[user]
            items = [rec[0] for rec in user_recs]
            scores = [rec[1] for rec in user_recs]

            df = pd.DataFrame({'Item': items, 'Score': scores})

            fig = px.bar(
                df,
                x='Item',
                y='Score',
                title=f'Top Recommendations for {user}',
                labels={'Score': 'Recommendation Score', 'Item': 'Product'},
                color='Score',
                color_continuous_scale='blues'
            )

            return fig
    else:
        # Create a summary of top recommended items across all users
        all_recs = []
        for user, recs in recommendations.items():
            for item, score in recs:
                all_recs.append({'User': user, 'Item': item, 'Score': score})

        df = pd.DataFrame(all_recs)

        # Get the most commonly recommended items
        item_counts = df['Item'].value_counts().reset_index()
        item_counts.columns = ['Item', 'Count']
        item_counts = item_counts.head(10)

        fig = px.bar(
            item_counts,
            x='Item',
            y='Count',
            title='Most Frequently Recommended Items',
            labels={'Count': 'Recommendation Frequency', 'Item': 'Product'},
            color='Count',
            color_continuous_scale='blues'
        )

        return fig


def plot_similarity_heatmap(similarity_matrix, top_n=20):
    """Plot a heatmap of the item similarity matrix."""
    # Get the top N items with the most interactions
    top_items = similarity_matrix.sum().sort_values(ascending=False).head(top_n).index.tolist()

    # Filter the similarity matrix to only include those items
    filtered_matrix = similarity_matrix.loc[top_items, top_items]

    # Create heatmap using matplotlib
    plt.figure(figsize=(10, 8))
    sns.heatmap(filtered_matrix, annot=False, cmap='viridis')
    plt.title(f'Item Similarity Heatmap (Top {top_n} Items)')
    plt.tight_layout()

    # Convert to Streamlit-compatible format
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return buf


def generate_report(df, recommendations, method):
    """Generate a report of recommendations."""
    # Create header
    report = "# Product Recommendation Report\n\n"
    report += f"## Method: {method}\n\n"

    # Data summary
    report += "### Data Summary\n\n"
    report += f"* Total records: {len(df)}\n"
    report += f"* Unique users: {df[st.session_state.user_column].nunique()}\n"
    report += f"* Unique items: {df[st.session_state.item_column].nunique()}\n\n"

    # Sample recommendations
    report += "### Sample Recommendations\n\n"

    sample_users = list(recommendations.keys())[:5]
    for user in sample_users:
        report += f"#### User: {user}\n\n"
        report += "| Product | Score |\n"
        report += "| ------- | ----- |\n"

        for item, score in recommendations[user]:
            report += f"| {item} | {score:.4f} |\n"

        report += "\n"

    # Most recommended products
    all_recs = []
    for user, recs in recommendations.items():
        for item, score in recs:
            all_recs.append({'User': user, 'Item': item, 'Score': score})

    rec_df = pd.DataFrame(all_recs)
    top_items = rec_df['Item'].value_counts().head(10)

    report += "### Most Frequently Recommended Products\n\n"
    report += "| Product | Recommendation Count |\n"
    report += "| ------- | ------------------- |\n"

    for item, count in top_items.items():
        report += f"| {item} | {count} |\n"

    return report


# Home page
if page == "Home":
    st.markdown("<h2 class='sub-header'>Welcome to the Product Recommendation Engine!</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        This tool helps you generate personalized product recommendations using machine learning algorithms based on your CSV data.

        ### Features:
        - Upload your own product and user interaction data
        - Choose from multiple recommendation algorithms
        - Visualize recommendation results
        - Download recommendations for implementation

        ### How to use:
        1. Go to the **Data Upload** page to upload your CSV file
        2. Configure your recommendation settings
        3. Generate and view personalized recommendations
        4. Download the results for implementation
        """)

        st.markdown("<div class='success-box'>Start by uploading your data in the <b>Data Upload</b> section!</div>",
                    unsafe_allow_html=True)

    with col2:
        st.image("https://www.svgrepo.com/show/531000/recommendation-engine.svg", width=200)

        if st.session_state.data is not None:
            st.success(
                f"✅ Data loaded: {st.session_state.data.shape[0]} rows, {st.session_state.data.shape[1]} columns")

        if st.session_state.recommendations is not None:
            st.success(f"✅ Recommendations generated for {len(st.session_state.recommendations)} users")

# Data Upload page
elif page == "Data Upload":
    st.markdown("<h2 class='sub-header'>Upload and Configure Your Data</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        Upload a CSV file containing your product and user interaction data. The file should include:

        - User identifiers (e.g., user_id, customer_id)
        - Product identifiers (e.g., product_id, item_id)
        - Optional: ratings or interaction strength
        - Optional: product categories or features
        """)

        uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.data = df
                st.success(f"File loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

                # Show data preview
                st.subheader("Data Preview")
                st.dataframe(df.head(5))

                # Column selection
                st.subheader("Column Configuration")

                col_a, col_b = st.columns(2)

                with col_a:
                    user_col = st.selectbox(
                        "Select user identifier column:",
                        options=df.columns.tolist(),
                        key="user_col_select"
                    )

                    item_col = st.selectbox(
                        "Select product/item identifier column:",
                        options=df.columns.tolist(),
                        key="item_col_select"
                    )

                with col_b:
                    rating_col = st.selectbox(
                        "Select rating column (optional):",
                        options=['None'] + df.columns.tolist(),
                        key="rating_col_select"
                    )

                    target_col = st.selectbox(
                        "Select target feature column (for content-based):",
                        options=['None'] + df.columns.tolist(),
                        key="target_col_select"
                    )

                # Save column selections
                if st.button("Save Column Configuration"):
                    st.session_state.user_column = user_col
                    st.session_state.item_column = item_col
                    st.session_state.rating_column = None if rating_col == 'None' else rating_col
                    st.session_state.target_column = None if target_col == 'None' else target_col

                    st.markdown(
                        "<div class='success-box'>Column configuration saved! Now go to the <b>Recommendation Settings</b> page to configure the algorithm.</div>",
                        unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error loading file: {str(e)}")

    with col2:
        st.markdown("### Data Requirements")
        st.info("""
        **Minimum Requirements:**
        - User identifiers
        - Product identifiers
        - At least one interaction per user

        **For Better Results:**
        - Rating values (e.g., 1-5 stars)
        - Product categories or features
        - Timestamps (for recency)
        """)

        if st.session_state.data is not None:
            st.markdown("### Data Statistics")
            df = st.session_state.data

            stats = {
                "Total Records": df.shape[0],
                "Unique Users": df[
                    st.session_state.user_column].nunique() if st.session_state.user_column else "Not selected",
                "Unique Products": df[
                    st.session_state.item_column].nunique() if st.session_state.item_column else "Not selected",
            }

            for key, value in stats.items():
                st.metric(key, value)

# Recommendation Settings page
elif page == "Recommendation Settings":
    st.markdown("<h2 class='sub-header'>Configure Recommendation Settings</h2>", unsafe_allow_html=True)

    if st.session_state.data is None:
        st.warning("Please upload data first in the Data Upload section!")
        st.stop()

    if not st.session_state.user_column or not st.session_state.item_column:
        st.warning("Please configure column mappings in the Data Upload section!")
        st.stop()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Algorithm Selection")

        algorithm = st.radio(
            "Select recommendation algorithm:",
            ["Collaborative Filtering", "Content-Based Filtering", "Matrix Factorization"],
            index=0 if st.session_state.method == "Collaborative Filtering" else
            1 if st.session_state.method == "Content-Based Filtering" else 2
        )

        # Algorithm-specific parameters
        st.markdown("### Algorithm Parameters")

        if algorithm == "Collaborative Filtering":
            st.markdown("""
            **Collaborative filtering** recommends products based on similar users' preferences.
            It works best when you have a good amount of user-item interaction data.
            """)

            st.session_state.top_n = st.slider(
                "Number of recommendations per user:",
                min_value=1,
                max_value=20,
                value=st.session_state.top_n
            )

        elif algorithm == "Content-Based Filtering":
            if not st.session_state.target_column:
                st.warning(
                    "Content-based filtering requires a target feature column. Please select one in the Data Upload section.")

            st.markdown("""
            **Content-based filtering** recommends products similar to what a user has liked before,
            based on product features or categories.
            """)

            st.session_state.top_n = st.slider(
                "Number of recommendations per user:",
                min_value=1,
                max_value=20,
                value=st.session_state.top_n
            )

        elif algorithm == "Matrix Factorization":
            st.markdown("""
            **Matrix Factorization** uses techniques like SVD to discover latent factors
            that explain observed preferences. Works well for sparse data.
            """)

            st.session_state.top_n = st.slider(
                "Number of recommendations per user:",
                min_value=1,
                max_value=20,
                value=st.session_state.top_n
            )

            components = st.slider(
                "Number of latent factors:",
                min_value=2,
                max_value=50,
                value=10
            )

        # Generate recommendations
        if st.button("Generate Recommendations"):
            with st.spinner("Generating recommendations..."):
                try:
                    df = st.session_state.data

                    # Create user-item matrix if needed
                    if algorithm in ["Collaborative Filtering", "Matrix Factorization"]:
                        user_item = create_user_item_matrix(
                            df,
                            st.session_state.user_column,
                            st.session_state.item_column,
                            st.session_state.rating_column
                        )
                        st.session_state.user_item_matrix = user_item

                    # Generate recommendations based on selected algorithm
                    if algorithm == "Collaborative Filtering":
                        recommendations = collaborative_filtering(
                            st.session_state.user_item_matrix,
                            k=st.session_state.top_n
                        )

                    elif algorithm == "Content-Based Filtering":
                        if not st.session_state.target_column:
                            st.error("Content-based filtering requires a target feature column")
                            st.stop()

                        recommendations = content_based_filtering(
                            df,
                            st.session_state.target_column,
                            st.session_state.item_column,
                            st.session_state.user_column,
                            k=st.session_state.top_n
                        )

                    elif algorithm == "Matrix Factorization":
                        recommendations = matrix_factorization(
                            st.session_state.user_item_matrix,
                            k=st.session_state.top_n,
                            components=components
                        )

                    # Save recommendations and method to session state
                    st.session_state.recommendations = recommendations
                    st.session_state.method = algorithm

                    st.success(f"Successfully generated recommendations for {len(recommendations)} users!")
                    st.markdown(
                        "<div class='success-box'>Recommendations generated! Go to the <b>Results</b> page to view them.</div>",
                        unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")

    with col2:
        st.markdown("### Algorithm Comparison")

        st.info("""
        **Collaborative Filtering**
        - Works well with lots of user interaction data
        - Doesn't need product features
        - Can discover unexpected recommendations

        **Content-Based Filtering**
        - Works with limited user data
        - Requires product features/categories
        - Good for new products with few interactions

        **Matrix Factorization**
        - Balances collaborative and content approaches
        - Works well for sparse datasets
        - Can find latent patterns in the data
        """)

        # Display data readiness
        st.markdown("### Data Readiness")

        df = st.session_state.data
        readiness = {}

        if st.session_state.user_column and st.session_state.item_column:
            users = df[st.session_state.user_column].nunique()
            items = df[st.session_state.item_column].nunique()

            readiness["Collaborative Filtering"] = "✅ Good" if users > 10 and items > 10 else "⚠️ Limited data"

            if st.session_state.target_column:
                features = df[st.session_state.target_column].nunique()
                readiness["Content-Based Filtering"] = "✅ Good" if features > 2 else "⚠️ Limited features"
            else:
                readiness["Content-Based Filtering"] = "❌ Missing feature column"

            density = df.shape[0] / (users * items)
            readiness["Matrix Factorization"] = "✅ Good" if density > 0.01 else "⚠️ Very sparse data"

        for algo, status in readiness.items():
            st.text(f"{algo}: {status}")

# Results page
elif page == "Results":
    st.markdown("<h2 class='sub-header'>Recommendation Results</h2>", unsafe_allow_html=True)

    if st.session_state.recommendations is None:
        st.warning("No recommendations generated yet! Please go to the Recommendation Settings page.")
        st.stop()

    # Display recommendations
    recommendations = st.session_state.recommendations
    method = st.session_state.method

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"### Recommendations Generated Using: {method}")

        # User filter
        df = st.session_state.data
        users = sorted(list(recommendations.keys()))

        selected_user = st.selectbox(
            "Select user to view recommendations:",
            options=users
        )

        if selected_user:
            st.subheader(f"Top {st.session_state.top_n} Recommendations for User: {selected_user}")

            user_recs = recommendations[selected_user]

            # Display recommendations as a table
            rec_df = pd.DataFrame(user_recs, columns=['Product', 'Score'])
            st.dataframe(rec_df)

            # Visualization
            fig = visualize_recommendations(recommendations, selected_user)
            st.plotly_chart(fig, use_container_width=True)

            # Show user's past interactions
            if st.session_state.user_item_matrix is not None:
                st.subheader("User's Past Interactions")

                user_data = df[df[st.session_state.user_column] == selected_user]
                user_items = user_data[st.session_state.item_column].value_counts()

                if len(user_items) > 0:
                    user_items_df = pd.DataFrame({
                        'Product': user_items.index,
                        'Interactions': user_items.values
                    })

                    st.dataframe(user_items_df)

        # Overall recommendations analysis
        st.markdown("### Overall Recommendation Analysis")

        overall_fig = visualize_recommendations(recommendations)
        st.plotly_chart(overall_fig, use_container_width=True)

        # Item similarity visualization
        if st.session_state.similarity_matrix is not None:
            st.subheader("Item Similarity Analysis")

            similarity_img = plot_similarity_heatmap(st.session_state.similarity_matrix)
            st.image(similarity_img)

    with col2:
        st.markdown("### Recommendation Stats")

        total_users = len(recommendations)
        total_items = len(set([item for user_recs in recommendations.values() for item, _ in user_recs]))

        st.metric("Total Users with Recommendations", total_users)
        st.metric("Unique Recommended Products", total_items)

        # Download results
        st.markdown("### Download Results")

        # Export recommendations as CSV
        all_recs = []
        for user, recs in recommendations.items():
            for rank, (item, score) in enumerate(recs, 1):
                all_recs.append({
                    'User': user,
                    'Product': item,
                    'Score': score,
                    'Rank': rank
                })

        recs_df = pd.DataFrame(all_recs)

        csv = recs_df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="product_recommendations.csv">Download Recommendations (CSV)</a>'
        st.markdown(href, unsafe_allow_html=True)

        # Generate and download report
        report = generate_report(st.session_state.data, recommendations, method)

        b64_report = base64.b64encode(report.encode()).decode()
        href_report = f'<a href="data:text/markdown;base64,{b64_report}" download="recommendation_report.md">Download Analysis Report (Markdown)</a>'
        st.markdown(href_report, unsafe_allow_html=True)

        # Clear results button
        if st.button("Clear Results and Start Over"):
            st.session_state.recommendations = None
            st.session_state.user_item_matrix = None
            st.session_state.similarity_matrix = None
            st.experimental_rerun()

# About page
else:
    st.markdown("<h2 class='sub-header'>About This Tool</h2>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Product Recommendation Engine

        This Streamlit application helps businesses leverage machine learning to generate personalized product recommendations based on user interaction data. The tool provides multiple recommendation algorithms to fit different business needs and data types.

        ### How It Works

        1. **Data Collection**: Upload your CSV file containing user-product interaction data
        2. **Data Processing**: Configure columns and prepare data for recommendation algorithms
        3. **Algorithm Selection**: Choose from collaborative filtering, content-based filtering, or matrix factorization
        4. **Recommendation Generation**: The system analyzes patterns and generates personalized recommendations
        5. **Results Visualization**: View and analyze the recommendations with interactive visualizations

        ### Recommendation Algorithms

        **Collaborative Filtering**: Recommends products based on similar users' preferences and behaviors. It identifies patterns in user interactions and finds users with similar tastes to make recommendations.

        **Content-Based Filtering**: Recommends products similar to what a user has liked before, based on product attributes or categories. It focuses on product features rather than user behavior.

        **Matrix Factorization**: Uses dimensionality reduction techniques to uncover latent factors that explain observed preferences. It's effective for sparse datasets and can find hidden patterns.

        ### Use Cases

        - E-commerce personalization
        - Product cross-selling and upselling
        - Customer retention strategies
        - Inventory management optimization
        - Marketing campaign targeting
        """)

    with col2:
        st.image("https://www.svgrepo.com/show/374994/recommendation.svg", width=150)

        st.markdown("### Technical Details")
        st.info("""
        - Built with Streamlit and Python
        - Uses scikit-learn for ML algorithms
        - Implements cosine similarity metrics
        - Supports SVD for matrix factorization
        - Visualizations with Plotly and Matplotlib
        """)

        st.markdown("### Tips for Better Results")
        st.success("""
        - Include as much user-item interaction data as possible
        - Add product categories or features for content-based filtering
        - Clean and preprocess your data for best results
        - Experiment with different algorithms for your specific use case
        """)

        st.markdown("### Need Help?")
        st.markdown("""
        Having trouble with the tool? Here are some resources:

        - [Understanding Recommendation Systems](https://developers.google.com/machine-learning/recommendation)
        - [Streamlit Documentation](https://docs.streamlit.io)
        - [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
        """)

# Add a footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Product Recommendation Engine | Built with Streamlit and ❤️</p>",
            unsafe_allow_html=True)

# Add explanatory tooltips
with st.sidebar:
    st.markdown("### Help")
    with st.expander("How to prepare your data"):
        st.markdown("""
        Your CSV file should contain at minimum:

        1. **User identifiers**: A column that uniquely identifies each user
        2. **Product identifiers**: A column that uniquely identifies each product
        3. **Interactions**: These could be:
           - Binary (purchased/not purchased)
           - Rating values (e.g., 1-5 stars)
           - View counts or interaction counts

        For content-based filtering, you'll also need:
        4. **Product features**: Categories, tags, or attributes of products

        Example CSV format:
        ```
        user_id,product_id,rating,category
        user1,product1,5,electronics
        user1,product2,3,clothing
        user2,product1,4,electronics
        ```
        """)

    with st.expander("Understanding the algorithms"):
        st.markdown("""
        **Collaborative Filtering**
        - Works by finding similar users or items
        - "Users who liked this also liked..."
        - Doesn't require product features
        - Cold start problem for new users/items

        **Content-Based Filtering**
        - Based on product attributes or features
        - "Because you liked products with these features..."
        - Works well for new products
        - Requires good product metadata

        **Matrix Factorization**
        - Decomposes user-item matrix to find latent factors
        - Combines strengths of both approaches
        - Works well with sparse data
        - Can discover hidden patterns
        """)

    with st.expander("Interpreting results"):
        st.markdown("""
        **Recommendation Score**: Higher scores indicate stronger recommendations. The score represents:

        - In collaborative filtering: Similarity between items based on user interactions
        - In content-based filtering: Similarity between item features
        - In matrix factorization: Predicted rating or preference strength

        **Item Similarity Heatmap**: Shows how similar products are to each other. Darker colors indicate stronger similarity.

        **Most Frequently Recommended Items**: Products that appear in many users' recommendation lists, suggesting they have broad appeal.
        """)

# Add a demo mode
if page == "Data Upload":
    with st.expander("Don't have data? Try the demo"):
        if st.button("Load Demo Data"):
            # Create demo data
            np.random.seed(42)
            n_users = 100
            n_products = 50
            n_categories = 5
            n_interactions = 2000

            users = [f"user_{i}" for i in range(1, n_users + 1)]
            products = [f"product_{i}" for i in range(1, n_products + 1)]
            categories = [f"category_{i}" for i in range(1, n_categories + 1)]

            # Generate random interactions
            demo_data = {
                'user_id': np.random.choice(users, n_interactions),
                'product_id': np.random.choice(products, n_interactions),
                'rating': np.random.randint(1, 6, n_interactions),
                'category': np.random.choice(categories, n_interactions)
            }

            demo_df = pd.DataFrame(demo_data)

            # Save to session state
            st.session_state.data = demo_df
            st.session_state.user_column = 'user_id'
            st.session_state.item_column = 'product_id'
            st.session_state.rating_column = 'rating'
            st.session_state.target_column = 'category'

            st.success("Demo data loaded! You can now proceed to the Recommendation Settings page.")
            st.dataframe(demo_df.head())

# Add advanced options for algorithms
if page == "Recommendation Settings":
    with st.expander("Advanced Algorithm Settings"):
        if st.session_state.method == "Collaborative Filtering":
            st.markdown("### Collaborative Filtering Settings")
            similarity_metric = st.selectbox(
                "Similarity Metric:",
                ["Cosine Similarity", "Pearson Correlation"],
                index=0
            )

            min_interactions = st.slider(
                "Minimum interactions per user:",
                min_value=1,
                max_value=10,
                value=1
            )

        elif st.session_state.method == "Content-Based Filtering":
            st.markdown("### Content-Based Filtering Settings")
            feature_weight = st.slider(
                "Feature importance weight:",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1
            )

        elif st.session_state.method == "Matrix Factorization":
            st.markdown("### Matrix Factorization Settings")
            learning_rate = st.slider(
                "Learning rate:",
                min_value=0.001,
                max_value=0.1,
                value=0.01,
                step=0.001,
                format="%.3f"
            )

            regularization = st.slider(
                "Regularization parameter:",
                min_value=0.001,
                max_value=0.1,
                value=0.02,
                step=0.001,
                format="%.3f"
            )

# Add batch processing option
if page == "Results" and st.session_state.recommendations is not None:
    with st.expander("Batch Processing"):
        st.markdown("### Generate Recommendations for Multiple Users")

        # Multi-select for users
        user_list = sorted(list(st.session_state.recommendations.keys()))
        selected_users = st.multiselect(
            "Select users for batch processing:",
            options=user_list,
            default=user_list[:5] if len(user_list) >= 5 else user_list
        )

        if selected_users and st.button("Generate Batch Report"):
            batch_results = {user: st.session_state.recommendations[user] for user in selected_users}

            # Create batch dataframe
            batch_data = []
            for user, recs in batch_results.items():
                for rank, (item, score) in enumerate(recs, 1):
                    batch_data.append({
                        'User': user,
                        'Product': item,
                        'Score': score,
                        'Rank': rank
                    })

            batch_df = pd.DataFrame(batch_data)

            st.dataframe(batch_df)

            # Download batch results
            csv = batch_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="batch_recommendations.csv">Download Batch Results (CSV)</a>'
            st.markdown(href, unsafe_allow_html=True)
