import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl
    import altair as alt
    import pyarrow as pa
    import plotly.express as px
    import marimo as mo
    from sklearn.cluster import KMeans
    return KMeans, alt, mo, pl, px


@app.cell
def _(pl):
    mall_customers_df = pl.read_csv('./mall-customer-segmentation/data/Mall_Customers.csv')
    return (mall_customers_df,)


@app.cell
def _(mall_customers_df):
    mc_df = mall_customers_df
    mc_df = mc_df.rename({"Annual Income (k$)": "AnnualIncomeK$"})
    mc_df = mc_df.rename({"Spending Score (1-100)": "SpendingScore1To100"})
    return (mc_df,)


@app.cell
def _(mc_df):
    mc_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 2D naive clustering""")
    return


@app.cell
def _(alt, mc_df):
    alt.Chart(mc_df).mark_circle().encode(
        x='AnnualIncomeK$',
        y='SpendingScore1To100'
    ).interactive()
    return


@app.cell
def _(KMeans, mc_df):
    kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto")
    labels = kmeans.fit_predict(mc_df.select(['AnnualIncomeK$', 'SpendingScore1To100']))
    return (labels,)


@app.cell
def _(labels, mc_df, pl):
    # Add cluster labels to the dataframe
    mc_with_clusters = mc_df.with_columns(
        pl.Series("ClusterLabel", labels)
    )
    return (mc_with_clusters,)


@app.cell
def _(alt, mc_with_clusters):
    # Create scatter plot colored by cluster
    alt.Chart(mc_with_clusters).mark_circle(size=60).encode(
        x=alt.X('AnnualIncomeK$:Q', title='Annual Income (k$)'),
        y=alt.Y('SpendingScore1To100:Q', title='Spending Score (1-100)'),
        color=alt.Color('ClusterLabel:N', 
                       scale=alt.Scale(scheme='category10'),
                       title='ClusterLabel'),
        tooltip=['CustomerID:N', 'AnnualIncomeK$:Q', 'SpendingScore1To100:Q', 'Age:Q', 'Gender:N', 'ClusterLabel:N']
    ).properties(
        title='Customer Segmentation - K-Means Clustering',
        width=500,
        height=400
    ).interactive()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        rf"""
    # 2D clustering - Elbow Method

    Run the KMeans N times and find the best K. In this context, "the best" K is the one where intertia gains sharply fall off.
    """
    )
    return


@app.cell
def _(KMeans):
    def elbow(df, n_clusters, features):
        # Elbow method to find optimal number of clusters
        k_values = range(1, n_clusters+1)
        inertias = []
        labels = []

        for k_ in k_values:
            kmeans_elbow = KMeans(n_clusters=k_, random_state=0, n_init="auto")
            kmeans_elbow.fit_predict(df.select(features))
            inertias.append(kmeans_elbow.inertia_)
            labels.append(kmeans_elbow.labels_)

        return k_values, labels, inertias
    return (elbow,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Based on the chart below, **K=5 is about as good as we can get**, which was obvious when looking at the scatter plot.""")
    return


@app.cell
def _(alt, elbow, mc_df, pl):
    k_values, labels_elbow, inertias_elbow = elbow(mc_df, 10, ['AnnualIncomeK$', 'SpendingScore1To100'])

    # Create dataframe for plotting
    elbow_df = pl.DataFrame({
        'K': list(k_values),
        'Inertia': inertias_elbow
    })

    # Create elbow plot
    alt.Chart(elbow_df).mark_line(point=True).encode(
        x=alt.X('K:O', title='Number of Clusters (K)'),
        y=alt.Y('Inertia:Q', title='Inertia (Within-cluster Sum of Squares)'),
        tooltip=['K:O', 'Inertia:Q']
    ).properties(
        title='Elbow Method for Optimal K',
        width=500,
        height=300
    )
    return k_values, labels_elbow


@app.cell(hide_code=True)
def _(alt, k_values, labels_elbow, mc_df, pl):
    # Create a long-format dataframe for faceting
    facet_data = []
    for i, k in enumerate(k_values):
        temp_df = mc_df.with_columns([
            pl.Series("K", [k] * len(mc_df)),
            pl.Series("ClusterLabel", labels_elbow[i])
        ])
        facet_data.append(temp_df)

    # Concatenate all dataframes
    facet_df = pl.concat(facet_data)

    # Create faceted scatter plot
    alt.Chart(facet_df).mark_circle(size=40).encode(
        x=alt.X('AnnualIncomeK$:Q', title='Annual Income (k$)'),
        y=alt.Y('SpendingScore1To100:Q', title='Spending Score (1-100)'),
        color=alt.Color('ClusterLabel:N', 
                       scale=alt.Scale(scheme='category10'),
                       title='Cluster'),
        facet=alt.Facet('K:O', columns=5, title='Number of Clusters (K)'),
        tooltip=['CustomerID:N', 'AnnualIncomeK$:Q', 'SpendingScore1To100:Q', 'ClusterLabel:N', 'K:O']
    ).properties(
        title='Customer Clustering Comparison Across Different K Values',
        width=180,
        height=150
    ).resolve_scale(
        color='independent'
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 3D clustering with Elbow method""")
    return


@app.cell(hide_code=True)
def _(mc_df, px):
    # Create 3D scatter plot
    fig_3d = px.scatter_3d(
        mc_df,
        x='Age',
        y='AnnualIncomeK$',
        z='SpendingScore1To100',
        color='Gender',
        symbol='Gender',
        hover_data=['CustomerID'],
        labels={
            'Age': 'Age (years)',
            'AnnualIncomeK$': 'Annual Income (k$)',
            'SpendingScore1To100': 'Spending Score (1-100)'
        },
        title='3D Customer Analysis: Age, Income, and Spending Score',
        color_discrete_sequence=['#FF6B6B', '#4ECDC4']
    )

    fig_3d.update_traces(marker=dict(size=5))
    fig_3d.update_layout(
        scene=dict(
            xaxis_title='Age (years)',
            yaxis_title='Annual Income (k$)',
            zaxis_title='Spending Score (1-100)'
        ),
        width=800,
        height=600
    )

    fig_3d
    return


@app.cell
def _(elbow, mc_df):
    kvals3d, labels3d, inertias3d = elbow(mc_df, 10, ['Age', 'AnnualIncomeK$', 'SpendingScore1To100'])
    return inertias3d, kvals3d, labels3d


@app.cell
def _(alt, inertias3d, kvals3d, pl):
    # Create dataframe for plotting
    elbow3d_df = pl.DataFrame({
        'K': list(kvals3d),
        'Inertia': inertias3d
    })

    # Create elbow plot
    alt.Chart(elbow3d_df).mark_line(point=True).encode(
        x=alt.X('K:O', title='Number of Clusters (K)'),
        y=alt.Y('Inertia:Q', title='Inertia'),
        tooltip=['K:O', 'Inertia:Q']
    ).properties(
        title='Elbow Method for Optimal K',
        width=500,
        height=300
    )
    return


@app.cell
def _(labels3d, mc_df, pl, px):
    # Get cluster labels for k=6 (index 5 since k_values starts from 1)
    k6_labels = labels3d[5]  # k=6 is at index 5

    # Add cluster labels to the dataframe for k=6
    mc_3d_clustered = mc_df.with_columns(
        pl.Series("Cluster_K6", k6_labels)
    )

    # Create 3D scatter plot colored by clusters
    fig_3d_clustered = px.scatter_3d(
        mc_3d_clustered,
        x='Age',
        y='AnnualIncomeK$',
        z='SpendingScore1To100',
        color='Cluster_K6',
        hover_data=['CustomerID', 'Gender'],
        labels={
            'Age': 'Age (years)',
            'AnnualIncomeK$': 'Annual Income (k$)',
            'SpendingScore1To100': 'Spending Score (1-100)',
            'Cluster_K6': 'Cluster (K=6)'
        },
        title='3D Customer Clustering: Age, Income, and Spending Score (K=6)',
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    fig_3d_clustered.update_traces(marker=dict(size=5))
    fig_3d_clustered.update_layout(
        scene=dict(
            xaxis_title='Age (years)',
            yaxis_title='Annual Income (k$)',
            zaxis_title='Spending Score (1-100)'
        ),
        width=1024,
        height=768
    )

    fig_3d_clustered
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# 4D clustering with encoded gender""")
    return


@app.cell
def _(alt, elbow, mc_df, pl):
    # First encode Gender as numerical for 4D clustering
    mc_encoded = mc_df.with_columns(
        pl.col("Gender").map_elements(lambda x: 1 if x == "Male" else 0, return_dtype=pl.Int8).alias("GenderCode")
    )

    # Perform 4D clustering using all available numerical features
    kvals4d, labels4d, inertias4d = elbow(mc_encoded, 10, ['Age', 'AnnualIncomeK$', 'SpendingScore1To100', 'GenderCode'])

    # Create elbow plot for 4D
    elbow4d_df = pl.DataFrame({
        'K': list(kvals4d),
        'Inertia': inertias4d
    })

    elbow_chart_4d = alt.Chart(elbow4d_df).mark_line(point=True).encode(
        x=alt.X('K:O', title='Number of Clusters (K)'),
        y=alt.Y('Inertia:Q', title='Inertia'),
        tooltip=['K:O', 'Inertia:Q']
    ).properties(
        title='4D Elbow Method: Age, Income, Spending Score, Gender',
        width=500,
        height=300
    )

    elbow_chart_4d
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Interpretation ...""")
    return


if __name__ == "__main__":
    app.run()
