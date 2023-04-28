import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn import linear_model
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
import shap
from sklearn import tree


def rename_df(df_, keyword="percent"):
    non_keyword_fields = {
        col: "_".join([col, "flg"])
        for col in df_.columns
        if col.lower().find(keyword) < 0
    }
    keyword_fields = {
        col: "_".join([col[: col.index(keyword)], "pct"])
        for col in df_.columns
        if col.lower().find(keyword) > 0
    }
    keyword_fields.update(non_keyword_fields)
    return df_.rename(columns=keyword_fields)


def load_dataset(url):
    df = (
        pd.read_csv(url, index_col="competitorname")
        # Downcasting integers
        .pipe(
            lambda df_: df_.astype(
                {col: "int8" for col in df_.select_dtypes("int").columns}
            )
        )
        # Downcasting floats
        .pipe(
            lambda df_: df_.astype(
                {col: "float32" for col in df_.select_dtypes("float").columns}
            )
        )
        # Renaming existing names to more explicit form
        .pipe(rename_df)
        # Defining new / existing features
        .assign(win_pct=lambda df_: df_.win_pct / 100)
    )

    return df


def obtain_names_based_on_rank(sr_: pd.Series, best: bool, n_rows: int = 10):
    return sr_.rank(ascending=best).nlargest(n_rows).index.to_list()


def create_comparative_analysis(df_: pd.DataFrame, y: str, samp_size=10):
    # Find indexes of top/bottom
    top = obtain_names_based_on_rank(df_.loc[:, y], True, samp_size)
    bot = obtain_names_based_on_rank(df_.loc[:, y], False, samp_size)

    df_tmp = df_

    # Define groups
    df_tmp["group"] = np.nan
    df_tmp.loc[top, "group"] = "best"
    df_tmp.loc[bot, "group"] = "worst"

    # Return output
    return (
        df_tmp.groupby(["group"])
        .agg(["mean"])
        .transpose()
        .reset_index(level=[1])
        .assign(diff=lambda df_: df_.best - df_.worst)
        .drop(columns="level_1")
        .rename_axis("properties")
    )


def univariate_analysis(df_: pd.DataFrame, y: float):
    df_ = df_.copy()

    int_tp = ["int8", "int16", "int32", "int64"]
    float_tp = ["float32", "float64"]

    # Binary properties EDA
    binary_properties = (
        df_.select_dtypes(int_tp)
        .describe()
        .loc["count":"std"]
        .transpose()
        .rename_axis("properties")
    )

    # Numeric properties EDA
    numeric_properties = (
        df_.select_dtypes(float_tp)
        .describe()
        .drop(["25%", "50%", "75%"], axis=0)
        .transpose()
        .rename_axis("properties")
    )

    # Get the comparative analysis of top 10 / top 3
    top_vs_bottom_3 = create_comparative_analysis(df_, y, samp_size=3)
    top_vs_bottom_10 = create_comparative_analysis(df_, y, samp_size=10)

    return {
        "top3_vs_bottom3": top_vs_bottom_3,
        "top10_vs_bottom10": top_vs_bottom_10,
        "numeric_properties": numeric_properties,
        "binary_properties": binary_properties,
    }


def visualize_cat_vars(
    df_: pd.DataFrame,
    y: str,
    x_label: str = "Win Percentage",
    y_label: str = "# Observations",
    legend_labs: str = ["Yes", "No"],
):
    cat_vars = df_.columns[df_.columns.str.contains("_flg") & (df_.columns != y)]

    n_cols = 3
    mod = np.min([len(cat_vars) % n_cols, 1])
    n_rows = (len(cat_vars) // n_cols) + mod

    fig = plt.figure(figsize=(16, 9))
    for pos, var_name in enumerate(cat_vars):
        ax = fig.add_subplot(n_rows, n_cols, pos + 1)
        plot = sns.histplot(x=y, data=df_, hue=var_name, ax=ax, bins=10)
        plot.set_title(var_name.removesuffix("_flg").capitalize(), fontsize=12)
        plot.set_xlabel(x_label, fontsize=12)
        plot.set_ylabel(y_label, fontsize=12)
        plot.tick_params(labelsize=12)
        plot.legend_.set_title("")
        plot.legend(labels=legend_labs, fontsize=12)
    plt.tight_layout()

    return plt.show()


def visualize_num_vars(df_: pd.DataFrame, y: str):
    num_vars = list(df_.columns[~df_.columns.str.contains("_flg") & (df_.columns != y)])
    x_labels = list(map(lambda x: x.removesuffix("_pct").capitalize(), num_vars))
    y_label = y.removesuffix("_pct").capitalize()

    n_cols = 3
    mod = np.min([len(num_vars) % n_cols, 1])
    n_rows = (len(num_vars) // n_cols) + mod

    fig = plt.figure(figsize=(16, 9))
    for pos, var_name in enumerate(num_vars):
        ax = fig.add_subplot(n_rows, n_cols, pos + 1)
        plot = sns.regplot(x=var_name, y="win_pct", data=df_)
        plot.set_title(" to ".join([x_labels[pos], y_label]), fontsize=12)
        plot.set_xlabel(x_labels[pos], fontsize=12)
        plot.set_ylabel(y_label, fontsize=12)
        plot.tick_params(labelsize=12)
    plt.tight_layout()

    return plt.show()


def correlated_pairs(df_: pd.DataFrame, y: str, threshold=0.7):
    c_m = df_.drop(columns=y).corr("pearson").reset_index()
    pairs = pd.melt(c_m, id_vars="index", value_vars=c_m.columns).rename(
        columns={"index": "var1", "variable": "var2", "value": "corr_coef"}
    )

    return pairs.loc[pairs["corr_coef"].abs().between(threshold, 1, "left")]


def correlation_matrix(df_: pd.DataFrame, y: str):
    corr = df_.drop(columns=y).corr("pearson")

    tick_lables = [x.removesuffix("_flg").replace("_pct", "") for x in corr.columns]
    f, ax = plt.subplots(figsize=(6, 6))
    mask = np.tril(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        mask=mask,
        cmap=cmap,
        cbar=False,
        linewidths=0.5,
        xticklabels=tick_lables,
        yticklabels=tick_lables,
        square=True,
    )

    return plt.show()


def linear_regression(x, y):
    model = linear_model.LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)

    # Metrics
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    # SM for statistical output
    x_wc = sm.add_constant(x)
    est = sm.OLS(y, x_wc).fit()

    return {
        "model_est": model,
        "mse": mse,
        "r2": r2,
        "model_fit": est.summary2().tables[0],
        "params_est": est.summary2().tables[1],
    }


def shapley_explainer(model, x, obs_to_explain):
    explainer = shap.Explainer(model.predict, x)
    shap_values = explainer(obs_to_explain)
    return shap_values


def visualize_shapley(shap_values, names_in):
    n_graphs = shap_values.shape[0]

    fig = plt.figure()
    for i in range(n_graphs):
        ax_i = fig.add_subplot(1, n_graphs, i + 1)
        shap.plots.waterfall(shap_values[i], max_display=14, show=False)
        plt.xlabel(names_in[i])
    plt.gcf().set_size_inches(20, 6)
    plt.tight_layout()
    plt.show()


def finding_optimal_depth(df_, x, y):
    clf = tree.DecisionTreeRegressor()
    path = clf.cost_complexity_pruning_path(df_[x], df_[y])
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    clfs = []

    for ccp_alpha in ccp_alphas:
        clf = tree.DecisionTreeRegressor(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(df_[x], df_[y])
        clfs.append(clf)

    tree_depths = [clf.tree_.max_depth for clf in clfs]

    sc = [np.mean(abs(clf.predict(df_[x]) - df_[y])) for clf in clfs]

    plt.figure(figsize=(10, 6))
    plt.plot(sc[:-1], tree_depths[:-1])
    plt.xlabel("Mean absolute error")
    plt.ylabel("total depth")
    plt.show()

def regression_tree(df_: pd.DataFrame, x: list, y: str, depth = 2, seed=1991):
    criterion = "squared_error"
    splitter = "best"
    mdepth = depth
    min_samples_leaf = 0.10
    random_state = seed

    model = tree.DecisionTreeRegressor(
        criterion=criterion,
        splitter=splitter,
        max_depth=mdepth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )

    clf = model.fit(df_[x], df_[y])

    pred_label = model.predict(df_[x])

    score_te = model.score(df_[x], df_[y])
    absolute_error = abs(pred_label - df_[y])
    MAE = round(np.mean(absolute_error), 2)

    importances = list(clf.feature_importances_)
    feature_importances = [
        (feature, round(importance, 2)) for feature, importance in zip(x, importances)
    ]
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

    print("*************** Tree Summary ***************")
    print("Tree Depth: ", clf.tree_.max_depth)
    print("No. of leaves: ", clf.tree_.n_leaves)
    print("No. of features: ", clf.n_features_in_)
    print("Accuracy Score: ", round(score_te, 2))
    print("Mean Absolute Error:", MAE)
    print("*************** Variable importance ***************")
    [
        print("Variable: {:20} Importance: {}".format(*pair))
        for pair in feature_importances
    ]
    print("--------------------------------------------------------")
    print("")

    fig, ax = plt.subplots(figsize=(25, 25))
    tree.plot_tree(clf, ax=ax, feature_names=x)
    plt.show()

    return model
