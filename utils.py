from datetime import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import plotly.graph_objects as go
import seaborn as sns

dictionnary = {
    'fashion_clothing_accessories': ['fashio', 'luggage'],
    'health_beauty': ['health', 'beauty', 'perfum', 'health_beauty'],
    'toys_baby': ['toy', 'baby', 'diaper'],
    'books_cds_media': ['book', 'cd', 'dvd', 'media'],
    'groceries_food_drink': ['grocer', 'food', 'drink'],
    'technology': ['phon', 'compu', 'tablet', 'electro', 'consol'],
    'home_furniture': ['home', 'furnitur', 'garden', 'bath', 'house', 'applianc'],
    'flowers_gifts': ['flow', 'gift', 'stuff'],
    'sport': ['sports_leisure', 'fashion_sport']
}


def renameCategorieType(value):
    # fonction qui simplifie les category par rapport
    # au dictionnay
    f = False
    for key in dictionnary:
        for val in dictionnary[key]:
            if (value.find(val) != -1):
                value = key
                f = True
    if (f == False):
        value = 'other'
    return value


def convertToDate(el):
    return pd.to_datetime(el)


def getDays(date):
    return date.days


def getMinMaxOrderInterval():
    # get min max date order
    path = "archive"
    orders = pd.read_csv(os.path.join(path, "olist_orders_dataset.csv"))
    orders['order_purchase_timestamp'] = orders['order_purchase_timestamp'].apply(convertToDate)
    return orders['order_purchase_timestamp'].min(), orders['order_purchase_timestamp'].max()


def getDatabyPeriod(start_time, stop_time):
    # fonction qui renvoie les clients en fonction des intervales d achats
    # Start_time end_time
    # Chemin d'accès aux données
    path = "archive"

    # Infos sur la géolocalisation
    geoloc = pd.read_csv(os.path.join(path, "olist_geolocation_dataset.csv"))
    # Infos sur les payments
    payments = pd.read_csv(os.path.join(path, "olist_order_payments_dataset.csv"))
    # Infos sur les achats
    orders = pd.read_csv(os.path.join(path, "olist_orders_dataset.csv"))
    # Infos sur l'évaluation des produits
    reviews = pd.read_csv(os.path.join(path, "olist_order_reviews_dataset.csv"))
    # Infos sur les items
    items = pd.read_csv(os.path.join(path, "olist_order_items_dataset.csv"))
    # Infos sur les produits
    products = pd.read_csv(os.path.join(path, "olist_products_dataset.csv"))
    # Infos sur les vendeurs
    sellers = pd.read_csv(os.path.join(path, "olist_sellers_dataset.csv"))
    # Infos sur les clients
    customers = pd.read_csv(os.path.join(path, "olist_customers_dataset.csv"))

    # Infos sur les catégories des produits
    category = pd.read_csv(os.path.join(path, "product_category_name_translation.csv"))


    # Traitement des NaN
    geoloc.drop_duplicates(inplace=True)

    payments = payments[~(payments['payment_type'] == 'not_defined')]
    # payments = payments.groupby('order_id')['payment_sequential','payment_installments', 'payment_value'].sum()

    payments = payments.groupby(['order_id'])[['payment_sequential', 'payment_installments', 'payment_value']].apply(sum)

    payments.columns = ['payment_sequential_nb', 'payment_instlmt_order_nb', 'total_payment_value']

    items['total_price'] = items["freight_value"] + items["price"]

    category['product_category_name_english'] = category['product_category_name_english'].apply(renameCategorieType)
    category['product_category_name_english'].unique()

    products["product_volume_cm3"] = products["product_length_cm"] * products["product_height_cm"] / products[
        "product_width_cm"]

    orders = orders[orders["order_status"] != "canceled"]

    # Conversion des données dates en datetime

    orders['order_approved_at'] = orders['order_approved_at'].apply(convertToDate)
    orders['order_estimated_delivery_date'] = orders['order_estimated_delivery_date'].apply(convertToDate)
    orders['order_delivered_customer_date'] = orders['order_delivered_customer_date'].apply(convertToDate)
    orders['order_purchase_timestamp'] = orders['order_purchase_timestamp'].apply(convertToDate)

    orders = orders[
        (orders["order_purchase_timestamp"] >= start_time) & (orders["order_purchase_timestamp"] < stop_time)]

    orders['delivery_time'] = orders['order_delivered_customer_date'] - orders['order_approved_at']
    orders['delivery_time'] = orders['delivery_time'].apply(getDays)
    orders['estimated_delivery_time'] = orders['order_estimated_delivery_date'] - orders['order_approved_at']
    orders['estimated_delivery_time'] = orders['estimated_delivery_time'].apply(getDays)
    # nombre de jours écoulés depuis dernier achat
    orders['delai_dernier_achat'] = orders['order_purchase_timestamp'].max() - orders['order_purchase_timestamp']
    orders['delai_dernier_achat'] = orders['delai_dernier_achat'].apply(getDays)

    # On s'interesse uniquement à l''heure, le jour, et le mois de l'achat
    orders.rename(columns={"order_purchase_timestamp": "purchase_time"}, inplace=True)

    orders["purchase_month"] = orders["purchase_time"].map(lambda d: d.month)

    geoloc.drop_duplicates(subset=["geolocation_zip_code_prefix"], keep="first", inplace=True)
    customers = customers.rename(columns={"customer_zip_code_prefix": "zip_code_prefix"})
    geoloc = geoloc.rename(columns={"geolocation_zip_code_prefix": "zip_code_prefix"})

    # jointure des fichiers

    products = pd.merge(products, category, how="left", on="product_category_name")

    del_features_list = ["product_category_name", "product_weight_g", "product_length_cm", "product_height_cm",
                         "product_width_cm", "product_description_lenght", "product_photos_qty", "product_name_lenght"]
    products.drop(del_features_list, axis=1, inplace=True)
    products = products.rename(columns={"product_category_name_english": "product_category_name"})

    order_items = pd.merge(items, orders, how="left", on="order_id")

    del_features_list = ["seller_id", "shipping_limit_date", "order_approved_at", "order_delivered_carrier_date",
                         "order_estimated_delivery_date"]
    order_items.drop(del_features_list, axis=1, inplace=True)

    order_items_payments = pd.merge(order_items, payments, how="left", on="order_id")

    # note moyenne des commentaires
    group_reviews = reviews.groupby("order_id").agg({"review_score": "mean"})

    order_items_payments = pd.merge(order_items_payments, group_reviews, how="left", on="order_id")
    customers_orders = pd.merge(order_items_payments, customers, how="left", on="customer_id")

    # Montant moyen des achats
    achats_moy = customers_orders.groupby(['customer_id', 'order_id'])['price'].sum().groupby(['customer_id']).mean()
    achats_moy.rename('Tot_moy_achats', inplace=True)

    products_per_order = customers_orders.groupby(["customer_id", "order_id"]).agg({"order_item_id": "count"})
    products_per_order = products_per_order.groupby("customer_id").agg({"order_item_id": "mean"})
    products_per_order = products_per_order.rename(columns={"order_item_id": "mean_items"})

    customers_orders = pd.merge(customers_orders, achats_moy, how="left", on="customer_id")
    customers_orders = pd.merge(customers_orders, products_per_order, how="left", on="customer_id")

    data = pd.merge(customers_orders, products, how="left", on="product_id")
   
    data = data.groupby("customer_unique_id").agg({
        "order_id": "nunique",
        "freight_value": "sum",
        # "payment_sequential_nb": "mean",
        "payment_instlmt_order_nb": "mean",
        "review_score": "mean",
        "delai_dernier_achat": "min",
        "delivery_time": "mean",
        "Tot_moy_achats": 'mean',
        # "mean_items": 'mean',
        "purchase_month": lambda x: x.value_counts().index[0]                   
    })

    data = data.rename(columns={# "order_id": "nb_orders",
                                "freight_value": "total_freight",
                                # "payment_sequential_nb": "mean_payment_sequential",
                                "payment_instlmt_order_nb": "mean_payment_installments",
                                "review_score": "mean_review_score",
                                "delai_dernier_achat": "delai_dernier_achat_mean",
                                "delivery_time": "delai_livraison",
                                "purchase_month": "favorite_sale_month",
                                "purchase_day": "favorite_sale_day",
                                "Nb_product_total": 'mean_nb_product_total',
                                })

    # passage au log de nos variables ne presentant pas de distribution normal
    data['total_freight'] = np.where(
        data['total_freight'] > 0, np.log2(data['total_freight']), 0)
    data['delai_livraison'] = np.where(
        data['delai_livraison'] > 0, np.log2(data['delai_livraison']), 0)
    data['Tot_moy_achats'] = np.where(
        data['Tot_moy_achats'] > 0, np.log2(data['Tot_moy_achats']), 0)
    data['mean_payment_installments'] = np.where(
        data['mean_payment_installments'] > 0,
        np.log2(data['mean_payment_installments']),
        0) 

    na_col = data.columns[data.isna().any()].tolist()
    if(len(na_col) > 0):
        data[na_col] = KNNImputer().fit_transform(data[na_col])
    #data = data[~data.isin([np.nan, np.inf, -np.inf]).any(1)]

    data.reset_index(inplace=True)
    data.set_index("customer_unique_id", inplace=True)

    return data


def plotPca(variance):
    # Plot of cumulated variance
    plt.figure(figsize=(12, 8))
    plt.bar(np.arange(len(variance)) + 1, variance)

    cumSumVar = variance.cumsum()
    plt.plot(np.arange(len(variance)) + 1, cumSumVar, c="red", marker='o')
    plt.axhline(y=95, linestyle="--",
                color="green",
                linewidth=1)

    limit = 95
    valid_idx = np.where(cumSumVar >= limit)[0]
    min_plans = valid_idx[cumSumVar[valid_idx].argmin()] + 1
    plt.axvline(x=min_plans, linestyle="--",
                color="green",
                linewidth=1)

    plt.xlabel("rang de l'axe d'inertie")
    plt.xticks(np.arange(len(variance)) + 1)
    plt.ylabel("pourcentage d'inertie")
    plt.title("{}% de la variance totale est expliquée"
              " par les {} premiers axes".format(limit, min_plans))
    plt.show(block=False)


def cerle_corr(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0):
    fig = plt.figure(figsize=(20, n_comp * 5))
    count = 1
    for d1, d2 in axis_ranks:
        if (d2 < n_comp):
            # initialisation de la figure
            ax = plt.subplot(int(n_comp / 2), 2, count)
            ax.set_aspect('equal', adjustable='box')
            # détermination des limites du graphique
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            # affichage des flèches
            ax.quiver(
                np.zeros(pcs.shape[1]),
                np.zeros(pcs.shape[1]),
                pcs[d1, :],
                pcs[d2, :],
                angles='xy',
                scale_units='xy',
                scale=1,
                color="grey",
                alpha=0.5)
            # et noms de variables
            for i, (x, y) in enumerate(pcs[[d1, d2]].T):
                ax.annotate(labels[i], (x, y), ha='center', va='center',
                            fontsize='14', color="#17aafa", alpha=0.8)

            # ajouter les axes
            ax.plot([-1, 1], [0, 0], linewidth=1, color='grey', ls='--')
            ax.plot([0, 0], [-1, 1], linewidth=1, color='grey', ls='--')

            # ajouter un cercle
            cercle = plt.Circle((0, 0), 1, color='#17aafa', fill=False)
            ax.add_artist(cercle)

            # nom des axes, avec le pourcentage d'inertie expliqué
            ax.set_xlabel('F{} ({}%)'
                          .format(d1 + 1,
                                  round(100 * pca.explained_variance_ratio_[d1],
                                        1)))
            ax.set_ylabel('F{} ({}%)'
                          .format(d2 + 1,
                                  round(100 * pca.explained_variance_ratio_[d2],
                                        1)))

            ax.set_title("Cercle des corrélations (F{} et F{})"
                         .format(d1 + 1, d2 + 1))
            count += 1
    plt.suptitle("Cercles des corrélations sur les {} premiers axes"
                 .format(n_comp), y=.9, color="blue", fontsize=18)
    plt.show(block=False)


def plot_dendrogram(model, **kwargs):
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


def plot_radars(data, group):
    scaler = MinMaxScaler()
    data = pd.DataFrame(scaler.fit_transform(data),
                        index=data.index,
                        columns=data.columns).reset_index()

    fig = go.Figure()

    for k in data[group]:
        fig.add_trace(go.Scatterpolar(
            r=data[data[group] == k].iloc[:, 1:].values.reshape(-1),
            theta=data.columns[1:],
            fill='toself',
            name='Cluster ' + str(k)
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title={
            'text': "Comparaison des clusters par leur moyennes des variables",
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
        title_font_color="blue",
        title_font_size=18)

    fig.show()


def plotTimeClusters(data,periode, col1, col2 ):
    # plot ARI Score
    fig = plt.figure(figsize=(12, 8))
    sns.lineplot(data=data, x=periode,
                 y=col1, color='green', markers='o')
    for x, y in zip(data[periode], data[col1]):
        plt.text(
         x=x,
         y=y+0.001,
         s=round(y, 3),
         color='black')
    sns.lineplot(data=data, x=periode, y=col2, color='blue')
    for x, y in zip(data[periode], data[col2]):
        plt.text(
         x=x,
         y=y+0.001,
         s=round(y, 3),
         color='black')
    plt.xlabel("Période (mois)")
    plt.title('Repartition des clusters dans le temps metric {}'.format(col1))
    plt.legend(labels=["modele global","modele initial"])
    plt.ylabel(col1)
    plt.axhline(y=np.median(data[col1]), color='blue', linestyle='--')
    
    plt.axhline(y=np.median(data[col2]), color='green', linestyle='--')

    plt.show()   
