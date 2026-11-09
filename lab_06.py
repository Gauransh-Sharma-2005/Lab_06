import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Olist Customer Analytics", layout="wide")


@st.cache_data
def load_data():
    folder = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(folder, "olist_customers_dataset.csv"))
    df["customer_city"] = df["customer_city"].str.title()
    df["zip_code"] = df["customer_zip_code_prefix"].astype(str).str.zfill(5)
    df["postal_region"] = df["customer_zip_code_prefix"] // 10000
    return df


df = load_data()

st.title("Olist Brazilian E-commerce Customer Analytics Dashboard")
st.write("Explore where the customers of the Olist marketplace are located across Brazil and how often they order.")

st.sidebar.header("Filters")

all_states = sorted(df["customer_state"].unique())
selected_states = st.sidebar.multiselect("Customer state", all_states, default=all_states)
region_range = st.sidebar.select_slider("Postal region (first digit of zip prefix)", options=list(range(10)), value=(0, 9))
city_search = st.sidebar.text_input("Search city name")

st.sidebar.header("Chart Options")
group_by = st.sidebar.selectbox("Bar chart groups customers by", ["State", "City"])
top_n = st.sidebar.slider("Number of bars (Top N)", 5, 20, 10)
trend_metric = st.sidebar.selectbox("Line chart metric", ["Customer records", "Unique customers", "Number of cities"])
min_customers = st.sidebar.slider("Minimum customers per city (scatter plot)", 1, 100, 1)
use_log = st.sidebar.checkbox("Use log scale on scatter plot", value=True)
preview_rows = st.sidebar.slider("Rows to preview in dataset", 5, 50, 10)

filtered = df[
    df["customer_state"].isin(selected_states)
    & df["postal_region"].between(region_range[0], region_range[1])
]
if city_search.strip() != "":
    filtered = filtered[filtered["customer_city"].str.contains(city_search.strip(), case=False)]

if filtered.empty:
    st.warning("No records match the selected filters. Please change the filters in the sidebar.")
    st.stop()

state_summary = filtered.groupby("customer_state").agg(
    customer_records=("customer_id", "count"),
    unique_customers=("customer_unique_id", "nunique"),
    cities=("customer_city", "nunique"),
    zip_prefixes=("customer_zip_code_prefix", "nunique"),
).sort_values("customer_records", ascending=False)

city_summary = filtered.groupby(["customer_city", "customer_state"]).agg(
    customers=("customer_id", "count"),
    zip_prefixes=("customer_zip_code_prefix", "nunique"),
).reset_index()
city_summary["label"] = city_summary["customer_city"] + " (" + city_summary["customer_state"] + ")"

unique_customers = filtered["customer_unique_id"].nunique()
orders_per_customer = filtered.groupby("customer_unique_id").size()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Records", f"{len(filtered):,}")
k2.metric("Unique Customers", f"{unique_customers:,}")
k3.metric("Cities", f"{len(city_summary):,}")
k4.metric("Avg Customers per City", f"{city_summary['customers'].mean():.1f}")
k5.metric("Avg Orders per Customer", f"{len(filtered) / unique_customers:.3f}")

tab1, tab2, tab3, tab4 = st.tabs(["Dataset Overview", "Data Analysis", "Visualizations", "Insights"])

with tab1:
    with st.container():
        st.subheader("About the Dataset")
        st.write(
            "This is the customers table of the Brazilian E-Commerce Public Dataset by Olist. "
            "A new customer_id is created for every order, while customer_unique_id identifies the actual person, "
            "so a person who ordered more than once appears in several records."
        )
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows (after filters)", f"{filtered.shape[0]:,}")
    c2.metric("Columns", filtered.shape[1])
    c3.metric("Missing values", int(filtered.isnull().sum().sum()))

    st.subheader("Data Preview")
    st.dataframe(
        filtered[["customer_id", "customer_unique_id", "zip_code", "customer_city", "customer_state"]].head(preview_rows),
    )

    with st.expander("Column descriptions"):
        st.write("customer_id: key to the orders table, a new value for every order")
        st.write("customer_unique_id: unique identifier of a customer")
        st.write("customer_zip_code_prefix: first five digits of the customer zip code")
        st.write("customer_city: customer city name")
        st.write("customer_state: customer state")
        st.write("zip_code: zip code prefix shown with leading zeros (created in the application)")
        st.write("postal_region: first digit of the zip code prefix (created in the application)")

    with st.expander("Summary statistics"):
        st.dataframe(filtered[["customer_zip_code_prefix", "postal_region"]].describe())

with tab2:
    st.subheader("Sum and Average")
    s1, s2, s3 = st.columns(3)
    s1.metric("Sum of customers over all cities", f"{city_summary['customers'].sum():,}")
    s2.metric("Average customers per state", f"{state_summary['customer_records'].mean():,.1f}")
    s3.metric("Average zip prefixes per city", f"{city_summary['zip_prefixes'].mean():.2f}")

    st.subheader("Category-wise Record Counts")
    a1, a2 = st.columns(2)
    with a1:
        st.write("Customer records per state")
        st.dataframe(state_summary["customer_records"])
    with a2:
        st.write("Customer records per postal region")
        region_counts = filtered.groupby("postal_region").size().rename("customer_records")
        st.dataframe(region_counts)

    st.subheader("Repeat Purchase Analysis")
    order_counts = orders_per_customer.value_counts().sort_index().rename("number_of_customers")
    order_counts.index.name = "orders_placed"
    st.dataframe(order_counts)

    with st.expander("State-wise summary"):
        st.dataframe(state_summary)

    with st.expander("Top 15 cities by customer records"):
        st.dataframe(city_summary.nlargest(15, "customers")[["label", "customers", "zip_prefixes"]])

with tab3:
    v1, v2 = st.columns(2)

    with v1:
        st.subheader("Bar Chart")
        if group_by == "State":
            bar_data = state_summary["customer_records"].head(top_n).sort_values()
            bar_labels = bar_data.index
            x_label = "State"
        else:
            top_cities = city_summary.nlargest(top_n, "customers").sort_values("customers")
            bar_data = top_cities.set_index("label")["customers"]
            bar_labels = bar_data.index
            x_label = "City"
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.barh(bar_labels, bar_data.values, color="tab:blue", label="Customer records")
        ax.set_title(f"Top {len(bar_data)} {x_label}s by Customer Records")
        ax.set_xlabel("Number of customer records")
        ax.set_ylabel(x_label)
        ax.legend(loc="lower right")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        share = bar_data.max() / len(filtered) * 100
        st.caption(
            f"Interpretation: '{bar_data.idxmax()}' has the most customer records ({bar_data.max():,}), "
            f"which is {share:.1f}% of the selected records."
        )

    with v2:
        st.subheader("Line Chart")
        by_region = filtered.groupby("postal_region").agg(
            records=("customer_id", "count"),
            unique=("customer_unique_id", "nunique"),
            cities=("customer_city", "nunique"),
        ).reset_index()
        column_map = {"Customer records": "records", "Unique customers": "unique", "Number of cities": "cities"}
        column = column_map[trend_metric]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(by_region["postal_region"], by_region[column], marker="o", color="tab:green", label=trend_metric)
        ax.set_title(f"{trend_metric} by Postal Region")
        ax.set_xlabel("Postal region (first digit of zip prefix)")
        ax.set_ylabel(trend_metric)
        ax.set_xticks(by_region["postal_region"])
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        peak = by_region.loc[by_region[column].idxmax()]
        low = by_region.loc[by_region[column].idxmin()]
        st.caption(
            f"Interpretation: postal region {int(peak['postal_region'])} has the highest value ({peak[column]:,}) "
            f"and postal region {int(low['postal_region'])} has the lowest ({low[column]:,})."
        )

    with st.container():
        st.subheader("Scatter Plot")
        scatter_data = city_summary[city_summary["customers"] >= min_customers]
        if len(scatter_data) < 2:
            st.info("Not enough cities for the scatter plot. Lower the minimum customers per city in the sidebar.")
        else:
            in_sp = scatter_data[scatter_data["customer_state"] == "SP"]
            other = scatter_data[scatter_data["customer_state"] != "SP"]
            fig, ax = plt.subplots(figsize=(10, 4.8))
            ax.scatter(other["zip_prefixes"], other["customers"], alpha=0.5, color="tab:blue", label="Cities in other states")
            ax.scatter(in_sp["zip_prefixes"], in_sp["customers"], alpha=0.6, color="tab:red", label="Cities in SP")
            if use_log:
                ax.set_xscale("log")
                ax.set_yscale("log")
            ax.set_title("Zip Code Prefixes vs Customers per City")
            ax.set_xlabel("Number of distinct zip code prefixes in the city")
            ax.set_ylabel("Number of customer records in the city")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            correlation = scatter_data["zip_prefixes"].corr(scatter_data["customers"])
            biggest = scatter_data.loc[scatter_data["customers"].idxmax()]
            st.caption(
                f"Interpretation: the correlation between zip code prefixes and customers per city is {correlation:.2f}. "
                f"The largest point is {biggest['label']} with {biggest['customers']:,} customers "
                f"spread over {biggest['zip_prefixes']} zip code prefixes."
            )

with tab4:
    st.subheader("Key Insights for the Selected Data")

    top_state = state_summary.index[0]
    top_state_share = state_summary["customer_records"].iloc[0] / len(filtered) * 100
    top_three_share = state_summary["customer_records"].head(3).sum() / len(filtered) * 100
    top_city = city_summary.loc[city_summary["customers"].idxmax()]
    repeat_customers = int((orders_per_customer > 1).sum())
    repeat_share = repeat_customers / unique_customers * 100
    single_cities = int((city_summary["customers"] == 1).sum())
    best_region = filtered.groupby("postal_region").size().idxmax()

    i1, i2 = st.columns(2)
    with i1:
        with st.container():
            st.write(f"Top state: {top_state} with {top_state_share:.1f}% of the selected records")
            st.write(f"Top three states together: {top_three_share:.1f}% of the selected records")
            st.write(f"Top city: {top_city['label']} with {top_city['customers']:,} records")
            st.write(f"Postal region with most records: {best_region}")
    with i2:
        with st.container():
            st.write(f"Repeat customers (more than one order): {repeat_customers:,} ({repeat_share:.1f}% of unique customers)")
            st.write(f"Most orders placed by one customer: {orders_per_customer.max()}")
            st.write(f"Cities with only one customer record: {single_cities:,} of {len(city_summary):,}")

    with st.expander("How to read these insights"):
        st.write("A high share for the top state or city shows that the customer base is concentrated in a few places.")
        st.write("A low repeat customer share shows that most customers buy only once from the marketplace.")

    with st.expander("Suggestions for the marketplace"):
        st.write("Strengthen delivery and seller support in the states and cities that hold most of the customers.")
        st.write("Run offers and reminders to turn one-time buyers into repeat customers.")
        st.write("Study the cities that have very few customers to find places where the marketplace can grow.")