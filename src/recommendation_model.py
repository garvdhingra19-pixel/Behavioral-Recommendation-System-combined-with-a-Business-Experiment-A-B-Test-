import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

def train_association_rules(data_path="data/sampled_instacart_orders.csv", top_n_products=150):
    print("Loading data for Market Basket Analysis...")
    df = pd.read_csv(data_path)
    
    # Exclude staples to force the algorithm to find unique, actionable pairings
    staple_items = ['Banana', 'Bag of Organic Bananas', 'Organic Strawberries', 'Organic Baby Spinach', 'Large Lemon']
    df = df[~df['product_name'].isin(staple_items)]
    
    top_products = df['product_name'].value_counts().head(top_n_products).index
    df_filtered = df[df['product_name'].isin(top_products)]
    
    baskets = df_filtered.groupby('order_id')['product_name'].apply(list).values.tolist()
    
    te = TransactionEncoder()
    te_ary = te.fit(baskets).transform(baskets)
    basket_df = pd.DataFrame(te_ary, columns=te.columns_)
    
    print("Running FP-Growth algorithm (Industry Standard for speed)...")
    # FP-growth is exponentially faster than Apriori for large e-commerce datasets
    frequent_itemsets = fpgrowth(basket_df, min_support=0.005, use_colnames=True)
    
    print("Generating recommendation rules...")
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
    
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x)[0])
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x)[0])
    
    # Sort by Lift (strength of pair) and Conviction (dependency)
    rules = rules.sort_values(['lift', 'conviction'], ascending=[False, False])
    
    # Save for Power BI / Tableau later
    rules[['antecedents', 'consequents', 'support', 'confidence', 'lift', 'conviction']].to_csv("data/recommendation_rules.csv", index=False)
    
    print("\nModel trained successfully using FP-Growth!")
    return rules

if __name__ == "__main__":
    train_association_rules()