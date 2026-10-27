import pandas as pd
import os

def load_and_sample_data(data_dir="data", sample_fraction=0.1, random_state=42):
    """Loads Instacart data, merges tables, and returns a manageable sample."""
    print("Loading raw CSVs...")
    orders = pd.read_csv(os.path.join(data_dir, "orders.csv"))
    order_products = pd.read_csv(os.path.join(data_dir, "order_products_prior.csv"))
    
    products = pd.read_csv(os.path.join(data_dir, "products.csv"))
    
    print("Merging datasets...")
    merged_df = order_products.merge(products[['product_id', 'product_name']], on='product_id')
    final_df = merged_df.merge(orders[['order_id', 'user_id', 'eval_set']], on='order_id')
    
    # Filter only prior orders to train the recommendation engine
    final_df = final_df[final_df['eval_set'] == 'prior']
    
    print(f"Sampling {sample_fraction * 100}% of users...")
    unique_users = final_df['user_id'].unique()
    sampled_users = pd.Series(unique_users).sample(frac=sample_fraction, random_state=random_state)
    
    sampled_data = final_df[final_df['user_id'].isin(sampled_users)]
    
    return sampled_data

if __name__ == "__main__":
    df = load_and_sample_data()
    df.to_csv("data/sampled_instacart_orders.csv", index=False)
    print(f"Sample of {len(df)} rows saved to data/sampled_instacart_orders.csv")