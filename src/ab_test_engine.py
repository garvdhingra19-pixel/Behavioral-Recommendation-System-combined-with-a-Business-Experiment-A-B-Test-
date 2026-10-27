import pandas as pd
import numpy as np
from scipy import stats

def simulate_ab_test(data_path="data/sampled_instacart_orders.csv", recommendation_conversion=0.05):
    print("Loading sampled data...")
    df = pd.read_csv(data_path)
    
    # Calculate baseline basket sizes (items per order)
    basket_sizes = df.groupby('order_id').size().values
    
    print("Simulating Control vs. Treatment split...")
    # 50/50 Split for A/B Test
    np.random.seed(42)
    split = np.random.rand(len(basket_sizes))
    
    # Control Group A: Standard baseline behavior
    control_aov = basket_sizes[split < 0.5]
    
    # Treatment Group B: Baseline + Recommendation Widget Uplift
    # Simulates showing 3 recommended items with a 5% baseline conversion probability
    treatment_base = basket_sizes[split >= 0.5]
    uplift = np.random.binomial(n=3, p=recommendation_conversion, size=len(treatment_base))
    treatment_aov = treatment_base + uplift
    
    print("Running Statistical Hypothesis Test (Two-Sample T-Test)...")
    t_stat, p_value = stats.ttest_ind(treatment_aov, control_aov, alternative='greater')
    
    results = {
        "Control_Avg_Basket_Size": round(np.mean(control_aov), 4),
        "Treatment_Avg_Basket_Size": round(np.mean(treatment_aov), 4),
        "Absolute_Uplift": round(np.mean(treatment_aov) - np.mean(control_aov), 4),
        "P_Value": p_value,
        "Statistically_Significant": p_value < 0.05
    }
    
    print("\n--- A/B Test Results ---")
    for key, value in results.items():
        print(f"{key}: {value}")
        
    return results

if __name__ == "__main__":
    simulate_ab_test()