# E-Commerce Recommendation Engine & A/B Testing Pipeline

**Executive Summary**
This project simulates an enterprise-grade quick-commerce data pipeline using the Instacart dataset. It applies Market Basket Analysis to discover high-value product associations and utilizes a statistical A/B testing simulation to mathematically validate the revenue impact (Average Order Value uplift) of the recommendation algorithm.

**Core Technologies**
* **Languages & Libraries:** Python (Pandas, NumPy, SciPy)
* **Machine Learning:** `mlxtend` (FP-Growth, Association Rules)
* **Interactive Frontend:** Streamlit
* **Business Intelligence:** Power BI / Tableau

**Methodology & Business Value**
* **Algorithm Upgrade:** Transitioned from standard Apriori to the **FP-Growth** algorithm to efficiently process massive transactional datasets using in-memory tree structures.
* **Solving the "Banana Problem":** Filtered out universal staple goods (e.g., bananas, spinach) to force the algorithm to identify niche, actionable purchasing patterns (e.g., Limes + Jalapeno Peppers).
* **Advanced Metrics:** Prioritized product rules by **Lift** (strength of association) and **Conviction** (dependency level) rather than raw Confidence to maximize upselling profitability.
* **Statistical Validation:** Engineered a Two-Sample T-Test simulation engine using SciPy. Split historical orders into Control and Treatment groups, proving a statistically significant increase ($p < 0.05$) in Average Order Value (AOV) when recommendations are applied.

**Local Installation & Execution**
1. **Clone the repository:** `git clone https://github.com/garvdhingra19-pixel/Behavioral-Recommendation-System-combined-with-a-Business-Experiment-A-B-Test-.git`
2. **Generate the data sample:** `python src/data_loader.py`
3. **Train the ML model:** `python src/recommendation_model.py`
4. **Launch the A/B test dashboard:** `streamlit run dashboard/app.py`