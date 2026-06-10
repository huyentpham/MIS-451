# E-commerce Shipping via Classification Analysis (MIS-451) - Group Lazy

A machine learning classification project that helps an e-commerce business optimize operations by predicting whether an order will reach its destination on time using historical logistics data.

### Business Problem

In a competitive e-commerce environment, late deliveries lead to customer dissatisfaction and increased customer care calls. Organizations often react to delayed shipments after the fact, which is inefficient. By predicting delivery status based on historical data, the logistics team can proactively allocate resources to high-risk orders, and marketing can better align promotional campaigns with warehouse and transportation capacities.

### Objectives

* Conduct comprehensive data analysis to explore the structure, distribution, and relationships between variables to identify key factors influencing delivery delays.
* Develop a robust preprocessing pipeline and compare four classification models (KNN, SVM, Logistic Regression, ANN) to select the best performer.
* Translate technical results into actionable business recommendations that the operations department can implement immediately.

### Dataset

* **Source:** E-Commerce Shipping Dataset (Kaggle).
* **Size:** 10,999 observations.
* **Inputs:** 10 variables (6 numeric, 3 categorical) including demographics, shipping details, and product info.
* **Target:** `Reached.on.Time_Y.N` (0 = On-time delivery, 1 = Late delivery).

### Workflow (Methods)

**EDA**
* Univariate and bivariate exploration (categorical vs target, numerical vs target) to identify strong predictors like `Discount_offered` and `Weight_in_gms`.
* Outlier inspection using IQR; outliers in `Prior_purchases` and `Discount_offered` were kept because they reflect real business reality (loyal customers, flash sales) rather than data errors.

**Data Cleaning & Feature Engineering**
* Removed `ID` and `Gender` columns as they had no analytical value or correlation with delivery status.
* Dataset confirmed to have zero missing values and zero duplicates, skipping the need for imputation.

**Preprocessing Pipeline**
* One-Hot Encoding for the 3 categorical variables (`Warehouse_block`, `Mode_of_Shipment`, `Product_importance`) to prevent data leakage (fitted on train, transformed on test).
* StandardScaler applied to normalize the feature matrix to a mean of 0 and standard deviation of 1.

**Modeling (Classification)**
* Logistic Regression (baseline).
* K-Nearest Neighbors (KNN) optimized with GridSearchCV.
* Support Vector Machine (SVM).
* Artificial Neural Network (ANN - Deep Learning model) utilizing Dropout and EarlyStopping.

**Evaluation**
* Focus on the F1-macro metric rather than accuracy due to a slight class imbalance (60% late / 40% on-time).
* Used a stratified train-test split (80/20) and 10-Fold Cross-Validation on the training set to ensure fair evaluation.

### Key Results (Summary)
* **Best selected model:** Artificial Neural Network (ANN).
* **Test performance:** Achieved an overall accuracy of 67.41% on the test set.
* The model demonstrated high cost-efficiency with a Precision of 85.06% for the 'Delayed Delivery' class, meaning out of 100 alerts generated, 85 are accurate.
* The Recall for delayed orders was 55.06%, successfully preventing 32.9% of total test order delays.

### Business Insights & Recommendations

* **Implement a Real-Time Alert System:** Integrate the ANN model into the Order Management System (OMS) to automatically assess risk and prioritize high-probability delayed orders in the processing queue.
* **Manage Promotions with Logistics Capacity:** Plan flash sales in advance, as high `Discount_offered` strongly correlates with delays. Ensure warehouse and transportation capacities are expanded before large campaigns.
* **Proactive Customer Care:** Equip the customer service department with tracking tools for at-risk orders to improve NPS and reduce cancellation rates.
* **Technical Tuning:** Consider lowering the classification threshold from 0.5 to 0.35–0.40 to increase Recall, adjusting based on the specific cost trade-off of false positives versus false negatives.

### Repository Structure

* `Project_451_Group.ipynb` — Jupyter Notebook (EDA, preprocessing, modeling)
* `Group_MIS 451_Report.docx` — Full project report
* `MIS451_GroupLazy.pdf` — Presentation slides
* `README.md` — Project overview

### How to Run

1. Open `Project_451_Group.ipynb`.
2. Install requirements (typical): `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`.
3. Ensure TensorFlow/Keras is installed for the ANN model.
4. Run all cells sequentially. The pipeline handles data splitting, one-hot encoding, standard scaling, and model evaluation automatically.

### Team (Group Lazy)

* **Lê Anh Khương:** Model Development and Evaluation.
* **Phạm Thúy Huyền:** Data Processing and Transformation.
* **Võ Thị Hoài Anh:** Data Overview and Preprocessing.
