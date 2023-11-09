import streamlit as st
from tensorflow.keras.models import load_model
import pandas as pd

# Load the Random Forest model
# with open('random_forest_model.pkl', 'rb') as model_file:
#     rf_model = pickle.load(model_file)

# Load the ANN model
st.write("Loading ANN model...")
ann_model = load_model('ann_model.h5')
st.write("ANN model loaded successfully.")

st.image('alpha-1.png', width=400)
#END OF BANNER

# FOR GitHub Link
st.header('Credit Card Fraud Detection App', divider='rainbow')
st.header(':blue[_GROUP 6_] MH6804: Python For Data Analysis :sunglasses:')

def load_css():
    st.markdown("""
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ775FNIXfOHhU8K0iwVfZJ0JaxNSEO+lj5n" crossorigin="anonymous">
""", unsafe_allow_html=True)

def create_link_button(label, url, button_style='primary', button_size='md'):
    st.markdown(f'<a href="{url}" target="_blank">'
                f'<button type="button" class="btn btn-{button_style} btn-{button_size}">'
                f'{label}</button></a>', unsafe_allow_html=True)

# Example usage of the create_link_button function

load_css()  # Make sure to load the CSS styles
#END OF ICON

st.write("""
###### NANYANG TECHNOLOGICAL UNIVERSITY, Master of Science in Financial Technology
###### By Dymasius Y Sitepu, Cevin A N Putra, Ling Zhou, Shining Zhang, Fang An
         """)
create_link_button('Visit The GitHub Code', 'https://github.com/dymasius12/CC_Fraud_Detection_App', 'primary', 'lg')

with st.expander("STEP 0: What is this app about?"):
    st.write("""
###### - Project Context:
It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase.
This app is to help identify the fraudulent credit card transactions

###### - About the Dataset:
The dataset contains transactions made by credit cards in September 2013 by European cardholders.
This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.
             """)


with st.expander("STEP 1: How to use this Machine Learning app?"):
    st.write("""
###### Step-By-Step Guide:
1. Input your data at "Input User Parameter" on the sidebar
2. See your desired data on the center page
3. See the predicted output
4. Play around
5. Please contact Dymasius Y Sitepu for queries: dymasius001@e.ntu.edu.sg
             """)

# For Sidebar User Input Parameter
st.sidebar.header('User Input Parameter')
st.sidebar.write("Note: input your data, amount = $$$")

def user_input_features():
    amount = st.sidebar.slider('amount', 0, 25691, 1000)
    v1 = st.sidebar.slider('v1', -70.0, 120.0, 5.4)
    v2 = st.sidebar.slider('v2', -70.0, 120.0, 3.4)
    v3 = st.sidebar.slider('v3', -70.0, 120.0, 1.3)
    v4 = st.sidebar.slider('v4', -70.0, 120.0, 0.2)
    v5 = st.sidebar.slider('v5', -70.0, 120.0, 0.2)
    v6 = st.sidebar.slider('v6', -70.0, 120.0, 5.4)
    v7 = st.sidebar.slider('v7', -70.0, 120.0, 3.4)
    v8 = st.sidebar.slider('v8', -70.0, 120.0, 1.3)
    v9 = st.sidebar.slider('v9', -70.0, 120.0, 0.2)
    v10 = st.sidebar.slider('v10', -70.0, 120.0, 0.2)
    v11 = st.sidebar.slider('v11', -70.0, 120.0, 5.4)
    v12 = st.sidebar.slider('v12', -70.0, 120.0, 3.4)
    v13 = st.sidebar.slider('v13', -70.0, 120.0, 1.3)
    v14 = st.sidebar.slider('v14', -70.0, 120.0, 0.2)
    v15 = st.sidebar.slider('v15', -70.0, 120.0, 0.2)
    v16 = st.sidebar.slider('v16', -70.0, 120.0, 5.4)
    v17 = st.sidebar.slider('v17', -70.0, 120.0, 3.4)
    v18 = st.sidebar.slider('v18', -70.0, 120.0, 1.3)
    v19 = st.sidebar.slider('v19', -70.0, 120.0, 0.2)
    v20 = st.sidebar.slider('v20', -70.0, 120.0, 0.2)
    v21 = st.sidebar.slider('v21', -70.0, 120.0, 5.4)
    v22 = st.sidebar.slider('v22', -70.0, 120.0, 3.4)
    v23 = st.sidebar.slider('v23', -70.0, 120.0, 1.3)
    v24 = st.sidebar.slider('v24', -70.0, 120.0, 0.2)
    v25 = st.sidebar.slider('v25', -70.0, 120.0, 0.2)
    v26 = st.sidebar.slider('v26', -70.0, 120.0, 5.4)
    v27 = st.sidebar.slider('v27', -70.0, 120.0, 3.4)
    v28 = st.sidebar.slider('v28', -70.0, 120.0, 1.3)
    data = {
            'amount': amount,
            'v1': v1,
            'v2': v2,
            'v3': v3,
            'v4': v4,
            'v5': v5,
            'v6': v6,
            'v7': v7,
            'v8': v8,
            'v9': v9,
            'v10': v10,
            'v11': v11,
            'v12': v12,
            'v13': v13,
            'v14': v14,
            'v15': v15,
            'v16': v16,
            'v17': v17,
            'v18': v18,
            'v19': v19,
            'v20': v20,
            'v21': v21,
            'v22': v22,
            'v23': v23,
            'v24': v24,
            'v25': v25,
            'v26': v26,
            'v27': v27,
            'v28': v28,
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()
# End of User Input Parameter

# Displaying User parameters
st.subheader('User Parameters Display:')
st.write(df)

st.subheader('Class labels and their corresponding index number')

st.subheader('Prediction')

st.subheader('Prediction Probability')

with st.expander("STEP 2: Model Prediction"):
    st.write("""
###### Wait for the Model to Predict the Result:
             """)
    
# Predict the output using the loaded models
#rf_prediction = rf_model.predict(df)
ann_prediction = ann_model.predict(df)
ann_prediction = (ann_prediction > 0.5)  # Threshold predictions at 0.5
st.subheader('STEP 3: Prediction')
st.write("ANN Prediction:", ann_prediction)

# Display the predictions
st.subheader('STEP 3: Prediction')
#st.write("Random Forest Prediction:", rf_prediction)
st.write("ANN Prediction:", ann_prediction)

with st.expander("Credits & Acknowledgements:"):
    st.write("""
The dataset has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection.
More details on current and past projects on related topics are available on https://www.researchgate.net/project/Fraud-detection-5 and the page of the DefeatFraud project

Please cite the following works:

Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015

Dal Pozzolo, Andrea; Caelen, Olivier; Le Borgne, Yann-Ael; Waterschoot, Serge; Bontempi, Gianluca. Learned lessons in credit card fraud detection from a practitioner perspective, Expert systems with applications,41,10,4915-4928,2014, Pergamon

Dal Pozzolo, Andrea; Boracchi, Giacomo; Caelen, Olivier; Alippi, Cesare; Bontempi, Gianluca. Credit card fraud detection: a realistic modeling and a novel learning strategy, IEEE transactions on neural networks and learning systems,29,8,3784-3797,2018,IEEE

Dal Pozzolo, Andrea Adaptive Machine learning for credit card fraud detection ULB MLG PhD thesis (supervised by G. Bontempi)

Carcillo, Fabrizio; Dal Pozzolo, Andrea; Le Borgne, Yann-Aël; Caelen, Olivier; Mazzer, Yannis; Bontempi, Gianluca. Scarff: a scalable framework for streaming credit card fraud detection with Spark, Information fusion,41, 182-194,2018,Elsevier

Carcillo, Fabrizio; Le Borgne, Yann-Aël; Caelen, Olivier; Bontempi, Gianluca. Streaming active learning strategies for real-life credit card fraud detection: assessment and visualization, International Journal of Data Science and Analytics, 5,4,285-300,2018,Springer International Publishing

Bertrand Lebichot, Yann-Aël Le Borgne, Liyun He, Frederic Oblé, Gianluca Bontempi Deep-Learning Domain Adaptation Techniques for Credit Cards Fraud Detection, INNSBDDL 2019: Recent Advances in Big Data and Deep Learning, pp 78-88, 2019

Fabrizio Carcillo, Yann-Aël Le Borgne, Olivier Caelen, Frederic Oblé, Gianluca Bontempi Combining Unsupervised and Supervised Learning in Credit Card Fraud Detection Information Sciences, 2019

Yann-Aël Le Borgne, Gianluca Bontempi Reproducible machine Learning for Credit Card Fraud Detection - Practical Handbook

Bertrand Lebichot, Gianmarco Paldino, Wissam Siblini, Liyun He, Frederic Oblé, Gianluca Bontempi Incremental learning strategies for credit cards fraud detection, IInternational Journal of Data Science and Analytics
    """)
