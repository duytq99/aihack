import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
import seaborn as sns
from utils import load_data, load_xgboost_model, footer

product_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
default_list = ['A', 'B', 'C', 'D']
########################### HEADING ###########################
st.title("ỨNG DỤNG AI TỐI ƯU LƯỢNG HÀNG TỒN KHO")


########################## SALE PRED ##########################
st.header(f"Dự đoán nhu cầu sản phẩm")

# Select products
st.subheader("Sản phẩm")
product_options = st.multiselect("Chọn sản phẩm",product_list,default_list)

# Historical data
time = 54
source1 = pd.DataFrame(np.cumsum(np.random.randn(time, len(product_options)), 0).round(2),
                    columns=product_options, index=pd.RangeIndex(time, name='x'))
source = source1.reset_index().melt('x', var_name='category', value_name='y')

st.subheader("Thông tin bán hàng quá khứ")
st.dataframe(source1.loc[:,product_options].T)
line_chart = alt.Chart(source).mark_line(interpolate='basis').encode(alt.X('x', title='Tuần'),alt.Y('y', title='Số lượng'),color='category:N').properties(title='Lượng sản phẩm bán ra theo thời gian')
st.altair_chart(line_chart, use_container_width=True)

# Prediction
st.subheader(f"Dự đoán nhu cầu sản phẩm trong 2 tuần tới")
st.write('Tuần thứ nhất')
col_tuple = st.columns(len(product_options))
for i, col in enumerate(col_tuple):
    predict_result = 0
    last_result = 0
    col.metric(product_options[i], predict_result, predict_result-last_result)
st.write('Tuần thứ hai')
col_tuple = st.columns(len(product_options))
for i, col in enumerate(col_tuple):
    predict_result = 0
    last_result = 0
    col.metric(product_options[i], predict_result, predict_result-last_result)
########################## INV OPTIM ##########################
st.header(f"Dự đoán số lượng tồn kho tối ưu")
# Current inventory
st.subheader(f"Tồn kho hiện tại")
inv = pd.DataFrame(np.random.randint(1, 100, (1,len(product_options))),columns=product_options)
st.bar_chart(inv.T, use_container_width=True)
# Optimal order quantity
st.subheader(f"Tồn kho tối ưu")
col_tuple = st.columns(len(product_options))
print(inv)
for i, col in enumerate(col_tuple):
    last_result = inv[product_options[i]].values[0]
    predict_result = last_result + np.random.randint(0, 20)
    col.metric(product_options[i], int(last_result+predict_result), int(predict_result))
# Evaluation metrics
# st.subheader(f"Chỉ số đánh giá")
########################### FOOTER ############################
footer()
