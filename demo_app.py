import streamlit as st
import pandas as pd 
import numpy as np
import altair as alt
from utils import load_data, load_xgboost_model, footer

pred_2017_32week = np.load('data/pred_2017_32week.npy').round(2)*100
true_2017_32week = np.load('data/true_2017_32week.npy').round(2)*100
true_2016_week = np.load('data/true_2016_week.npy').round(2)*100

product_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
default_list = ['A', 'B', 'C', 'D']

df_pred_2017_32week = pd.DataFrame(pred_2017_32week.T, columns = product_list, index=pd.RangeIndex(32, name='x'))
df_true_2017_32week = pd.DataFrame(true_2017_32week.T, columns = product_list, index=pd.RangeIndex(32, name='x'))
df_true_2016_week = pd.DataFrame(true_2016_week.T, columns = product_list, index=pd.RangeIndex(51, name='x'))

########################### HEADING ###########################
st.title("ỨNG DỤNG AI TỐI ƯU LƯỢNG HÀNG TỒN KHO")


########################## SALE PRED ##########################
st.header(f"Dự đoán nhu cầu sản phẩm")

# Select products
st.subheader("Sản phẩm")
product_options = st.multiselect("Chọn sản phẩm",product_list,default_list)

# Historical data
time = 51
source1 = df_true_2016_week[product_options]
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
    predict_result = pred_2017_32week[i, 0]
    last_result = true_2016_week[i,-1]
    col.metric(product_options[i], predict_result.round(2), (predict_result-last_result).round(2))
st.write('Tuần thứ hai')
col_tuple = st.columns(len(product_options))
for i, col in enumerate(col_tuple):
    predict_result = pred_2017_32week[i, 1]
    last_result = true_2016_week[i,-1]
    col.metric(product_options[i], predict_result.round(2), (predict_result-last_result).round(2))
########################## INV OPTIM ##########################
st.header(f"Dự đoán số lượng tồn kho tối ưu")
# Current inventory
st.subheader(f"Tồn kho hiện tại")
inv = pd.DataFrame(np.random.randint(1, 100, (1,len(product_options))),columns=product_options)
st.bar_chart(inv.T, use_container_width=True)
# Optimal order quantity
st.subheader(f"Tồn kho tối ưu")
col_tuple = st.columns(len(product_options))
# print(inv)
for i, col in enumerate(col_tuple):
    last_result = inv[product_options[i]].values[0]
    predict_result = last_result + np.random.randint(0, 20)
    col.metric(product_options[i], int(last_result+predict_result), int(predict_result))
# Evaluation metrics
# st.subheader(f"Chỉ số đánh giá")
########################### FOOTER ############################
footer()
