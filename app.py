import streamlit as st
import numpy as np

model = np.load('D:/Diamond_Price_Predict/pythonProject/weight.npz')
xmean = model['x_mean']
xstd = model['x_std']
theta = model['the_ta']

def predict(carat,	cut,	color,	clarity,	depth,	table,	x,	y,	z):
    #tạo bộ từ diển -encoding
    cut_mapping = {"Fair": 1, "Good": 2, "Very Good": 3, "Premium": 4, "Ideal": 5}
    color_mapping = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
    clarity_mapping = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}

    #chuyển đổi những thuộc tính dạng text sang số từ những giá trị người dùng nhập vào
    cut = cut_mapping.get(cut)
    color = color_mapping.get(color)
    clarity = clarity_mapping.get(clarity)

    #normaliz data nhập từ người dùng
    data_input = np.array([carat,	cut,	color,	clarity,	depth,	table,	x,	y,	z], dtype='float')
    data_input = (data_input - xmean)/xstd
    b = np.array([1])
    data_input = np.concatenate((b,data_input), axis=0)
    pred = data_input.dot(theta)
    return pred


#Thêm giao diện
st.title("DIAMOND PRICE PREDICTION")
st.header("Nhập các thông số về kim cương bạn muốn mua ")
carat = st.number_input('Carat: ', min_value=0.1, max_value=10.0, value=1.0)
cut = st.selectbox('Cut: ', ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox('Color: ', ["J", "I", "H", "G", "F", "E", "D"])
clarity = st.selectbox('Clarity', ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])
depth = st.number_input('Depth: ', min_value=0.1, max_value=100.0, value=1.0)
table = st.number_input('Table ', min_value=0.1, max_value=100.0, value=1.0)
x = st.number_input('X: ', min_value=0.1, max_value=100.0, value=1.0)
y = st.number_input('Y: ', min_value=0.1, max_value=100.0, value=1.0)
z = st.number_input('Z: ', min_value=0.1, max_value=100.0, value=1.0)
if st.button('Predict Price'):
    output = predict(carat, cut, color, clarity, depth, table, x, y, z)
    st.success(f"Giá dự đoán kim cương là: {output[0]:.2f} USD")






