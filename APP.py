import streamlit as st
import pickle
from utils import preprocess_input, get_prediction, get_probability

st.title("ProfitGuard")
st.subheader("Retail Profitability Decision Support System")
st.write("Enter details to check risk of loss")

model=pickle.load(open("model.pkl","rb"))
le1=pickle.load(open("le1.pkl","rb"))
le2=pickle.load(open("le2.pkl","rb"))

st.markdown("---")
st.header("Input Details")
col1, col2 = st.columns(2)

with col1:

    Sales = st.number_input("Sales", min_value=0.0, value=500.0)

    Quantity = st.number_input("Quantity", min_value=1, value=2)

    Category = st.selectbox(
        "Category",
        ["Furniture", "Technology", "Office Supplies"]
    )
with col2:

    Discount = st.slider("Discount", 0.0, 1.0, 0.1)

    Region = st.selectbox(
        "Region",
        ["West", "East", "Central", "South"]
    )

if Sales <= 0:
    st.warning("Sales must be greater than 0")

st.markdown("---")


if st.button("Predict Risk"):
    input_data = preprocess_input(Sales, Discount, Quantity, Category, Region, le1, le2)
    result = get_prediction(model, input_data)
    prob = get_probability(model, input_data)

    st.header("Prediction Result")
    if result == 1:
        st.error("High Risk")
        st.write("Higher discount may reduce profitability")
    else:
        st.success("Low Risk")
        st.write("Order looks safe for business")

    st.write(f"Risk Probability: {prob * 100:.2f}%")

    st.info("Higher discounts generally increase profitability risk.")


    st.markdown("---")

    st.subheader("Profitability Scenario Analysis")
    test_discounts = [0.0, 0.1, 0.2, 0.3]

    for d in test_discounts:

        temp_input = preprocess_input(
            Sales,
            d,
            Quantity,
            Category,
            Region,
            le1,
            le2
        )

        temp_result = get_prediction(model, temp_input)

        if temp_result == 1:

            st.write(f"Discount {d*100:.0f}% → High Risk")

        else:

            st.write(f"Discount {d*100:.0f}% → Low Risk")