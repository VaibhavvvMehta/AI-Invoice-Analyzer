import streamlit as st
import langchain_helper

st.title("🍽️ Restaurant Name Generator")

# Dropdown for cuisine selection
cuisine = st.sidebar.selectbox("Pick a Cuisine", ("Indian", "Italian", "Mexican", "Arabic", "American"))

if cuisine:
    response = langchain_helper.generate_restaurant_name_and_items(cuisine)
    st.header(response['restaurant_name'].strip())

    # Get and split menu items
    menu_items = response['menu_items'].split(",")
    
    st.write("**Menu Items:**")
    for item in menu_items:
        st.write(f"- {item.strip()}")
