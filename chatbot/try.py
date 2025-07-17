import streamlit as st

st.title("🎥 Free GenAI Movie Recommender")
user_input = st.text_input("What type of movie are you looking for?")

if st.button("Get Recommendations"):
    query = chain.run(user_input)
    movies = get_movies_tmdb(query)

    if not movies:
        st.error("No movies found.")
    for m in movies:
        st.subheader(m['title'])
        st.write("⭐", m.get('vote_average', 'N/A'), "| 📅", m.get('release_date', 'N/A'))
        st.write(m.get('overview', 'No description available'))
        st.markdown(f"[Watch Online]({get_watch_link(m['title'])})", unsafe_allow_html=True)
