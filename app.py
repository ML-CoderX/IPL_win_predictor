import streamlit as st
import pickle as p
import pandas as pd
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="IPL Win Predictor", layout="centered")

# Load model
pipe = p.load(open('C:\\Users\\rsaad\\PycharmProjects\\ipl_win_predictor\\pipe.pkl', 'rb'))

# Team Logos (URLs or local paths if needed)
team_logos = {
    'Sunrisers Hyderabad': "C:\\Users\\rsaad\\PycharmProjects\\ipl_win_predictor\\sunrisers_hyderabad.jpg",
    'Mumbai Indians': "C:\\Users\\rsaad\\PycharmProjects\\ipl_win_predictor\\mumbai_indians.png",
    'Royal Challengers Bangalore': 'C:\\Users\\rsaad\\PycharmProjects\\ipl_win_predictor\\rcb.jpg',
    'Kolkata Knight Riders': 'C:\\Users\\rsaad\\PycharmProjects\\ipl_win_predictor\\kkr.png',
    'Kings XI Punjab': 'C:\\Users\\rsaad\\PycharmProjects\\ipl_win_predictor\\kings punjab.png',
    'Chennai Super Kings': "C:\\Users\\rsaad\\PycharmProjects\\ipl_win_predictor\\csk.png",
    'Rajasthan Royals': 'C:\\Users\\rsaad\\PycharmProjects\\ipl_win_predictor\\rajasthan.png',
    'Delhi Capitals': 'C:\\Users\\rsaad\\PycharmProjects\\ipl_win_predictor\\delhi.png'
}

teams = list(team_logos.keys())

cities = [
    'Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
    'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
    'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
    'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
    'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
    'Sharjah', 'Mohali', 'Bengaluru'
]

# Title
st.markdown('<h1 style="text-align:center; color:#ef476f;">🏏 IPL Win Predictor</h1>', unsafe_allow_html=True)
st.markdown("---")

# Inputs
col1, col2 = st.columns(2)
with col1:
    batting_team = st.selectbox('🏏 Select Batting Team', sorted(teams))
with col2:
    bowling_team = st.selectbox('⚾ Select Bowling Team', sorted(teams))

selected_city = st.selectbox('📍 Match Location', sorted(cities))
target = st.number_input('🎯 Target Score', min_value=1)

col3, col4, col5 = st.columns(3)
with col3:
    score = st.number_input('✅ Current Score', min_value=0)
with col4:
    overs = st.number_input('⏱ Overs Completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets_fallen = st.number_input('❌ Wickets Lost', min_value=0, max_value=10)

# Predict Button
if st.button('Predict Probability'):
    if overs == 0 or overs > 20 or wickets_fallen > 10:
        st.warning("Please enter valid match data!")
    else:
        # Calculations
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        wickets = 10 - wickets_fallen
        crr = score / overs
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        # DataFrame
        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets],
            'total_runs_x': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # Prediction
        result = pipe.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]

        # Show logos
        col_logo1, col_vs, col_logo2 = st.columns([4, 1, 4])

        with col_logo1:
            st.image(team_logos[batting_team], width=150, caption=batting_team)

        with col_vs:
            st.markdown("<h3 style='text-align: center; margin-top: 50px;'>🆚</h3>", unsafe_allow_html=True)

        with col_logo2:
            st.image(team_logos[bowling_team], width=150, caption=bowling_team)

        # Prediction Results
        st.markdown("### 🧮 Win Probability")
        st.success(f"🔸 **{batting_team}** Win Chance: **{round(win*100, 2)}%**")
        st.error(f"🔹 **{bowling_team}** Win Chance: **{round(loss*100, 2)}%**")

        # Win Probability Graph
        fig, ax = plt.subplots(figsize=(6, 1.5))
        ax.barh([0], [win], color='#28a745', height=0.5, label=f"{batting_team}")
        ax.barh([0], [loss], left=[win], color='#dc3545', height=0.5, label=f"{bowling_team}")
        ax.set_xlim(0, 1)
        ax.axis('off')
        ax.legend(loc='center')
        st.pyplot(fig)

        # Commentary
        def get_commentary(prob):
            if prob > 0.8:
                return "🔥 Dominating performance! Almost a sure win!"
            elif prob > 0.6:
                return "🚀 In a strong position!"
            elif prob > 0.4:
                return "😬 It's anybody's game!"
            elif prob > 0.2:
                return "😓 Need a miracle now!"
            else:
                return "💀 Things look bleak..."

        st.info(f"📣 Commentary: {get_commentary(win)}")
