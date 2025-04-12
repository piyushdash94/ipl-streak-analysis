# IPL Playoff Streak Analysis 📊🏏

This Streamlit app analyzes how **win streaks** impact IPL playoff qualification. It uses both:
- A **Pure Math Model** (50% win rate, combinatorics)
- A **Simulation Model** with realistic team strengths (40%, 50%, 60%)

## 🔍 Features

- 🧮 Explanation of IPL Point System
- 🔁 Definition and simulation of win streaks
- 📊 Comparison of Math vs Simulation qualification chances
- 📘 Real case study: **RCB 2023** (7W-7L, missed playoffs on NRR)
- 📈 Line graph of actual team momentum from 2020–2024
- ✅ Final insights with real-world implications

## 📸 Screenshots

> Replace these with your own screenshots after running the app

![Streak Distribution](images/streak_distribution.png)  
![Qualification Models](images/model_comparison.png)  
![Team Momentum](images/team_momentum.png)

## 🚀 How to Run

```bash
pip install -r requirements.txt
streamlit run ipl_streamlit_complete_app.py
```

## 🧠 Insights

- Teams need **at least one 3+ match win streak** to qualify comfortably
- Simulations show only ~40% teams naturally achieve this
- Math models underestimate how difficult qualification is under streak constraints

---

## 🔗 License
MIT License – feel free to fork, modify, and share!