# NFL Parlay Probability Calculator

This project provides a **simple, transparent tool** for estimating how likely a given NFL parlay is to hit.  It comprises:

* A Python script (`parlay_predictor.py`) that uses historical game results to compute per‑team ratings and apply a logistic model to estimate the probability of one team beating another.
* A lightweight web interface (`web/index.html`) that runs entirely in the browser, allowing anyone to enter a list of matchups and instantly see estimated probabilities.  Because it is a static page, it can be deployed with GitHub Pages and used from a mobile phone without any backend server.

The underlying data come from the **nflverse** project.  The nflverse stores data for NFL analytics in its `nflverse-data` repository and makes it available in a variety of formats that can be downloaded directly【15544497006062†L186-L191】.  The Python package `nfl_data_py` also interfaces with these datasets【866195987737861†L108-L118】, but for this project we simply downloaded the `games.csv` file from the `nfldata` repository and stored it in this repository.

> **Disclaimer:** This model is extremely basic.  It relies only on each team's average point differential per game and assumes independence across legs of a parlay.  It does **not** account for injuries, weather, betting lines, player matchups or any of the rich contextual information that professional bookmakers and modelers use.  It is provided for entertainment and educational purposes only.  No warranty is expressed or implied.

## Getting Started (Python CLI)

1. **Download the data.**  The repository already includes `games.csv`, which contains every NFL game from 1999 through 2025.  If you wish to refresh the data, download the latest `games.csv` from the `nfldata` repository on GitHub and replace the file in the root of this project.
2. **Install dependencies.**  The script only requires `pandas` (for CSV parsing).  You can install it with:

   ```bash
   pip install pandas
   ```

3. **Run the predictor.**  From within the `nfl_parlay_analyzer` directory, invoke the script with the desired season and legs:

   ```bash
   python parlay_predictor.py --season 2025 --legs "IND@SEA,KC@DEN,BUF@GB"
   ```

   This will output each leg's win probability and the combined probability assuming independence.  If you omit `--legs`, the script lists all available team abbreviations for the chosen season.

### How the model works

* The script loads the `games.csv` file and filters it to the specified season and regular‑season games.
* It computes a **rating** for each team: `(total points scored – total points allowed) / number of games`.  A positive rating means the team scores more than it allows on average.
* To estimate the probability that Team A beats Team B, the difference in their ratings is passed through a logistic function: `p = 1 / (1 + exp(-k * (rating_A - rating_B)))`.  The scale parameter `k` controls how steeply the function converts rating differences into probabilities (the default, `k = 0.25`, corresponds to a difference of 10 rating points yielding a probability around 92 %).
* A parlay is evaluated by multiplying the probabilities of its individual legs.  This multiplication assumes the events are independent – an assumption that will often be false in practice but serves as a baseline.

## Using the Web Interface

The `web` directory contains a fully self‑contained static site.  To try it locally:

1. Open `nfl_parlay_analyzer/web/index.html` in your browser.  (No server is required.)
2. Click **“Add Leg”** to add as many matchups as you like.  For each leg, enter the betting team (the one you think will win) and its opponent using official three‑letter abbreviations (e.g., `IND` for Indianapolis Colts).
3. Click **“Calculate”**.  The page shows the probability of each individual outcome and the combined parlay probability.  If an unrecognised team is entered, it displays an error for that leg.

### Deploy to GitHub Pages

To make the tool available online, you can publish the `web` folder using GitHub Pages:

1. Create a new repository on GitHub (e.g., `nfl-parlay-calculator`).
2. Copy the contents of the `web` directory into the root of that repository (or a `docs` folder if you prefer).  Commit and push.
3. In the repository settings, enable GitHub Pages and select the branch and folder where the site resides.  After a few minutes, your parlay calculator will be accessible at `https://<your-username>.github.io/<repo-name>/`.

Your friends can then visit the page from any device without installing anything.  Because all the logic runs in the browser using pre‑computed ratings, the site is responsive and light‑weight.

## Improving the model

This project deliberately takes a minimalist approach.  There are many avenues for enhancement:

* **Incorporate market data.**  Integrating closing betting lines, totals and spreads could help estimate probabilities that better reflect public sentiment.
* **Use richer statistics.**  Beyond point differential, incorporate yardage, turnovers, EPA, ELO ratings or drive‑level efficiency metrics.
* **Model dependencies.**  Parlays often contain correlated legs (e.g., both over/under and winner in the same game).  Accounting for correlation would provide more accurate combined probabilities.
* **Update dynamically.**  Ratings could be recalculated each week and hosted as a JSON feed that the web app fetches, ensuring the tool stays current without manual updates.

Feel free to fork this repository and experiment with your own models.  The code is intentionally straightforward to support tinkering.