"""
Parlay Predictor
================

This module implements a simple model to estimate the probability of individual
NFL game outcomes and evaluate parlays composed of multiple game legs.

The underlying data powering the model comes from the public
`nfldata` repository (https://github.com/nflverse/nfldata).  A copy of the
`games.csv` file should be downloaded into the project directory (the root of
this repository includes a `games.csv` file saved from the nflverse data
release).  If you do not have this file, follow the instructions in the
README to obtain it before running this script.

The model uses a very simple rating scheme: for each team in a given season
we calculate the average point differential per game (points scored minus
points allowed).  The probability of team A beating team B is then derived
from the difference in their ratings using a logistic function.  While this
approach is far from state of the art, it provides a transparent baseline
that is easy to understand and can be improved upon later.

Example usage from the command line:

    python parlay_predictor.py --season 2025 --legs "IND@SEA,LAR@DET"

The script prints the probability of each leg winning and the combined
probability of all legs hitting (assuming independence).

Author: ChatGPT for demonstration purposes
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Iterable, List, Tuple

import pandas as pd


def load_games(csv_path: str) -> pd.DataFrame:
    """Load the NFL games dataset from a CSV file.

    The expected file should have the columns defined in the nflverse
    `games.csv` file, including `season`, `game_type`, `home_team`,
    `home_score`, `away_team` and `away_score`.

    Parameters
    ----------
    csv_path: str
        Path to the games CSV file.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the games data.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Games file not found at {csv_path}. Please download games.csv from the nflverse"
        )
    df = pd.read_csv(csv_path)
    return df


def compute_team_ratings(df: pd.DataFrame, season: int) -> Dict[str, float]:
    """Compute per–team ratings for a given season.

    Each team's rating is defined as the average point differential per game:
    (total points scored − total points allowed) divided by the number of games
    played.  Only regular-season games (game_type == 'REG') are included.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame of all NFL games.
    season: int
        The season to compute ratings for.

    Returns
    -------
    Dict[str, float]
        A dictionary mapping team abbreviations to their rating.
    """
    season_df = df[(df['season'] == season) & (df['game_type'] == 'REG')]
    teams: set[str] = set(season_df['home_team']).union(set(season_df['away_team']))
    ratings: Dict[str, float] = {}
    for team in teams:
        home_games = season_df[season_df['home_team'] == team]
        away_games = season_df[season_df['away_team'] == team]
        pts_scored = home_games['home_score'].sum() + away_games['away_score'].sum()
        pts_allowed = home_games['away_score'].sum() + away_games['home_score'].sum()
        total_games = len(home_games) + len(away_games)
        rating = 0.0
        if total_games > 0:
            rating = (pts_scored - pts_allowed) / total_games
        ratings[team] = rating
    return ratings


def logistic(x: float, k: float = 0.25) -> float:
    """Compute the logistic function for a given input.

    The parameter `k` controls the steepness of the curve.  A higher value
    produces more confident predictions for a given rating difference.

    Parameters
    ----------
    x: float
        The input value (e.g., rating difference).
    k: float, optional
        Scale factor controlling the steepness; default is 0.25.

    Returns
    -------
    float
        The output of the logistic function, between 0 and 1.
    """
    return 1.0 / (1.0 + math.exp(-k * x))


def predict_game(team_a: str, team_b: str, ratings: Dict[str, float], k: float = 0.25) -> float:
    """Predict the probability that team_a beats team_b.

    The probability is computed by taking the difference between the teams'
    ratings and applying a logistic transformation.

    Parameters
    ----------
    team_a: str
        Abbreviation of the first team (the one you are betting on).
    team_b: str
        Abbreviation of the opponent team.
    ratings: Dict[str, float]
        Dictionary of team ratings for the season.
    k: float, optional
        Logistic scaling factor; default 0.25.

    Returns
    -------
    float
        Estimated probability that team_a wins.
    """
    if team_a not in ratings:
        raise ValueError(f"Team '{team_a}' not found in ratings for the season.")
    if team_b not in ratings:
        raise ValueError(f"Team '{team_b}' not found in ratings for the season.")
    diff = ratings[team_a] - ratings[team_b]
    return logistic(diff, k=k)


def evaluate_parlay(legs: Iterable[Tuple[str, str]], ratings: Dict[str, float], k: float = 0.25) -> Tuple[List[float], float]:
    """Evaluate a parlay composed of multiple game legs.

    Each leg is represented as a tuple `(team_a, team_b)`, corresponding to a
    bet that `team_a` beats `team_b`.  The function returns a list of
    individual probabilities and the combined probability (assumes
    independence between legs).

    Parameters
    ----------
    legs: Iterable[Tuple[str, str]]
        A sequence of (team_a, team_b) pairs.
    ratings: Dict[str, float]
        Team ratings for the season.
    k: float, optional
        Logistic scale factor.

    Returns
    -------
    Tuple[List[float], float]
        A list with each leg's predicted win probability and the product of
        these probabilities (combined probability).
    """
    probs: List[float] = []
    combined = 1.0
    for team_a, team_b in legs:
        p = predict_game(team_a, team_b, ratings, k=k)
        probs.append(p)
        combined *= p
    return probs, combined


def parse_legs(legs_str: str) -> List[Tuple[str, str]]:
    """Parse a comma-separated list of legs into (team_a, team_b) tuples.

    The expected format for each leg is "TEAM1@TEAM2" (e.g., "IND@SEA").
    Whitespace is ignored.  Team abbreviations are case-insensitive.

    Parameters
    ----------
    legs_str: str
        Comma-separated string of legs.

    Returns
    -------
    List[Tuple[str, str]]
        A list of (team_a, team_b) tuples.
    """
    legs: List[Tuple[str, str]] = []
    if not legs_str:
        return legs
    for leg in legs_str.split(','):
        leg = leg.strip()
        if not leg:
            continue
        if '@' not in leg:
            raise ValueError(f"Invalid leg format '{leg}'. Expected TEAM@OPP.")
        team_a, team_b = leg.split('@', 1)
        legs.append((team_a.strip().upper(), team_b.strip().upper()))
    return legs


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate NFL parlays using a simple logistic model.")
    parser.add_argument(
        '--csv',
        type=str,
        default=os.path.join(os.path.dirname(__file__), '..', 'games.csv'),
        help="Path to games.csv (downloaded from nflverse)."
    )
    parser.add_argument(
        '--season',
        type=int,
        default=2025,
        help="Season year for which to compute team ratings (e.g., 2025)."
    )
    parser.add_argument(
        '--legs',
        type=str,
        default='',
        help="Comma-separated legs of the form TEAM@OPP (e.g., 'IND@SEA,KC@DEN')."
    )
    parser.add_argument(
        '--k',
        type=float,
        default=0.25,
        help="Logistic scale factor controlling steepness."
    )
    args = parser.parse_args()

    # Load data and compute ratings
    df = load_games(args.csv)
    ratings = compute_team_ratings(df, args.season)

    # If no legs were provided, show available teams and exit
    if not args.legs:
        print(f"Computed ratings for {len(ratings)} teams in {args.season}.")
        print("Available teams (abbreviations):")
        print(', '.join(sorted(ratings.keys())))
        print("\nProvide legs with --legs TEAM@OPP to compute probabilities.")
        return

    try:
        legs = parse_legs(args.legs)
    except ValueError as e:
        print(f"Error parsing legs: {e}")
        return

    if not legs:
        print("No valid legs provided.")
        return

    probs, combined = evaluate_parlay(legs, ratings, k=args.k)
    for (team_a, team_b), p in zip(legs, probs):
        print(f"Probability {team_a} beats {team_b}: {p:.3f}")
    print(f"Combined parlay probability: {combined:.3f}")


if __name__ == '__main__':
    main()