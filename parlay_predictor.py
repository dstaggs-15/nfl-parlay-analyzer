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


def predict_game_advanced(
    team_a: str,
    team_b: str,
    metrics: Dict[str, Dict[str, float]],
    coeffs: List[float],
    intercept: float,
) -> float:
    """Predict the probability that ``team_a`` beats ``team_b`` using the advanced model.

    This function implements a logistic regression model that uses three
    handcrafted features derived from offensive and defensive ratings:

    * ``rating_diff`` – the difference in total team strength, defined as
      ``(offense_rating + defense_rating)`` for the two teams.
    * ``offense_diff`` – the difference between team A's offense and team B's
      defensive rating.
    * ``defense_diff`` – the difference between team A's defensive rating and
      team B's offensive rating.

    The provided coefficients and intercept should match this feature vector in
    length.  If the coefficient list is longer than three elements, any
    additional features will be ignored (to maintain backward compatibility
    with older JSON files).

    Parameters
    ----------
    team_a: str
        Abbreviation of the first team (the team you're betting on).
    team_b: str
        Abbreviation of the opponent team.
    metrics: Dict[str, Dict[str, float]]
        Per-team metrics containing at least ``offense_rating`` and
        ``defense_rating`` for each team.
    coeffs: List[float]
        Logistic regression coefficients.  Only the first three are used
        by this function.
    intercept: float
        Logistic regression intercept.

    Returns
    -------
    float
        Estimated probability that team_a wins the matchup.
    """
    if team_a not in metrics:
        raise ValueError(f"Team '{team_a}' not found in metrics.")
    if team_b not in metrics:
        raise ValueError(f"Team '{team_b}' not found in metrics.")
    teamA = metrics[team_a]
    teamB = metrics[team_b]
    # Construct the feature vector
    rating_diff = (teamA['offense_rating'] + teamA['defense_rating']) - (
        teamB['offense_rating'] + teamB['defense_rating']
    )
    offense_diff = teamA['offense_rating'] - teamB['defense_rating']
    defense_diff = teamA['defense_rating'] - teamB['offense_rating']
    features = [rating_diff, offense_diff, defense_diff]
    # Use only as many coefficients as we have features
    coeffs_trim = coeffs[: len(features)]
    z = intercept + sum(c * f for c, f in zip(coeffs_trim, features))
    prob = 1.0 / (1.0 + math.exp(-z))
    return prob


def predict_game_enhanced(
    team_a: str,
    team_b: str,
    metrics: Dict[str, Dict[str, float]],
    coeffs: List[float],
    intercept: float,
) -> float:
    """Predict the probability that ``team_a`` beats ``team_b`` using the enhanced model.

    The enhanced model extends the advanced model by incorporating additional
    per-team statistics beyond offense and defense ratings.  By default it
    considers the following four features:

    * ``rating`` – point differential per game for the season.
    * ``offense`` – average points scored per game.
    * ``defense`` – negative average points allowed per game (so higher is better).
    * ``last3`` – average point differential over the team's last three games.

    The prediction is computed by taking the difference of these metrics between
    team A and team B.  If the list of coefficients has more than four
    elements, any additional coefficients are applied to zero-valued features,
    allowing compatibility with models trained on more features (e.g., rest
    days, spread lines, moneylines, etc.) without requiring those inputs at
    prediction time.  This means that the extra features simply contribute
    nothing (because their values are assumed zero) but the intercept and
    existing coefficients are still used.

    Parameters
    ----------
    team_a: str
        Abbreviation of the team you are betting on.
    team_b: str
        Abbreviation of the opposing team.
    metrics: Dict[str, Dict[str, float]]
        Dictionary mapping team abbreviations to their per-team metrics.  Each
        metric dict must include ``rating``, ``offense``, ``defense`` and
        ``last3``.
    coeffs: List[float]
        Logistic regression coefficients.  The first four correspond to the
        feature differences described above.  Any additional coefficients are
        ignored unless you pass in explicit values for those features (which
        this function does not currently do).
    intercept: float
        Logistic regression intercept.

    Returns
    -------
    float
        Estimated probability that ``team_a`` wins.
    """
    if team_a not in metrics:
        raise ValueError(f"Team '{team_a}' not found in metrics.")
    if team_b not in metrics:
        raise ValueError(f"Team '{team_b}' not found in metrics.")
    mA = metrics[team_a]
    mB = metrics[team_b]
    # Basic features: rating, offense, defense, last3 differences
    features = [
        mA.get('rating', 0) - mB.get('rating', 0),
        mA.get('offense', 0) - mB.get('offense', 0),
        mA.get('defense', 0) - mB.get('defense', 0),
        mA.get('last3', 0) - mB.get('last3', 0),
    ]
    # Pad feature vector with zeros if the coefficient list is longer
    if len(coeffs) > len(features):
        features += [0.0] * (len(coeffs) - len(features))
    z = intercept + sum(c * f for c, f in zip(coeffs, features))
    return 1.0 / (1.0 + math.exp(-z))


def evaluate_parlay(
    legs: Iterable[Tuple[str, str]],
    ratings: Dict[str, float] | None = None,
    k: float = 0.25,
    *,
    advanced: bool = False,
    enhanced: bool = False,
    metrics: Dict[str, Dict[str, float]] | None = None,
    coeffs: List[float] | None = None,
    intercept: float | None = None,
) -> Tuple[List[float], float]:
    """Evaluate a parlay composed of multiple game legs.

    Each leg is represented as a tuple `(team_a, team_b)`, corresponding to a
    bet that `team_a` beats `team_b`.  The function returns a list of
    individual probabilities and the combined probability (assumes
    independence between legs).

    Parameters
    ----------
    legs: Iterable[Tuple[str, str]]
        A sequence of (team_a, team_b) pairs representing each leg of the parlay.
    ratings: Dict[str, float], optional
        Team ratings for the simple model (required if neither ``advanced`` nor
        ``enhanced`` is True).
    k: float, optional
        Logistic scale factor for the simple model; default 0.25.
    advanced: bool, optional
        Whether to use the advanced model based on offensive/defensive ratings.  If
        set, you must also provide ``metrics``, ``coeffs`` and ``intercept``.
    enhanced: bool, optional
        Whether to use the enhanced model which operates on per-team metrics
        (rating, offense, defense, last3) and logistic regression parameters.  As
        with ``advanced``, you must supply ``metrics``, ``coeffs`` and ``intercept``.
    metrics: Dict[str, Dict[str, float]], optional
        Per-team metrics for the advanced or enhanced model.
    coeffs: List[float], optional
        Coefficients for the logistic regression in the advanced or enhanced model.
    intercept: float, optional
        Intercept term for the advanced or enhanced model.

    Returns
    -------
    Tuple[List[float], float]
        A list with each leg's predicted win probability and the product of
        these probabilities (combined probability).
    """
    probs: List[float] = []
    combined = 1.0
    for team_a, team_b in legs:
        if enhanced:
            if metrics is None or coeffs is None or intercept is None:
                raise ValueError(
                    "Enhanced model selected but metrics/coeffs/intercept not provided."
                )
            p = predict_game_enhanced(team_a, team_b, metrics, coeffs, intercept)
        elif advanced:
            if metrics is None or coeffs is None or intercept is None:
                raise ValueError(
                    "Advanced model selected but metrics/coeffs/intercept not provided."
                )
            p = predict_game_advanced(team_a, team_b, metrics, coeffs, intercept)
        else:
            if ratings is None:
                raise ValueError("Simple model selected but ratings are not provided.")
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


def load_advanced_metrics(path: str) -> Tuple[Dict[str, Dict[str, float]], List[float], float]:
    """Load advanced team metrics and logistic regression parameters from JSON.

    The JSON file must contain a top-level object with keys:
    ``metrics`` (a dict of per-team metric dicts), ``coefficients`` (a list of floats),
    and ``intercept`` (a float).

    Parameters
    ----------
    path: str
        Path to the JSON file with advanced model parameters.

    Returns
    -------
    Tuple[metrics, coefficients, intercept]
        ``metrics`` is a dict mapping team abbreviations to metric dicts,
        ``coefficients`` is a list of floats, and ``intercept`` is a float.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Advanced metrics file not found at {path}")
    with open(path, 'r') as f:
        data = json.load(f)
    metrics = data.get('metrics')
    coeffs = data.get('coefficients')
    intercept = data.get('intercept')
    if metrics is None or coeffs is None or intercept is None:
        raise ValueError("Advanced metrics file missing required keys: 'metrics', 'coefficients', 'intercept'")
    return metrics, coeffs, float(intercept)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate NFL parlays using either a simple rating-based model or "
            "an advanced logistic regression model trained on offensive and defensive metrics."
        )
    )
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

    parser.add_argument(
        '--model',
        choices=['simple', 'advanced', 'enhanced'],
        default='simple',
        help=(
            "Which model to use for predictions: 'simple' uses average point differential per game; "
            "'advanced' uses a logistic regression trained on offensive/defensive ratings; and "
            "'enhanced' uses a logistic regression trained on per-team metrics (rating, offense, defense, last3)."
        ),
    )
    parser.add_argument(
        '--advanced-json',
        type=str,
        default=os.path.join(os.path.dirname(__file__), '..', 'advanced_ratings.json'),
        help=(
            "Path to a JSON file containing advanced metrics and model parameters. "
            "Used only when --model=advanced."
        ),
    )
    parser.add_argument(
        '--enhanced-json',
        type=str,
        default=os.path.join(os.path.dirname(__file__), '..', 'enhanced_ratings.json'),
        help=(
            "Path to a JSON file containing enhanced metrics and model parameters. "
            "Used only when --model=enhanced."
        ),
    )
    args = parser.parse_args()

    # Load data and compute ratings
    df = load_games(args.csv)
    ratings = compute_team_ratings(df, args.season)

    # If using advanced or enhanced models, load metrics and model parameters
    metrics: Dict[str, Dict[str, float]] | None = None
    coeffs: List[float] | None = None
    intercept: float | None = None
    if args.model in ('advanced', 'enhanced'):
        json_path = args.advanced_json if args.model == 'advanced' else args.enhanced_json
        try:
            metrics, coeffs, intercept = load_advanced_metrics(json_path)
        except Exception as e:
            print(f"Error loading {args.model} metrics: {e}")
            return

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

    # Evaluate parlay using selected model
    if args.model == 'enhanced':
        probs, combined = evaluate_parlay(
            legs,
            ratings=None,
            k=args.k,
            enhanced=True,
            metrics=metrics,
            coeffs=coeffs,
            intercept=intercept,
        )
    elif args.model == 'advanced':
        probs, combined = evaluate_parlay(
            legs,
            ratings=None,
            k=args.k,
            advanced=True,
            metrics=metrics,
            coeffs=coeffs,
            intercept=intercept,
        )
    else:
        probs, combined = evaluate_parlay(
            legs,
            ratings=ratings,
            k=args.k,
            advanced=False,
        )

    for (team_a, team_b), p in zip(legs, probs):
        print(f"Probability {team_a} beats {team_b}: {p:.3f}")
    print(f"Combined parlay probability: {combined:.3f}")


if __name__ == '__main__':
    main()