/*
 * Enhanced NFL Parlay Predictor
 *
 * This client-side script implements a parlay calculator that uses a
 * logistic regression model trained on multiple seasons of NFL data.  The
 * model parameters and per-team metrics are loaded from a JSON file
 * (enhanced_ratings.json) at runtime.  Metrics include:
 *   - rating: average point differential per game (points scored minus points allowed)
 *   - offense: average points scored per game
 *   - defense: negative average points allowed per game (so higher is better)
 *   - last3: average point differential over the last three games
 *
 * The logistic regression model combines these feature differences to
 * estimate the probability that one team beats another.  When more
 * coefficients are present than features (for example if the training
 * script included rest days or betting lines), the extra coefficients
 * multiply zero-valued features so that they do not affect the result.
 */

// Global variables for metrics and model parameters.  They will be
// populated asynchronously when the page loads.
let metrics = {};
let coefficients = [];
let interceptVal = 0;

// Fetch the enhanced metrics and logistic regression parameters.  The JSON
// file is expected to be located in the same directory as this script.
fetch('enhanced_ratings.json')
  .then((response) => response.json())
  .then((data) => {
    metrics = data.metrics || {};
    coefficients = data.coefficients || [];
    interceptVal = data.intercept || 0;
  })
  .catch((err) => {
    console.error('Failed to load enhanced ratings:', err);
  });

/**
 * Compute the win probability for teamA against teamB.
 *
 * @param {string} teamA - Abbreviation of the team you are betting on.
 * @param {string} teamB - Abbreviation of the opponent team.
 * @returns {number} Probability (0..1) that teamA wins.
 */
function predict(teamA, teamB) {
  const teamAData = metrics[teamA];
  const teamBData = metrics[teamB];
  if (!teamAData) {
    throw new Error(`Team ${teamA} is not recognized.`);
  }
  if (!teamBData) {
    throw new Error(`Team ${teamB} is not recognized.`);
  }
  // Compute feature differences
  const ratingDiff = (teamAData.rating || 0) - (teamBData.rating || 0);
  const offenseDiff = (teamAData.offense || 0) - (teamBData.offense || 0);
  const defenseDiff = (teamAData.defense || 0) - (teamBData.defense || 0);
  const last3Diff = (teamAData.last3 || 0) - (teamBData.last3 || 0);
  let feats = [ratingDiff, offenseDiff, defenseDiff, last3Diff];
  // Pad with zeros if there are more coefficients than features
  if (coefficients.length > feats.length) {
    feats = feats.concat(new Array(coefficients.length - feats.length).fill(0));
  }
  let z = interceptVal;
  for (let i = 0; i < coefficients.length; i++) {
    z += coefficients[i] * feats[i];
  }
  return 1.0 / (1.0 + Math.exp(-z));
}

/**
 * Dynamically add a new leg input row to the page.  Each leg consists
 * of two text inputs (betting team and opponent) and a remove button.
 */
function addLegRow() {
  const container = document.getElementById('legs-container');
  const legDiv = document.createElement('div');
  legDiv.className = 'leg';
  const inputA = document.createElement('input');
  inputA.type = 'text';
  inputA.placeholder = 'Betting team (e.g., IND)';
  const inputB = document.createElement('input');
  inputB.type = 'text';
  inputB.placeholder = 'Opponent (e.g., SEA)';
  const removeBtn = document.createElement('button');
  removeBtn.textContent = 'Remove';
  removeBtn.className = 'remove-leg';
  removeBtn.addEventListener('click', () => {
    container.removeChild(legDiv);
  });
  legDiv.appendChild(inputA);
  legDiv.appendChild(inputB);
  legDiv.appendChild(removeBtn);
  container.appendChild(legDiv);
}

/**
 * Gather input legs from the DOM, compute individual and combined
 * probabilities, and update the UI with the results.
 */
function calculate() {
  const legDivs = document.querySelectorAll('#legs-container .leg');
  const resultsDiv = document.getElementById('individual-results');
  const combinedP = document.getElementById('combined-result');
  const resultsSection = document.getElementById('results');
  resultsDiv.innerHTML = '';
  combinedP.textContent = '';
  let combinedProb = 1.0;
  let hadError = false;
  legDivs.forEach((div) => {
    const inputs = div.getElementsByTagName('input');
    const teamA = inputs[0].value.trim().toUpperCase();
    const teamB = inputs[1].value.trim().toUpperCase();
    if (!teamA || !teamB) {
      return;
    }
    try {
      const p = predict(teamA, teamB);
      combinedProb *= p;
      const pElem = document.createElement('div');
      pElem.textContent = `${teamA} beats ${teamB}: ${(p * 100).toFixed(1)}%`;
      resultsDiv.appendChild(pElem);
    } catch (err) {
      const errElem = document.createElement('div');
      errElem.textContent = err.message;
      errElem.style.color = 'red';
      resultsDiv.appendChild(errElem);
      hadError = true;
    }
  });
  if (!hadError) {
    combinedP.textContent = `Combined probability: ${(combinedProb * 100).toFixed(1)}%`;
  }
  resultsSection.style.display = 'block';
}

// Expose addLegRow and calculate functions to the global scope so that
// they can be called from inline HTML event handlers.
window.addLegRow = addLegRow;
window.calculate = calculate;

// When the page loads, wire up the buttons and insert an initial leg row
document.addEventListener('DOMContentLoaded', () => {
  const addBtn = document.getElementById('add-leg');
  const calcBtn = document.getElementById('calculate');
  if (addBtn) {
    addBtn.addEventListener('click', addLegRow);
  }
  if (calcBtn) {
    calcBtn.addEventListener('click', calculate);
  }
  // Start with one empty leg for convenience
  addLegRow();
});