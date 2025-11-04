// ReciprocalReactiveSession.js
// JS port of reactive (with belief update) from reactive (1).py

// ---------------------------
// Parameters (copied from Python)
// ---------------------------
const action_names = ['a_0', 'a_1', 'h_0', 'h_1'];
const obs_names    = ['s_0', 's_1', 'none'];
const mode_to_actions = { r_trap: ['a_0', 'a_1'], h_trap: ['h_0', 'h_1'] };
const actions_without_obs = ['h_0', 'h_1'];
const obs_to_idx = { 's_0': 0, 's_1': 1, 'none': 2 };
const action_to_idx = { 'a_0': 0, 'a_1': 1, 'h_0': 2, 'h_1': 3 };

// T[s, a, s’]  (4 x 4 x 4)
const T_learned = [
  [ [0.174191,0.347827,0.256781,0.221201], [0.170532,0.542945,0.21475 ,0.071774], [0.638933,0.163651,0.177468,0.019949], [0.228321,0.594894,0.039867,0.136918] ],
  [ [0.001243,0.808477,0.058532,0.131748], [0.469527,0.328452,0.103145,0.098876], [0.029327,0.416205,0.546775,0.007693], [0.190312,0.131439,0.40792 ,0.270329] ],
  [ [0.129733,0.09053 ,0.487412,0.292325], [0.755561,0.029041,0.04959 ,0.165808], [0.522198,0.190169,0.157702,0.129932], [0.015902,0.006323,0.027417,0.950358] ],
  [ [0.177933,0.176534,0.639426,0.006107], [0.000492,0.000524,0.026323,0.972661], [0.128007,0.192944,0.017404,0.661645], [0.052721,0.03149 ,0.014964,0.900825] ],
];

// O[s, a, z]  (4 x 4 x 3)
const O_learned = [
  [ [0.619529,0.380471,0.0], [0.992263,0.007737,0.0], [0.0,0.0,1.0], [0.0,0.0,1.0] ],
  [ [0.99264 ,0.00736 ,0.0], [0.999087,0.000913,0.0], [0.0,0.0,1.0], [0.0,0.0,1.0] ],
  [ [0.201602,0.798398,0.0], [0.701872,0.298128,0.0], [0.0,0.0,1.0], [0.0,0.0,1.0] ],
  [ [0.62139 ,0.37861 ,0.0], [0.000829,0.999171,0.0], [0.0,0.0,1.0], [0.0,0.0,1.0] ],
];

// π over 4 states
const pi_learned = [0.171832, 0.532675, 0.130354, 0.165139];

// --------------------
// Numeric stability helpers
// --------------------
const EPS = 1e-12, ONE = 1.0;

function sumKahan(x) {
  let s = 0.0, c = 0.0;
  for (let i = 0; i < x.length; i++) {
    const y = x[i] - c;
    const t = s + y;
    c = (t - s) - y;
    s = t;
  }
  return s;
}
function normalizeStable(p) {
  for (let i = 0; i < p.length; i++) if (p[i] < 0 && p[i] > -EPS) p[i] = 0;
  let z = sumKahan(p);
  if (!Number.isFinite(z) || z < EPS) {
    const u = ONE / p.length;
    for (let i = 0; i < p.length; i++) p[i] = u;
    return p;
  }
  const inv = ONE / z;
  for (let i = 0; i < p.length; i++) p[i] *= inv;
  return p;
}
function matVecMultiplyLeft(vec, mat /* shape: S x S’ */) {
  const S = mat[0].length;
  const out = new Float64Array(S);
  for (let s = 0; s < vec.length; s++) {
    const v = vec[s], row = mat[s];
    for (let sp = 0; sp < S; sp++) out[sp] += v * row[sp];
  }
  for (let i = 0; i < out.length; i++) if (out[i] < 0 && out[i] > -EPS) out[i] = 0;
  return out;
}
function transitionSlice(a) {
  const S = T_learned.length;
  const out = new Array(S);
  for (let s = 0; s < S; s++) out[s] = T_learned[s][a];
  return out;
}
function observationColumn(a, z) {
  const S = O_learned.length;
  const out = new Float64Array(S);
  for (let s = 0; s < S; s++) out[s] = O_learned[s][a][z];
  return out;
}

// --------------------
// Class
// --------------------
export default class ReciprocalReactiveSession {
  constructor() {
    this.action_names = action_names;
    this.obs_names = obs_names;
    this.mode_to_actions = mode_to_actions;
    this.actions_without_obs = actions_without_obs;
    this.obs_to_idx = obs_to_idx;
    this.action_to_idx = action_to_idx;

    this.T = T_learned;
    this.O = O_learned;

    this.belief = Float64Array.from(pi_learned);
    this.last_action = null;
    this.last_obs = null;

    // (observation, mode) -> action mapping (exact rule table)
    this.mapping = {
      's_0|r_trap': 'a_1',
      's_1|r_trap': 'a_0',
      's_0|h_trap': 'h_0',
      's_1|h_trap': 'h_1',
    };
  }

  updateBelief(action_idx, obs_idx /* may be null */) {
    const Tslice = transitionSlice(action_idx);
    const pred = matVecMultiplyLeft(this.belief, Tslice);

    let new_b;
    if (obs_idx == null) {
      new_b = pred;
    } else {
      const oz = observationColumn(action_idx, obs_idx);
      new_b = new Float64Array(pred.length);
      for (let i = 0; i < pred.length; i++) new_b[i] = pred[i] * oz[i];
    }

    // Retry guard (Python parity): if underflow, fall back to pred; else uniform
    let z = sumKahan(new_b);
    if (!Number.isFinite(z) || z < EPS) {
      const z2 = sumKahan(pred);
      if (!Number.isFinite(z2) || z2 < EPS) {
        this.belief = new Float64Array(pred.length).fill(1 / pred.length);
        return;
      }
      this.belief = normalizeStable(pred);
      return;
    }
    this.belief = normalizeStable(new_b);
  }

  reset(initialBelief = null) {
    if (initialBelief && initialBelief.length === this.belief.length) {
      this.belief = Float64Array.from(initialBelief);
    } else {
      const S = this.belief.length;
      this.belief = new Float64Array(S).fill(1 / S);
    }
    this.last_action = null;
    this.last_obs = null;
  }

  start(mode) {
    // Python default: 'a_0' if r_trap else 'h_0'
    const action = (mode === 'r_trap') ? 'a_0' : 'h_0';
    this.last_action = action;
    return [action, Array.from(this.belief)];
  }

  updateAndAct(obs_name, mode) {
    // Handle missing obs like Python: if None/'none', reuse last_obs or default 's_0'
    let obs = (obs_name == null || obs_name === 'none')
      ? (this.last_obs || 's_0')
      : obs_name;
    this.last_obs = obs;

    // Lookup rule
    const key = `${obs}|${mode}`;
    let action = this.mapping[key];

    // Fallback: first allowed action for the mode
    if (!action) {
      const allowed = this.mode_to_actions[mode] || [];
      action = allowed[0] || this.action_names[0];
    }

    this.last_action = action;

    // Belief update (new lines in Python version)
    const a_idx = this.action_to_idx[action];
    const obs_idx = (obs === 'none') ? null : this.obs_to_idx[obs];
    this.updateBelief(a_idx, obs_idx);

    return [action, Array.from(this.belief)];
  }
}
