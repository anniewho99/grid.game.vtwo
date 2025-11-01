// GreedyPolicySession.js
// JS port of myopic_greedy.py (one-step lookahead policy)

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

// Reward table R[s, a]
const R_table = [
  [  0.001   , -14.999   ,   0.001   , -14.999   ],
  [  2.935541, -12.064459,   2.935541, -12.064459],
  [ 13.910459,  -1.089541,  13.910459,  -1.089541],
  [ 40.822374,  25.822374,  40.822374,  25.822374],
];

// r_next (length 4 over next state)
const r_next = [0.001, 2.935541, 13.910459, 40.822374];

// --------------------
// Numeric stability
// --------------------
const EPS = 1e-12;
const ONE = 1.0;

function sumKahan(x) {
  let sum = 0.0, c = 0.0;
  for (let i = 0; i < x.length; i++) {
    const y = x[i] - c;
    const t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  return sum;
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
    const v = vec[s];
    const row = mat[s];
    for (let sp = 0; sp < S; sp++) out[sp] += v * row[sp];
  }
  for (let i = 0; i < S; i++) if (out[i] < 0 && out[i] > -EPS) out[i] = 0;
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
// Greedy (one-step) policy
// --------------------
export default class GreedyPolicySession {
  constructor() {
    this.T = T_learned;
    this.O = O_learned;
    this.R = R_table;
    this.action_names = action_names;
    this.obs_to_idx = obs_to_idx;
    this.action_to_idx = action_to_idx;
    this.mode_to_actions = mode_to_actions;
    this.actions_without_obs = actions_without_obs;
    this.r_next = r_next;

    this.belief = Float64Array.from(pi_learned);
    this.last_action = null;
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

    let z = sumKahan(new_b);
    if (!Number.isFinite(z) || z < EPS) {
      let z2 = sumKahan(pred);
      if (!Number.isFinite(z2) || z2 < EPS) {
        const u = ONE / pred.length;
        this.belief = new Float64Array(pred.length).fill(u);
        return;
      }
      this.belief = normalizeStable(pred);
      return;
    }
    this.belief = normalizeStable(new_b);
  }

  // One-step lookahead:
  // Q[a] = (b @ R[:,a]) + ((b @ T[:,a,:]) @ r_next)
  oneStepLookahead(allowed_actions) {
    const b = this.belief;
    const A = this.action_names.length;

    // imm[a] = b @ R[:,a]
    const imm = new Float64Array(A);
    for (let a = 0; a < A; a++) {
      let ssum = 0.0, c = 0.0;
      for (let s = 0; s < b.length; s++) {
        const prod = b[s] * this.R[s][a];
        const y = prod - c;
        const t = ssum + y;
        c = (t - ssum) - y;
        ssum = t;
      }
      imm[a] = ssum;
    }

    // predNext[a, s’] = (b @ T[:,a,:])
    const look = new Float64Array(A);
    for (let a = 0; a < A; a++) {
      const Tslice = transitionSlice(a); // S x S’
      const predNext = matVecMultiplyLeft(b, Tslice);
      // dot with r_next
      let dsum = 0.0, c2 = 0.0;
      for (let sp = 0; sp < predNext.length; sp++) {
        const prod = predNext[sp] * this.r_next[sp];
        const y = prod - c2;
        const t = dsum + y;
        c2 = (t - dsum) - y;
        dsum = t;
      }
      look[a] = dsum;
    }

    // Q = imm + look
    const Q = new Float64Array(A);
    for (let a = 0; a < A; a++) Q[a] = imm[a] + look[a];

    // pick best among allowed actions
    const a_to_idx = new Map(this.action_names.map((a, i) => [a, i]));
    const allowed_idx = allowed_actions.map(a => a_to_idx.get(a));

    let bestIdx = allowed_idx[0];
    let bestVal = Q[bestIdx];
    for (let i = 1; i < allowed_idx.length; i++) {
      const idx = allowed_idx[i];
      const v = Q[idx];
      if (v > bestVal + EPS) {
        bestVal = v;
        bestIdx = idx;
      }
    }
    return [this.action_names[bestIdx], Array.from(Q)];
  }

  reset(initialBelief = null) {
    if (initialBelief && initialBelief.length === this.belief.length) {
      this.belief = Float64Array.from(initialBelief);
    } else {
      const S = this.belief.length;
      this.belief = new Float64Array(S).fill(1 / S);
    }
    this.last_action = null;
  }

  start(mode) {
    const allowed = this.mode_to_actions[mode] ?? this.action_names;
    const [best_action, Q] = this.oneStepLookahead(allowed);
    this.last_action = best_action;
    return [best_action, Array.from(this.belief), Q];
  }

  updateAndAct(obs_name, mode) {
    if (this.last_action == null) {
      throw new Error("Call start(mode) before updateAndAct().");
    }

    let obs_idx = null;
    if (this.last_action && !this.actions_without_obs.includes(this.last_action)) {
      if (obs_name != null && obs_name !== 'none') {
        obs_idx = obs_to_idx[obs_name];
      }
    }

    const a_idx = this.action_to_idx[this.last_action];
    this.updateBelief(a_idx, obs_idx);

    const allowed = this.mode_to_actions[mode] ?? this.action_names;
    const [next_action, Q] = this.oneStepLookahead(allowed);
    this.last_action = next_action;
    return [next_action, Array.from(this.belief), Q];
  }
}
