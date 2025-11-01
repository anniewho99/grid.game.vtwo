// LsPomdpPolicySession.js
// Port of ls_pomdp.py for browser use (no server needed)

// ---------------------------
// Parameters from your Python
// ---------------------------
const action_names = ['a_0', 'a_1', 'h_0', 'h_1'];
const obs_names    = ['s_0', 's_1', 'none'];
const mode_to_actions = { r_trap: ['a_0', 'a_1'], h_trap: ['h_0', 'h_1'] };
const actions_without_obs = ['h_0', 'h_1'];

const obs_to_idx = { 's_0': 0, 's_1': 1, 'none': 2 };
const action_to_idx = { 'a_0': 0, 'a_1': 1, 'h_0': 2, 'h_1': 3 };

// Transition T[s, a, s’] (shape: 4 x 4 x 4)
const T_learned = [
    [[0.174191, 0.347827, 0.256781, 0.221201],
    [0.170532, 0.542945, 0.21475 , 0.071774],
    [0.638933, 0.163651, 0.177468, 0.019949],
    [0.228321, 0.594894, 0.039867, 0.136918]],
    [[0.001243, 0.808477, 0.058532, 0.131748],
    [0.469527, 0.328452, 0.103145, 0.098876],
    [0.029327, 0.416205, 0.546775, 0.007693],
    [0.190312, 0.131439, 0.40792 , 0.270329]],
    [[0.129733, 0.09053 , 0.487412, 0.292325],
    [0.755561, 0.029041, 0.04959 , 0.165808],
    [0.522198, 0.190169, 0.157702, 0.129932],
    [0.015902, 0.006323, 0.027417, 0.950358]],
    [[0.177933, 0.176534, 0.639426, 0.006107],
    [0.000492, 0.000524, 0.026323, 0.972661],
    [0.128007, 0.192944, 0.017404, 0.661645],
    [0.052721, 0.03149 , 0.014964, 0.900825]]
];

// Observation O[s, a, z] (shape: 4 x 4 x 3)
const O_learned = [
    [[0.619529, 0.380471, 0.      ],
    [0.992263, 0.007737, 0.      ],
    [0.      , 0.      , 1.      ],
    [0.      , 0.      , 1.      ]],
    [[0.99264 , 0.00736 , 0.      ],
    [0.999087, 0.000913, 0.      ],
    [0.      , 0.      , 1.      ],
    [0.      , 0.      , 1.      ]],
    [[0.201602, 0.798398, 0.      ],
    [0.701872, 0.298128, 0.      ],
    [0.      , 0.      , 1.      ],
    [0.      , 0.      , 1.      ]],
    [[0.62139 , 0.37861 , 0.      ],
    [0.000829, 0.999171, 0.      ],
    [0.      , 0.      , 1.      ],
    [0.      , 0.      , 1.      ]]
];

// Initial belief π over 4 states
const pi_learned = [0.171832, 0.532675, 0.130354, 0.165139];

// Alpha-vectors (8 x 4) and their action mapping (each α → action id)
const alpha_vectors = [
    [358.133, 352.398, 361.647, 484.912],
    [399.85 , 393.018, 432.614, 445.57 ],
    [369.429, 374.63 , 392.287, 484.536],
    [372.838, 376.292, 392.062, 484.307],
    [356.376, 378.639, 454.267, 475.789],
    [355.38 , 397.875, 447.962, 468.629],
    [361.974, 406.267, 387.048, 458.937],
    [399.018, 397.984, 426.475, 440.163]
];
// Each entry says which action (by index in action_names) that α-vector corresponds to
const action_ids = [1, 0, 1, 1, 3, 3, 2, 0]; // → [a_1, a_0, a_1, a_1, h_1, h_1, h_0, a_0]

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
function safeDot(a, b) {
  let sum = 0.0, c = 0.0;
  for (let i = 0; i < a.length; i++) {
    const prod = a[i] * b[i];
    const y = prod - c;
    const t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  return sum;
}
function matVecMultiplyLeft(vec, mat /* shape: S x S’ for fixed action */) {
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

// Slices for T and O
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

// ---------
// The class
// ---------
export default class LsPomdpPolicySession {
  constructor() {
    this.action_names = action_names;
    this.obs_names = obs_names;
    this.mode_to_actions = mode_to_actions;
    this.actions_without_obs = actions_without_obs;
    this.action_to_idx = action_to_idx;
    this.obs_to_idx = obs_to_idx;

    this.alpha_vectors = alpha_vectors;
    this.action_ids = action_ids;

    this.belief = Float64Array.from(pi_learned);
    this.last_action = null;
    this.step_count = 0;
  }

  // α·b for each α, pick best among allowed actions (deterministic ties)
  chooseActionWithConstraints(allowed) {
    let bestVal = -Infinity;
    let bestAction = null;
    for (let i = 0; i < this.alpha_vectors.length; i++) {
      const v = safeDot(this.alpha_vectors[i], this.belief);
      const a_id = this.action_ids[i];
      const a_name = this.action_names[a_id];
      if (!allowed.includes(a_name)) continue;

      if (v > bestVal + EPS || (Math.abs(v - bestVal) <= EPS && bestAction === null)) {
        bestVal = v;
        bestAction = a_name;
      }
    }
    return bestAction;
  }

  // Belief update with Python-parity guard:
  // b’ = normalize( (b @ T[:,a,:]) * O[:,a,z] ) if z given; retry on underflow else uniform
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
      // retry with pred
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

  reset(initialBelief = null) {
    if (initialBelief && initialBelief.length === this.belief.length) {
      this.belief = Float64Array.from(initialBelief);
    } else {
      const S = this.belief.length;
      this.belief = new Float64Array(S).fill(1 / S);
    }
    this.last_action = null;
    this.step_count = 0;
  }

  start(mode) {
    const allowed = this.mode_to_actions[mode] ?? this.action_names;
    const a = this.chooseActionWithConstraints(allowed);
    this.last_action = a;
    this.step_count = 1;
    return [a, Array.from(this.belief)];
  }

  updateAndAct(obs_name, mode) {
    let obs_idx = null;
    if (this.last_action && !this.actions_without_obs.includes(this.last_action)) {
      if (obs_name != null && obs_name !== 'none') {
        obs_idx = this.obs_to_idx[obs_name];
      }
    }
    const a_idx = this.action_to_idx[this.last_action];
    this.updateBelief(a_idx, obs_idx);

    const allowed = this.mode_to_actions[mode] ?? this.action_names;
    const next_action = this.chooseActionWithConstraints(allowed);

    this.last_action = next_action;
    this.step_count += 1;
    return [next_action, Array.from(this.belief)];
  }
}
