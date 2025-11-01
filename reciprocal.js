// ReciprocalReactiveSession.js
// JS port of reactive.py (rule-based reciprocal policy)

// ---------------------------
// Minimal params from Python
// ---------------------------
const action_names = ['a_0', 'a_1', 'h_0', 'h_1'];
const obs_names    = ['s_0', 's_1', 'none'];
const mode_to_actions = { r_trap: ['a_0', 'a_1'], h_trap: ['h_0', 'h_1'] };

// (kept only for completeness; belief isn't updated in this policy)
const pi_learned = [0.171832, 0.532675, 0.130354, 0.165139];

export default class ReciprocalReactiveSession {
  constructor() {
    this.action_names = action_names;
    this.obs_names = obs_names;
    this.mode_to_actions = mode_to_actions;

    this.belief = Float64Array.from(pi_learned); // copied to mirror Python
    this.last_action = null;
    this.last_obs = null;

    // (observation, mode) -> action mapping (exactly as in Python)
    this.mapping = {
      's_0|r_trap': 'a_1',
      's_1|r_trap': 'a_0',
      's_0|h_trap': 'h_0',
      's_1|h_trap': 'h_1',
    };
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

    // Fallback: first allowed action for the mode (mirrors Python)
    if (!action) {
      const allowed = this.mode_to_actions[mode] || [];
      action = allowed[0] || this.action_names[0];
    }

    this.last_action = action;
    // Belief stays the same (no update in the reactive version)
    return [action, Array.from(this.belief)];
  }
}
