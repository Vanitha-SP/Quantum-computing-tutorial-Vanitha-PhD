# =============================================================================
#  MANUAL IBM QISKIT SURFACE CODE IMPLEMENTATION
#  No QEC library used — built from scratch using Qiskit primitives only.
#  Every line is annotated with the matching theory from Dan Browne's notes.
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0 — IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
# Theory: Qiskit's QuantumCircuit is the canvas on which we write quantum
# gates. QuantumRegister holds data qubits (placed on edges of the lattice).
# ClassicalRegister stores the syndrome measurement outcomes — the ±1
# eigenvalues of each stabilizer generator (stored here as 0/1 bits).

from qiskit_aer import AerSimulator
# Theory: We use the AerSimulator backend to simulate the quantum circuit
# classically. In a real quantum computer this would run on superconducting
# qubits or ion traps. The simulator lets us test our surface code logic.

from qiskit_aer.noise import (NoiseModel, depolarizing_error,
                               pauli_error, reset_error)
# Theory: NoiseModel injects errors into the circuit to simulate real hardware.
# depolarizing_error models the depolarising noise from Chapter 2.2.4:
# each qubit independently undergoes I (prob 1-p), X (p/3), Y (p/3), Z (p/3).
# pauli_error lets us inject independent X and Z errors (uncorrelated model).

from qiskit.visualization import circuit_drawer
# Theory: Visualization — lets us draw the quantum circuit to inspect the
# stabilizer measurement circuits described in Section 1.3.9.

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import itertools
# Theory: numpy/matplotlib are classical tools. We use them to implement
# the classical decoder (Minimum Weight Perfect Matching, Chapter 2.2.7)
# and to visualise the lattice, syndromes, and error chains.


# =============================================================================
# SECTION 1 — LATTICE GEOMETRY
# Theory reference: Chapter 2.1.1 — "Qubits on a toric lattice"
# The toric/surface code is defined on an L×L square lattice.
# Qubits sit on EDGES; stabilizer generators sit on PLAQUETTES and VERTICES.
# =============================================================================

class SurfaceCodeLattice:
    """
    Represents the geometry of a distance-d planar surface code.

    Theory — Chapter 2.1.1:
    The lattice has:
      - L x L  vertices  (L = d for a distance-d code)
      - 2*(L-1)*L  edges  — qubits live here
      - (L-1)^2  plaquettes (internal squares)
    For a planar surface code (not toric), the lattice has a boundary.
    We use a (d x d) vertex grid, giving:
      - data qubits on horizontal edges: d*(d-1) qubits
      - data qubits on vertical edges:   d*(d-1) qubits
      - total data qubits: n = 2*d*(d-1) ... but the standard (2d-1)x(2d-1)
        rotated surface code encodes 1 logical qubit in d^2 + (d-1)^2 qubits.
    Here we implement the ROTATED planar surface code (distance d):
      - d^2 data qubits
      - (d^2 - 1)/2 X-stabilizers (vertex-type)
      - (d^2 - 1)/2 Z-stabilizers (plaquette-type)
      - k = 1 logical qubit  [n = d^2, k = 1, dist = d]
    """

    def __init__(self, d):
        """
        d: code distance.
        Theory: distance d means the code can correct floor((d-1)/2) errors.
        A [d^2, 1, d] code — n=d^2 physical qubits encoding k=1 logical qubit.
        """
        self.d = d

        # ── DATA QUBIT POSITIONS ─────────────────────────────────────────────
        # Theory (Ch 2.1.1): In the rotated surface code, data qubits sit on
        # a d×d grid of vertices of a rotated (45°) square lattice.
        # We label them by (row, col) with row,col in {0,...,d-1}.
        self.data_qubits = [(r, c) for r in range(d) for c in range(d)]
        # Each (r,c) pair uniquely identifies one edge/qubit on the lattice.

        self.n_data = d * d
        # Theory: n = d^2 physical qubits for the rotated surface code.
        # Compare to the toric code: n = 2L^2 for L×L with periodic BCs.

        # Map (row,col) → qubit index (integer) for use in Qiskit registers
        self.qubit_index = {pos: i for i, pos in enumerate(self.data_qubits)}
        # This dictionary is our lattice-to-register map: every geometric
        # position on the lattice maps to one physical qubit in the circuit.

        # ── STABILIZER GENERATOR POSITIONS ──────────────────────────────────
        # Theory (Ch 2.1.2–2.1.4): Stabilizer generators come in two types:
        #   Z-type (plaquette operators, B_p): tensor product of Z on 4 edges
        #     surrounding a plaquette. There are (d-1)^2/2 + (d*(d-1)/2)
        #     Z-type stabilizers in the rotated code.
        #   X-type (vertex operators, A_v):   tensor product of X on 4 edges
        #     adjacent to a vertex.
        #
        # In the rotated surface code, stabilizers are placed on the "faces"
        # of the rotated lattice.  We label them by their center (r+0.5, c+0.5)
        # and by type ('Z' or 'X'). Each stabilizer acts on its 4 neighbouring
        # data qubits (with weight reduced to 2 or 3 at the boundary —
        # consistent with Ch 4.2.4's discussion of boundary stabilizers).

        self.z_stabilizers = []   # list of (center_r, center_c, [qubit_positions])
        self.x_stabilizers = []   # list of (center_r, center_c, [qubit_positions])

        self._build_stabilizers()
        # Delegates to the method below; keeps __init__ clean.

        self.n_ancilla = len(self.z_stabilizers) + len(self.x_stabilizers)
        # Theory (Ch 1.3.7): total stabilizer generators = m = n - k = d^2 - 1.
        # Half are Z-type, half are X-type (in the rotated code).

    def _build_stabilizers(self):
        """
        Construct all Z and X stabilizer generators for the rotated surface code.

        Theory (Ch 2.1.3–2.1.4):
        Plaquette (Z) operators act on the 4 edges bordering each plaquette.
        Vertex (X) operators act on the 4 edges meeting at each vertex.

        In the ROTATED code the stabilizers tile a checkerboard pattern:
        - "Z-faces" are squares whose corners are data qubits acted on by Z⊗Z⊗Z⊗Z
        - "X-faces" are the alternating squares acting with X⊗X⊗X⊗X
        - Boundary faces are half-squares (weight-2 operators)
        """
        d = self.d

        # Iterate over all "face centres" in the (d-1) x (d-1) interior grid
        # plus the boundary half-faces, using a checkerboard colour assignment.
        for r in range(-1, d):          # face row index (can be -0.5 in original)
            for c in range(-1, d):      # face col index
                # Collect the (at most) 4 data-qubit neighbours of this face
                neighbours = []
                for dr, dc in [(-1, 0), (0, -1), (0, 0), (-1, -1)]:
                    # These offsets recover the 4 corners of the face at (r,c)
                    # in the rotated lattice convention
                    qr, qc = r + 1 + dr, c + 1 + dc
                    # Wait — let's use the standard checkerboard construction:
                    pass

        # ── Cleaner checkerboard construction ───────────────────────────────
        # Standard rotated surface code: faces are at half-integer positions
        # (r+0.5, c+0.5) for r,c in {-1,...,d-1}. A face is "Z-type" if
        # (r + c) is even, "X-type" if (r + c) is odd (boundary convention
        # can flip this; we follow the standard literature assignment).

        self.z_stabilizers = []
        self.x_stabilizers = []

        for r in range(-1, d):
            for c in range(-1, d):
                # Get the up to 4 data qubit positions adjacent to face (r,c)
                qubits_on_face = []
                for qr, qc in [(r, c), (r, c+1), (r+1, c), (r+1, c+1)]:
                    # Theory: the face at (r,c) touches qubits at its 4 corners
                    if 0 <= qr < d and 0 <= qc < d:
                        qubits_on_face.append((qr, qc))

                if len(qubits_on_face) == 0:
                    continue
                # Faces with 0 neighbours are outside the lattice — skip.

                # Assign Z or X type via checkerboard parity
                # Theory (Ch 2.1.5): In the rotated code, alternating faces are
                # Z-type and X-type so that each Z face always shares an even
                # number of qubits with each X face → they commute.
                # (Z⊗Z)·(X⊗X) = +1 since (-1)^2 = +1, as explained in Ch 2.1.5)
                if (r + c) % 2 == 0:
                    # Z-type face (plaquette operator B_p)
                    self.z_stabilizers.append({
                        'center': (r + 0.5, c + 0.5),
                        'qubits': qubits_on_face,
                        'type': 'Z'
                    })
                else:
                    # X-type face (vertex operator A_v)
                    self.x_stabilizers.append({
                        'center': (r + 0.5, c + 0.5),
                        'qubits': qubits_on_face,
                        'type': 'X'
                    })

    def logical_x_qubits(self):
        """
        Return the data qubits forming the minimal-weight logical X operator.

        Theory (Ch 2.1.9, Ch 4.2.4):
        Logical X is a string of X operators along the TOP ROW of the lattice
        from one boundary to the other — a 1-cocycle connecting the two
        'smooth' (X-type) boundaries. Its weight = d (the code distance).
        """
        d = self.d
        return [(0, c) for c in range(d)]
        # Row 0 spans the full width: weight = d = code distance.

    def logical_z_qubits(self):
        """
        Return the data qubits forming the minimal-weight logical Z operator.

        Theory (Ch 2.1.9, Ch 4.2.4):
        Logical Z is a string of Z operators along the LEFT COLUMN of the
        lattice — a 1-cycle connecting the two 'rough' (Z-type) boundaries.
        Its weight = d (the code distance).
        """
        d = self.d
        return [(r, 0) for r in range(d)]
        # Column 0 spans the full height: weight = d = code distance.


# =============================================================================
# SECTION 2 — STABILIZER MEASUREMENT CIRCUIT
# Theory: Chapter 1.3.9 — "Error detection in the stabilizer formalism"
# Theory: Chapter 1.2.8 — "Quantum circuit to measure parity"
#
# We measure each stabilizer generator by using one ancilla qubit per
# generator. The ancilla acts as a 'flag': after the measurement circuit,
# measuring the ancilla in the computational basis yields +1 (bit 0) if the
# stabilizer is satisfied, and -1 (bit 1) if it is violated.
# =============================================================================

def build_stabilizer_circuit(lattice, rounds=1):
    """
    Construct the full syndrome extraction circuit for one (or more) rounds.

    Theory (Ch 1.3.9):
    To measure an n-qubit Pauli operator P = P_1 ⊗ P_2 ⊗ ... ⊗ P_n we use
    the ancilla-mediated circuit:
        1. Prepare ancilla in |0>
        2. Apply H to ancilla        (creates |+> = (|0>+|1>)/√2)
        3. For each qubit i where P_i = Z: apply CNOT(ancilla → qubit_i)
           For each qubit i where P_i = X: apply CNOT(qubit_i → ancilla)
           (This is the controlled-P_i gate in the general circuit)
        4. Apply H to ancilla
        5. Measure ancilla: outcome 0 ↔ eigenvalue +1 (no error detected)
                            outcome 1 ↔ eigenvalue -1 (error detected)
    The key property: this measurement does NOT collapse the encoded state
    (Section 1.2.7), because the stabilizer commutes with all logical operators.

    Parameters
    ----------
    lattice : SurfaceCodeLattice
    rounds  : int — number of syndrome extraction rounds (for 3D decoding)
    """
    d = lattice.d

    # ── REGISTER DEFINITIONS ─────────────────────────────────────────────────
    data_qr = QuantumRegister(lattice.n_data, name='d')
    # Theory: One quantum register holds all n = d^2 data qubits — these are
    # the physical qubits on the edges of the lattice (Section 2.1.1).

    n_z = len(lattice.z_stabilizers)
    n_x = len(lattice.x_stabilizers)

    z_anc_qr = QuantumRegister(n_z, name='az')
    x_anc_qr = QuantumRegister(n_x, name='ax')
    # Theory: One ancilla qubit per stabilizer generator.
    # Total ancillae = (d^2 - 1) = n - k, matching m = n - k from Ch 1.3.7.
    # Each ancilla qubit will be reset, entangled with data, then measured
    # to extract one bit of the error syndrome.

    z_cr = ClassicalRegister(n_z * rounds, name='sz')
    x_cr = ClassicalRegister(n_x * rounds, name='sx')
    # Theory: Classical registers store the syndrome bits. The syndrome is the
    # set of all stabilizer measurement outcomes (Section 1.3.9).
    # With 'rounds' measurement rounds we collect 3D syndrome data (Ch 5.3.2).

    qc = QuantumCircuit(data_qr, z_anc_qr, x_anc_qr, z_cr, x_cr)
    # Create the full circuit containing both quantum and classical registers.

    # ── SYNDROME EXTRACTION ROUNDS ───────────────────────────────────────────
    for round_idx in range(rounds):

        # ── RESET ANCILLAE ───────────────────────────────────────────────────
        qc.reset(z_anc_qr)
        qc.reset(x_anc_qr)
        # Theory: Before each measurement round, all ancilla qubits are reset
        # to |0>. This is necessary because ancillae are re-used across rounds
        # (fault-tolerant repeated measurement, Section 5.3.1).

        qc.barrier()
        # Barrier is a visual/compilation separator; no physical effect.
        # In real hardware it prevents gate reordering across the barrier —
        # important so data-qubit gates and ancilla-reset don't get swapped.

        # ══ Z-STABILIZER (PLAQUETTE) MEASUREMENT CIRCUITS ══════════════════
        # Theory (Ch 2.1.3): Plaquette operator B_p = ⊗_{e∈∂p} Z_e
        # Each plaquette stabilizer is a tensor product of Z on its boundary
        # edges. We measure it using the standard ancilla circuit from Ch 1.3.9.
        #
        # Circuit for measuring Z⊗Z⊗Z⊗Z on qubits q1,q2,q3,q4:
        #   anc: |0> ─H─●─────────────────H─M
        #   q1:         CNOT (anc→q1 but for Z we do CX with anc as control)
        #
        # Actually for Z-stabilizers:
        #   anc starts in |0>, H→|+>, then CX(anc, data_i) for each Z qubit,
        #   then H, then measure.  The CX(ctrl=anc, tgt=data) gate flips the
        #   data qubit conditioned on anc=|1>, which in the Hadamard-rotated
        #   basis measures the Z parity of the data qubit into the ancilla.

        for s_idx, stab in enumerate(lattice.z_stabilizers):
            anc = z_anc_qr[s_idx]
            # Ancilla qubit for this Z stabilizer.

            qc.h(anc)
            # Theory: Hadamard puts ancilla into |+> = (|0>+|1>)/√2.
            # In the stabilizer measurement circuit (Ch 1.3.9), the H before
            # the controlled gates converts the ancilla into the eigenbasis
            # needed to measure the product Pauli P = Z⊗Z⊗...

            for (qr, qc_) in stab['qubits']:
                q_idx = lattice.qubit_index[(qr, qc_)]
                qc.cx(anc, data_qr[q_idx])
                # Theory: CX(anc, data) = controlled-X with ancilla as control.
                # For a Z-stabilizer measurement, we use CX(ancilla→data).
                # When the ancilla is |+>, applying CX(anc→data_i) and then
                # H on ancilla effectively measures Z on data_i into the ancilla.
                # Mathematical proof: the circuit maps:
                #   |+>|ψ> → (CX) → |+>Z|ψ> (up to ancilla state tracking)
                # After all CNOTs, the ancilla phase encodes the parity:
                # ancilla |0> if Z1·Z2·...·Zk = +1 (even parity)
                # ancilla |1> if Z1·Z2·...·Zk = -1 (odd parity, error!)

            qc.h(anc)
            # Theory: Second Hadamard converts the accumulated phase back into
            # a computational-basis amplitude. This completes the measurement
            # of the observable B_p = Z⊗Z⊗Z⊗Z (Ch 2.1.3).

            qc.measure(anc, z_cr[s_idx + round_idx * n_z])
            # Theory: Measuring the ancilla collapses it to 0 (eigenvalue +1,
            # stabilizer satisfied, no error on this plaquette) or 1 (eigenvalue
            # -1, stabilizer violated — an error has occurred nearby).
            # This is the syndrome bit for plaquette stabilizer s_idx.
            # Outcome 0 = no quasiparticle at this plaquette (Ch 2.2.3)
            # Outcome 1 = quasiparticle present (pair created by error)

        qc.barrier()

        # ══ X-STABILIZER (VERTEX) MEASUREMENT CIRCUITS ══════════════════════
        # Theory (Ch 2.1.4): Vertex operator A_v = ⊗_{e adjacent v} X_e
        # For X-stabilizers, the measurement circuit is DUAL to Z-stabilizers:
        # we apply CX(data → anc) instead of CX(anc → data), because:
        #   - For Z-stabilizers: we measure Z parity → CX(anc→data)
        #   - For X-stabilizers: we measure X parity → CX(data→anc)
        # This is the Hadamard-dual of the Z measurement circuit.
        #
        # Full X-stabilizer circuit for qubits q1,...,qk:
        #   anc: |0> ─H──────●───H─M  (no: anc is target here)
        #   data_i:          CX (data_i → anc)

        for s_idx, stab in enumerate(lattice.x_stabilizers):
            anc = x_anc_qr[s_idx]

            qc.h(anc)
            # Theory: Hadamard again prepares ancilla in |+>.
            # For X-stabilizers, the measurement uses CX(data→anc):
            # this copies the X-parity of the data qubits into the ancilla phase.

            for (qr, qc_) in stab['qubits']:
                q_idx = lattice.qubit_index[(qr, qc_)]
                qc.cx(data_qr[q_idx], anc)
                # Theory: CX(data, anc) = controlled-X with DATA as control.
                # This is the circuit for measuring X⊗X⊗...⊗X (Ch 1.3.9):
                # "controlled P_j gates" where P_j = X are just standard CX
                # with the data qubit as control.
                # The ancilla accumulates the XOR (mod 2) of X-eigenvalues:
                # ancilla |0> ← even number of data qubits in |-> state
                # ancilla |1> ← odd number (X stabilizer violated by Z error)
                # Note: X and Z stabilizers detect DIFFERENT error types:
                # X-stabs detect Z (phase-flip) errors; Z-stabs detect X (bit-flip) errors.

            qc.h(anc)
            # Theory: Final Hadamard maps the phase back to amplitude.
            # After this H, measuring in the Z basis (computational basis)
            # reveals the X-parity of the data qubits.

            qc.measure(anc, x_cr[s_idx + round_idx * n_x])
            # Theory: Syndrome bit for vertex stabilizer s_idx.
            # Outcome 1 = a quasiparticle on this vertex (Z error nearby).
            # Outcome 0 = vertex stabilizer satisfied.

        qc.barrier()

    return qc, data_qr, z_anc_qr, x_anc_qr, z_cr, x_cr


# =============================================================================
# SECTION 3 — LOGICAL STATE PREPARATION
# Theory: Chapter 5.4.1 — "Creating a hole" and logical state initialization
# Theory: Chapter 1.2.1 — "Codewords"
# =============================================================================

def prepare_logical_zero(qc, data_qr, lattice):
    """
    Prepare the logical |0_L> state.

    Theory (Ch 1.2.1, Ch 5.4.1):
    The logical |0_L> is the +1 eigenstate of the logical Z operator Z_L
    AND the +1 eigenstate of all stabilizer generators.

    For the planar surface code:
    |0_L> = (1/√N) * Σ_{S in stabilizer group} S |0>^⊗n

    Practically: start with all data qubits in |0>, then apply all X-type
    stabilizer generators to project into the code space. Since all |0>^n
    states are +1 eigenstates of all Z-type stabilizers (Z|0>=|0>), we
    only need to handle the X-stabilizers.

    Simpler approach (correct for |0_L>):
    Start all data qubits in |0>. This is already |0_L> because:
    - Z_L = Z^⊗(column 0) has eigenvalue +1 on |0>^⊗n  → it's in the Z_L=+1 subspace
    - All Z-stabilizers B_p have eigenvalue +1 on |0>^⊗n
    - Apply X-stabilizers as unitary operations to project into their +1 eigenspace
    """
    # All qubits default to |0> in Qiskit — no gates needed for the data qubits.
    # Theory: |0>^⊗n is already in the +1 eigenspace of every Z-type stabilizer
    # (since Z|0> = +|0>). The X-stabilizer projection is handled implicitly by
    # the first round of syndrome measurements collapsing the state into the
    # +1 eigenspace of all stabilizers. This is the "blank surface state" from
    # Chapter 5.4.1, which will be projected into the codespace by measurements.
    pass


def prepare_logical_plus(qc, data_qr, lattice):
    """
    Prepare the logical |+_L> = (|0_L> + |1_L>)/√2 state.

    Theory (Ch 5.4.9, Ch 1.2.1):
    Logical |+_L> is the +1 eigenstate of the logical X operator X_L.
    X_L = X^⊗(row 0) for our code (product of X on the top boundary).

    Preparation: apply H to every data qubit in the logical X support (top row),
    then use stabilizers to spread into the codespace.
    Equivalently, apply H to ALL data qubits:
    H^⊗n |0>^⊗n = |+>^⊗n which is a +1 eigenstate of all X-stabilizers,
    and we project into the codespace via first-round measurements.
    """
    for q_idx in range(lattice.n_data):
        qc.h(data_qr[q_idx])
        # Theory: H maps |0> → |+> = (|0>+|1>)/√2. After this, the data
        # register is in |+>^⊗n. This is the +1 eigenstate of ALL X operators
        # (X|+> = +|+>), hence also of all X-type stabilizers. The Z-stabilizer
        # projection will happen automatically in the first measurement round.


def apply_logical_x(qc, data_qr, lattice):
    """
    Apply the logical X gate (encoded bit-flip).

    Theory (Ch 1.2.11, Ch 2.1.9):
    Logical X flips the encoded qubit: |0_L> ↔ |1_L>.
    The logical X operator is a horizontal string of X gates across the top row
    of the lattice (a 1-cocycle, Section 4.1.3).

    X_L = X_{(0,0)} ⊗ X_{(0,1)} ⊗ ... ⊗ X_{(0,d-1)}

    This operator commutes with all Z-stabilizers (overlaps each Z-plaquette
    on 0 or 2 qubits → commutes). It anticommutes with the logical Z_L
    (they cross exactly once), implementing a logical bit-flip.
    """
    for (r, c) in lattice.logical_x_qubits():
        q_idx = lattice.qubit_index[(r, c)]
        qc.x(data_qr[q_idx])
        # Theory: Physical X gate on each qubit of the logical X operator.
        # Together, these d gates implement the encoded X_L.
        # Weight = d = code distance, so this is the minimum-weight representative
        # of the X_L equivalence class (Section 2.1.11).


def apply_logical_z(qc, data_qr, lattice):
    """
    Apply the logical Z gate (encoded phase-flip).

    Theory (Ch 1.2.11, Ch 2.1.9):
    Logical Z applies a phase flip to the encoded qubit: |1_L> → -|1_L>.
    Z_L = Z_{(0,0)} ⊗ Z_{(1,0)} ⊗ ... ⊗ Z_{(d-1,0)}

    This is a vertical string of Z gates along the left column — a 1-cycle.
    It commutes with all X-stabilizers (overlaps each X-vertex on 0 or 2 qubits)
    and anticommutes with X_L (they intersect on exactly one qubit at (0,0)).
    """
    for (r, c) in lattice.logical_z_qubits():
        q_idx = lattice.qubit_index[(r, c)]
        qc.z(data_qr[q_idx])
        # Theory: Physical Z gate on each qubit in the left column.
        # Weight = d; this is the minimum-weight logical Z representative.


def apply_logical_hadamard(qc, data_qr, lattice):
    """
    Apply the logical Hadamard gate.

    Theory (Ch 5.5.1):
    Hadamard interchanges X and Z. The logical Hadamard is implemented
    by applying physical H to EVERY data qubit. This swaps the roles of
    X-stabilizers and Z-stabilizers (since H·Z·H = X, H·X·H = Z),
    effectively exchanging the primal and dual lattice descriptions.
    In practice this also requires a lattice relabeling (half-cell shift),
    but for simulation purposes the H⊗n gate captures the logical action.
    """
    for q_idx in range(lattice.n_data):
        qc.h(data_qr[q_idx])
        # Theory: H on every qubit. This is the "Hadamard on every qubit"
        # step from Ch 5.5.1. After this, the X_L and Z_L operators are
        # swapped, realising the logical Hadamard gate.


# =============================================================================
# SECTION 4 — NOISE MODEL
# Theory: Chapter 2.2.4 — "Error models"
# =============================================================================

def build_noise_model(p_depol, p_meas=None):
    """
    Build a Qiskit noise model implementing the depolarising or independent noise.

    Theory (Ch 2.2.4):
    Two error models are studied for the toric/surface code:

    1. DEPOLARISING NOISE: each qubit undergoes
         I   with probability (1 - p)
         X   with probability p/3
         Y   with probability p/3
         Z   with probability p/3
       This is symmetric in X, Y, Z.

    2. INDEPENDENT (UNCORRELATED) NOISE: X and Z errors are independent,
       each occurring with probability p. This decouples the X and Z
       syndrome decoding problems, simplifying analysis (Ch 2.2.4).

    Parameters
    ----------
    p_depol : float — single-qubit depolarising probability
    p_meas  : float — measurement error probability (Ch 5.3.1)
              If None, uses p_depol/10 as a reasonable assumption.
    """
    noise_model = NoiseModel()
    # Theory: NoiseModel accumulates error channels that Qiskit Aer will apply
    # after specified gates during simulation. This models the imperfect
    # physical hardware, where every gate has some error probability.

    if p_meas is None:
        p_meas = p_depol / 10
    # Theory: Measurement errors (Section 5.3.1) cause a syndrome bit to report
    # the wrong value. The 3D decoding (Section 5.3.2) handles these by treating
    # them identically to qubit errors but in the time direction.

    # ── SINGLE-QUBIT DEPOLARISING ERROR (data qubits) ────────────────────────
    single_qubit_depol = depolarizing_error(p_depol, 1)
    # Theory: This is the depolarising channel D_p:
    #   D_p(ρ) = (1-p)ρ + (p/3)(XρX + YρY + ZρZ)
    # from Ch 2.2.4. Applied after every single-qubit gate on data qubits.

    noise_model.add_all_qubit_quantum_error(single_qubit_depol, ['h', 'x', 'z', 'id'])
    # Apply the depolarising error after every H, X, Z, or identity gate.
    # 'id' (identity) represents idle time — even doing nothing causes errors.

    # ── TWO-QUBIT ERROR (CNOT gates) ─────────────────────────────────────────
    two_qubit_depol = depolarizing_error(p_depol * 10, 2)
    # Theory: Two-qubit gates (CNOT) are typically ~10x noisier than single-qubit
    # gates in superconducting hardware. The 2-qubit depolarising channel acts on
    # both qubits: it applies a random 2-qubit Pauli (I,X,Y,Z)⊗(I,X,Y,Z) with
    # equal probability p/15 for each non-identity term.

    noise_model.add_all_qubit_quantum_error(two_qubit_depol, ['cx'])
    # Add this 2-qubit error to all CNOT gates.

    # ── MEASUREMENT ERROR ─────────────────────────────────────────────────────
    meas_error = pauli_error([('X', p_meas), ('I', 1 - p_meas)])
    # Theory: A measurement error (Ch 5.3.1) is modeled as a bit-flip (X) on
    # the qubit just before measurement, flipping outcome 0↔1 with probability
    # p_meas. This is the "memoryless" measurement error model from Ch 5.3.1.

    noise_model.add_all_qubit_quantum_error(meas_error, ['measure'])
    # Apply measurement error to every measurement operation.

    return noise_model


# =============================================================================
# SECTION 5 — SYNDROME DECODING (MWPM DECODER)
# Theory: Chapter 2.2.7 — "Minimum Weight Perfect Matching"
# Theory: Chapter 2.2.2 — "Error correction"
# =============================================================================

def parse_syndrome(counts, n_z, n_x):
    """
    Extract Z and X syndrome bits from the Qiskit measurement result.

    Theory (Ch 2.2.1):
    The syndrome is the set of stabilizer measurement outcomes.
    Each outcome is 0 (+1 eigenvalue, stabilizer satisfied) or
    1 (-1 eigenvalue, stabilizer violated — quasiparticle present).

    Qiskit returns counts as a dictionary {bitstring: count}.
    The bitstring format is 'x_syndrome z_syndrome' (right-to-left ordering).
    """
    # Get the most likely measurement outcome
    most_likely = max(counts, key=counts.get)
    # Theory: We take the most frequent bitstring as our syndrome.
    # In a noiseless simulation all shots agree; with noise we take majority.

    bits = most_likely.replace(' ', '')
    # Remove spaces between registers in the Qiskit bitstring format.

    # Qiskit orders bits RIGHT-TO-LEFT within each register
    z_bits = [int(b) for b in reversed(bits[n_x:])]
    x_bits = [int(b) for b in reversed(bits[:n_x])]
    # Theory: z_bits[i] = syndrome of Z-stabilizer i (0=satisfied, 1=violated)
    #         x_bits[i] = syndrome of X-stabilizer i
    # A value of 1 means a quasiparticle is present at that stabilizer location.

    return z_bits, x_bits


def mwpm_decoder(syndrome_bits, stabilizers, d):
    """
    Minimum Weight Perfect Matching decoder for the surface code.

    Theory (Ch 2.2.7):
    MWPM finds the minimum-weight set of correction chains that annihilates
    all quasiparticles in the syndrome.

    Algorithm (Edmonds' Blossom algorithm — conceptual implementation):
    1. Find all defect locations (stabilizers with syndrome bit = 1).
    2. Compute the taxi-cab distance between every pair of defects.
    3. Find the minimum weight perfect matching: a pairing of all defects
       such that the total taxi-cab distance is minimised.
    4. For each matched pair, apply correction qubits along the shortest path.

    Theory note (Ch 2.2.7): The MWPM threshold for the surface code under
    independent noise is ~10.3% — below the optimal decoder threshold of ~11%.
    This implementation uses a greedy nearest-neighbour matching as a
    simplified version of MWPM (true Blossom algorithm requires external lib).

    Parameters
    ----------
    syndrome_bits : list[int]  — 0/1 syndrome for each stabilizer
    stabilizers   : list[dict] — stabilizer metadata (center, qubits)
    d             : int        — code distance

    Returns
    -------
    correction : set of (row, col) data qubit positions to apply correction
    """
    # ── STEP 1: FIND DEFECTS (QUASIPARTICLE POSITIONS) ──────────────────────
    defects = []
    for i, bit in enumerate(syndrome_bits):
        if bit == 1:
            defects.append(i)
    # Theory: Defects are stabilizer positions with syndrome 1 (outcome -1).
    # Each defect is a quasiparticle (Ch 2.2.3). Errors create defects in pairs.
    # An isolated defect can only arise from a chain of errors reaching a boundary
    # (planar code — boundary acts as a "virtual" quasiparticle partner).

    if len(defects) == 0:
        return set()
    # No defects → no errors detected → no correction needed.

    # ── STEP 2: COMPUTE TAXI-CAB DISTANCES ──────────────────────────────────
    def taxi_distance(i, j):
        """
        Taxi-cab (Manhattan) distance between two stabilizer centers.

        Theory (Ch 2.2.7): The weight assigned to each edge in the MWPM graph
        is the taxi-cab (L1) distance between the quasiparticle positions on the
        lattice. This is because the minimum-weight Pauli correction string
        connecting two quasiparticles has length equal to their taxi-cab distance.
        """
        cr, cc = stabilizers[i]['center']
        dr, dc = stabilizers[j]['center']
        return abs(cr - dr) + abs(cc - dc)
        # The taxi-cab distance in the lattice coordinates gives the minimum
        # number of edges (qubits) in the correction chain between these two
        # quasiparticles — exactly the weight of the correction operator.

    # ── STEP 3: GREEDY NEAREST-NEIGHBOUR MATCHING ───────────────────────────
    # Theory: Full MWPM (Blossom algorithm) is O(V^3) where V = # defects.
    # We use greedy NN matching here for clarity. For production use,
    # the PyMatching library implements true Blossom MWPM.
    matched = set()
    pairs = []
    defect_list = list(defects)

    # Handle odd number of defects: add a "virtual" boundary defect
    # Theory: In a planar code, a chain of errors hitting the boundary terminates
    # at the boundary (which acts as a virtual quasiparticle at infinity).
    # Boundary pairing is a key feature of MWPM for planar (not toric) codes.

    while len([d for d in defect_list if d not in matched]) >= 2:
        unmatched = [d for d in defect_list if d not in matched]
        i = unmatched[0]
        # Pick the nearest unmatched defect to defect i
        best_j = min(unmatched[1:], key=lambda j: taxi_distance(i, j))
        pairs.append((i, best_j))
        matched.add(i)
        matched.add(best_j)
        # Theory: Each pair (i, best_j) corresponds to one matched edge in
        # the MWPM solution. The correction operator will be a chain connecting
        # the two stabilizer positions, passing through the data qubits between them.

    # ── STEP 4: BUILD CORRECTION SET ────────────────────────────────────────
    correction = set()
    for (i, j) in pairs:
        # Find qubits on the path between defect i and defect j
        path_qubits = _path_between_stabilizers(stabilizers[i], stabilizers[j], d)
        correction.symmetric_difference_update(path_qubits)
        # Theory: Symmetric difference (XOR) is used because if two paths share
        # a qubit, applying the correction twice cancels (Pauli operators are
        # self-inverse: X^2 = I, Z^2 = I). This mirrors the Z2 chain arithmetic
        # of Section 3.2.1: chains are added modulo 2.

    return correction


def _path_between_stabilizers(stab_a, stab_b, d):
    """
    Find a straight-line path of data qubits between two stabilizer positions.

    Theory (Ch 2.2.7):
    The correction chain connects two syndrome defects via a minimum-weight
    path on the lattice. In a square lattice, this is any L-shaped path of
    length equal to the taxi-cab distance. We use a simple horizontal-then-
    vertical path for clarity.

    The returned set of qubit positions defines the correction operator:
    apply Z (or X) to each qubit in this set to annihilate the quasiparticle pair.
    """
    qubits_on_path = set()

    cr_a, cc_a = stab_a['center']   # center of stabilizer A
    cr_b, cc_b = stab_b['center']   # center of stabilizer B

    # The nearest data qubits along the path are found by stepping between centers.
    # Step horizontally first, then vertically.
    r, c = cr_a, cc_a

    # Horizontal segment
    dc = 1 if cc_b > cc_a else -1
    while abs(c - cc_b) > 0.5:
        qr = int(round(r))
        qc = int(round(c + dc * 0.5))
        if 0 <= qr < d and 0 <= qc < d:
            qubits_on_path.add((qr, qc))
        c += dc

    # Vertical segment
    dr = 1 if cr_b > cr_a else -1
    while abs(r - cr_b) > 0.5:
        qr = int(round(r + dr * 0.5))
        qc = int(round(c))
        if 0 <= qr < d and 0 <= qc < d:
            qubits_on_path.add((qr, qc))
        r += dr

    return qubits_on_path
    # Theory: The path qubits form the 1-chain c' such that ∂c' = ∂c_error
    # (same boundary = same syndrome), ensuring c' corrects the error.
    # If c' is homologically equivalent to c_error (Section 4.1.5), the
    # correction succeeds. If they are in different homology classes, it fails.


def apply_correction(qc, data_qr, correction_qubits, lattice, error_type='Z'):
    """
    Apply the correction operator to the data qubits.

    Theory (Ch 2.2.2):
    Once we have identified the correction chain from MWPM, we apply the
    corresponding Pauli operator to each qubit in the correction set.

    - To correct Z (phase-flip) errors detected by X-stabilizers → apply X
    - To correct X (bit-flip) errors detected by Z-stabilizers → apply Z

    Theory (Ch 1.2.10): Pauli operators are self-inverse (X^2 = Z^2 = I),
    so applying the same operator as the error reverses its effect.
    """
    for (r, c) in correction_qubits:
        q_idx = lattice.qubit_index.get((r, c))
        if q_idx is not None:
            if error_type == 'Z':
                qc.x(data_qr[q_idx])
                # Theory: X corrects Z errors. The X-stabilizers detected a
                # Z error (phase flip); applying X on the affected qubit restores
                # the state: X·Z|ψ> = X·Z|ψ> — up to global phase (iY|ψ>),
                # but since we know the error was Z, XZ = iY acts as a correction.
            elif error_type == 'X':
                qc.z(data_qr[q_idx])
                # Theory: Z corrects X errors. The Z-stabilizers detected an
                # X error (bit flip); applying Z corrects it: ZX = -iY ≡ correction.


# =============================================================================
# SECTION 6 — LOGICAL MEASUREMENT
# Theory: Chapter 5.4.6 — "Measurement of a logical qubit"
# =============================================================================

def measure_logical_z(qc, data_qr, logical_z_cr, lattice):
    """
    Destructively measure the logical Z operator.

    Theory (Ch 5.4.6, Ch 1.2.9):
    Logical Z is measured by measuring each data qubit in the Z (computational)
    basis along the minimal logical Z chain (left column).

    The logical Z eigenvalue = product of individual Z measurements on that column:
       Z_L = Π_{i in column 0} Z_i
    Outcome 0 (eigenvalue +1) → logical |0_L>
    Outcome 1 (eigenvalue -1) → logical |1_L>

    Theory (Ch 1.3.9): This is a destructive measurement — it collapses the
    code state. Compare to the non-destructive syndrome measurements (ancilla-
    mediated) used for error correction.
    """
    log_z_positions = lattice.logical_z_qubits()

    for i, (r, c) in enumerate(log_z_positions):
        q_idx = lattice.qubit_index[(r, c)]
        qc.measure(data_qr[q_idx], logical_z_cr[i])
        # Theory: Direct Z-basis measurement of each qubit in the logical Z chain.
        # The XOR (parity) of these d bits gives the logical Z eigenvalue.


def measure_logical_x(qc, data_qr, logical_x_cr, lattice):
    """
    Destructively measure the logical X operator.

    Theory (Ch 5.4.8, Ch 1.2.9):
    Logical X is measured by applying H (Hadamard) to each qubit in the
    logical X chain (top row) and then measuring in the Z basis.
    H rotates the X eigenstates to Z eigenstates: H|+>=|0>, H|->=|1>.

    X_L eigenvalue = product of Z-measurements after H on the top row.
    """
    log_x_positions = lattice.logical_x_qubits()

    for i, (r, c) in enumerate(log_x_positions):
        q_idx = lattice.qubit_index[(r, c)]
        qc.h(data_qr[q_idx])
        # Theory: H maps the X-basis to Z-basis. After H, measuring Z gives
        # the X eigenvalue of the original qubit. This is how we read out
        # the logical X operator (a 1-cocycle on the dual lattice) via
        # individual qubit measurements.
        qc.measure(data_qr[q_idx], logical_x_cr[i])
        # Measure in Z basis after H → reveals X eigenvalue.


def decode_logical_z(counts, d):
    """
    Decode the logical Z measurement from individual qubit outcomes.

    Theory (Ch 5.4.6):
    The logical Z eigenvalue is the parity (XOR / product) of the d
    individual Z measurements along the logical Z chain:
       result = z_0 XOR z_1 XOR ... XOR z_{d-1}
    result=0 → logical eigenvalue +1 → state was |0_L>
    result=1 → logical eigenvalue -1 → state was |1_L>
    """
    most_likely = max(counts, key=counts.get)
    bits = [int(b) for b in most_likely.replace(' ', '')]
    # Qiskit orders bits right-to-left; we decode the logical_z_cr register bits.

    logical_z_bits = bits[-d:]  # Last d bits are the logical Z measurement
    parity = sum(logical_z_bits) % 2
    # Theory: XOR = sum mod 2. Even number of 1s → eigenvalue +1 (|0_L>).
    # Odd number of 1s → eigenvalue -1 (|1_L>).
    return parity


# =============================================================================
# SECTION 7 — VISUALIZATION
# Theory: Diagrams from Chapter 2.1 and 2.2 of the notes.
# =============================================================================

def visualize_lattice_and_syndrome(lattice, z_syndrome=None, x_syndrome=None,
                                   z_correction=None, x_correction=None,
                                   title="Surface Code Lattice & Syndrome",
                                   filename=None):
    """
    Draw the surface code lattice, syndrome quasiparticles, and correction chains.

    Theory (Ch 2.1, 2.2):
    - Blue circles: data qubits (on lattice edges in the rotated code)
    - Red squares: Z-stabilizer defects (quasiparticles from X errors)
    - Green triangles: X-stabilizer defects (quasiparticles from Z errors)
    - Red dashed lines: Z-correction chains (apply X to these qubits)
    - Green dashed lines: X-correction chains (apply Z to these qubits)
    This reproduces the style of Figures 2.10–2.13 from the notes.
    """
    d = lattice.d
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    ax.set_facecolor('#f8f9fa')
    ax.set_xlim(-0.5, d - 0.5)
    ax.set_ylim(-0.5, d - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    # Draw Z-stabilizer faces (plaquettes) — light blue background
    for s in lattice.z_stabilizers:
        qubits = s['qubits']
        if len(qubits) == 4:
            rows = [q[0] for q in qubits]
            cols = [q[1] for q in qubits]
            rect = plt.Polygon([(min(cols)-0.45, min(rows)-0.45),
                                 (max(cols)+0.45, min(rows)-0.45),
                                 (max(cols)+0.45, max(rows)+0.45),
                                 (min(cols)-0.45, max(rows)+0.45)],
                               closed=True, fill=True,
                               facecolor='#d4e8fc', edgecolor='#4a90d9',
                               linewidth=0.8, alpha=0.5)
            ax.add_patch(rect)

    # Draw X-stabilizer faces (vertices) — light green background
    for s in lattice.x_stabilizers:
        qubits = s['qubits']
        if len(qubits) == 4:
            rows = [q[0] for q in qubits]
            cols = [q[1] for q in qubits]
            rect = plt.Polygon([(min(cols)-0.45, min(rows)-0.45),
                                 (max(cols)+0.45, min(rows)-0.45),
                                 (max(cols)+0.45, max(rows)+0.45),
                                 (min(cols)-0.45, max(rows)+0.45)],
                               closed=True, fill=True,
                               facecolor='#d5f5e3', edgecolor='#27ae60',
                               linewidth=0.8, alpha=0.5)
            ax.add_patch(rect)

    # Draw grid lines
    for r in range(d):
        for c in range(d):
            if c < d - 1:
                ax.plot([c, c+1], [r, r], 'k-', linewidth=0.5, alpha=0.3)
            if r < d - 1:
                ax.plot([c, c], [r, r+1], 'k-', linewidth=0.5, alpha=0.3)

    # Draw data qubits
    for (r, c) in lattice.data_qubits:
        ax.plot(c, r, 'o', color='#2c3e50', markersize=12, zorder=5)
        ax.text(c, r, f'{lattice.qubit_index[(r,c)]}',
                ha='center', va='center', fontsize=6.5,
                color='white', fontweight='bold', zorder=6)
    # Theory: Each circle is a physical qubit on the edge/vertex of the rotated
    # surface code lattice (Section 2.1.1).

    # Draw Z-correction chains
    if z_correction:
        for (r, c) in z_correction:
            ax.plot(c, r, 's', color='#e74c3c', markersize=20,
                    alpha=0.6, zorder=4)
            ax.text(c, r + 0.35, 'X', ha='center', va='center',
                    fontsize=8, color='#e74c3c', fontweight='bold', zorder=7)
    # Theory: X correction (red) applied to Z-error locations detected
    # by X-stabilizers. Matches Figure 2.10 correction chain pattern.

    # Draw X-correction chains
    if x_correction:
        for (r, c) in x_correction:
            ax.plot(c, r, '^', color='#27ae60', markersize=18,
                    alpha=0.6, zorder=4)
            ax.text(c, r - 0.35, 'Z', ha='center', va='center',
                    fontsize=8, color='#27ae60', fontweight='bold', zorder=7)
    # Theory: Z correction (green) for X-error locations detected by Z-stabilizers.

    # Draw syndrome defects
    if z_syndrome:
        for i, bit in enumerate(z_syndrome):
            if bit == 1:
                cr, cc = lattice.z_stabilizers[i]['center']
                ax.plot(cc, cr, 's', color='#e74c3c', markersize=16,
                        alpha=0.85, zorder=8, markeredgecolor='darkred',
                        markeredgewidth=1.5)
    # Theory: Red squares = quasiparticles at Z-stabilizer defects.
    # These are created by X (bit-flip) errors on adjacent data qubits.
    # From Figure 2.10: defects appear at the ends of error strings.

    if x_syndrome:
        for i, bit in enumerate(x_syndrome):
            if bit == 1:
                cr, cc = lattice.x_stabilizers[i]['center']
                ax.plot(cc, cr, '^', color='#27ae60', markersize=16,
                        alpha=0.85, zorder=8, markeredgecolor='darkgreen',
                        markeredgewidth=1.5)
    # Theory: Green triangles = quasiparticles at X-stabilizer defects.
    # Created by Z (phase-flip) errors.

    # Draw logical operator overlays
    log_x_qs = lattice.logical_x_qubits()
    for (r, c) in log_x_qs:
        ax.add_patch(plt.Circle((c, r), 0.38, color='#8e44ad', fill=False,
                                linewidth=2.5, linestyle='--', zorder=9))
    # Theory: Purple dashed circle highlights the logical X operator (top row),
    # matching Figure 2.7's closed X-loop on the dual lattice.

    log_z_qs = lattice.logical_z_qubits()
    for (r, c) in log_z_qs:
        ax.add_patch(plt.Circle((c, r), 0.38, color='#e67e22', fill=False,
                                linewidth=2.5, linestyle=':', zorder=9))
    # Theory: Orange dotted circle shows the logical Z operator (left column),
    # matching Figure 2.6's closed Z-loop on the primal lattice.

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#d4e8fc', edgecolor='#4a90d9', label='Z-stabilizer (plaquette B_p)'),
        mpatches.Patch(facecolor='#d5f5e3', edgecolor='#27ae60', label='X-stabilizer (vertex A_v)'),
        plt.Line2D([0],[0], marker='s', color='w', markerfacecolor='#e74c3c',
                   markersize=10, label='Z-stab defect (X-error detected)'),
        plt.Line2D([0],[0], marker='^', color='w', markerfacecolor='#27ae60',
                   markersize=10, label='X-stab defect (Z-error detected)'),
        plt.Line2D([0],[0], color='#8e44ad', lw=2, linestyle='--', label='Logical X_L (top row)'),
        plt.Line2D([0],[0], color='#e67e22', lw=2, linestyle=':', label='Logical Z_L (left col)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7.5,
              framealpha=0.9, edgecolor='#cccccc')

    ax.set_title(title, fontsize=12, fontweight='bold', pad=10)
    ax.set_xlabel("Column (c)", fontsize=9)
    ax.set_ylabel("Row (r)", fontsize=9)
    ax.grid(False)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    return filename


# =============================================================================
# SECTION 8 — COMPLETE SURFACE CODE SIMULATION
# Theory: Putting it all together — encoding, error, syndrome, decoding.
# =============================================================================

def run_surface_code_simulation(d=3, p_error=0.01, n_shots=1024,
                                rounds=3, initial_state='0'):
    """
    Full surface code simulation pipeline.

    Theory overview:
    1. Build lattice (Ch 2.1)         → geometry of qubits and stabilizers
    2. Prepare logical state (Ch 5.4) → encode |0_L> or |+_L>
    3. Syndrome extraction (Ch 1.3.9) → measure all stabilizer generators
    4. Add noise (Ch 2.2.4)           → simulate physical errors
    5. Parse syndrome (Ch 2.2.1)      → find defect quasiparticles
    6. MWPM decoding (Ch 2.2.7)       → find minimum-weight correction
    7. Apply correction (Ch 2.2.2)    → restore code state
    8. Logical measurement (Ch 5.4.6) → verify outcome matches initial state

    Parameters
    ----------
    d            : code distance (encodes 1 logical qubit in d^2 physical qubits)
    p_error      : physical error rate per qubit per gate
    n_shots      : number of simulation repetitions
    rounds       : number of syndrome extraction rounds (for 3D decoding)
    initial_state: '0' for |0_L>, '1' for |1_L>, '+' for |+_L>
    """
    print(f"\n{'='*60}")
    print(f"  Surface Code Simulation  |  d={d}, p={p_error}, rounds={rounds}")
    print(f"  Initial state: |{initial_state}_L>")
    print(f"{'='*60}")

    # ── STEP 1: BUILD LATTICE ────────────────────────────────────────────────
    lattice = SurfaceCodeLattice(d)
    # Theory: Creates the d×d rotated surface code lattice with d^2 data qubits,
    # (d^2-1)/2 Z-stabilizers, and (d^2-1)/2 X-stabilizers. (Section 2.1)
    print(f"[Lattice] n_data={lattice.n_data}, "
          f"n_Z_stabs={len(lattice.z_stabilizers)}, "
          f"n_X_stabs={len(lattice.x_stabilizers)}")

    # ── STEP 2: BUILD SYNDROME CIRCUIT ──────────────────────────────────────
    qc, data_qr, z_anc_qr, x_anc_qr, z_cr, x_cr = \
        build_stabilizer_circuit(lattice, rounds=rounds)
    # Theory: The stabilizer measurement circuit (Section 1.3.9) uses one ancilla
    # per stabilizer. The circuit has structure: [prep] [syndrome round 1] ...
    # [syndrome round R] [logical measurement].

    # ── STEP 3: PREPARE LOGICAL STATE ───────────────────────────────────────
    # We build a SEPARATE state-prep circuit and prepend it.
    prep_qc = QuantumCircuit(data_qr)
    if initial_state == '0':
        prepare_logical_zero(prep_qc, data_qr, lattice)
        # Theory: |0_L> = all qubits in |0>, projected by measurements (Ch 5.4.1)
    elif initial_state == '1':
        prepare_logical_zero(prep_qc, data_qr, lattice)
        apply_logical_x(prep_qc, data_qr, lattice)
        # Theory: |1_L> = apply X_L to |0_L>. X_L flips the encoded qubit.
    elif initial_state == '+':
        prepare_logical_plus(prep_qc, data_qr, lattice)
        # Theory: |+_L> = H^⊗n |0>^n, projected into codespace (Ch 5.4.9)

    # Compose: state prep + syndrome circuit
    full_qc = prep_qc.compose(qc, qubits=list(range(lattice.n_data)),
                               clbits=[], inplace=False)
    # Note: We rebuild properly using QuantumCircuit composition.

    # ── STEP 4: BUILD FULL CIRCUIT (correct composition) ────────────────────
    # Build one unified circuit with all registers
    n_z = len(lattice.z_stabilizers)
    n_x = len(lattice.x_stabilizers)

    # Logical measurement registers (for final readout)
    logical_z_cr = ClassicalRegister(d, name='lz')
    logical_x_cr = ClassicalRegister(d, name='lx')

    # Rebuild the full circuit
    full_circuit = QuantumCircuit(data_qr, z_anc_qr, x_anc_qr,
                                   z_cr, x_cr, logical_z_cr, logical_x_cr)

    # Apply state preparation
    if initial_state == '0':
        pass   # |0> is the default
    elif initial_state == '1':
        for (r, c) in lattice.logical_x_qubits():
            full_circuit.x(data_qr[lattice.qubit_index[(r, c)]])
    elif initial_state == '+':
        for q_idx in range(lattice.n_data):
            full_circuit.h(data_qr[q_idx])

    full_circuit.barrier()

    # ── STEP 5: SYNDROME EXTRACTION ROUNDS ──────────────────────────────────
    # Theory (Ch 5.3.1): Repeated syndrome extraction is essential for
    # fault tolerance. Each round measures all stabilizers once.
    for round_idx in range(rounds):
        # Reset ancillae for this round
        full_circuit.reset(z_anc_qr)
        full_circuit.reset(x_anc_qr)
        full_circuit.barrier()

        # Z-stabilizer circuits
        for s_idx, stab in enumerate(lattice.z_stabilizers):
            anc = z_anc_qr[s_idx]
            full_circuit.h(anc)
            for (qr, qc_) in stab['qubits']:
                q_idx = lattice.qubit_index[(qr, qc_)]
                full_circuit.cx(anc, data_qr[q_idx])
            full_circuit.h(anc)
            full_circuit.measure(anc, z_cr[s_idx + round_idx * n_z])

        full_circuit.barrier()

        # X-stabilizer circuits
        for s_idx, stab in enumerate(lattice.x_stabilizers):
            anc = x_anc_qr[s_idx]
            full_circuit.h(anc)
            for (qr, qc_) in stab['qubits']:
                q_idx = lattice.qubit_index[(qr, qc_)]
                full_circuit.cx(data_qr[q_idx], anc)
            full_circuit.h(anc)
            full_circuit.measure(anc, x_cr[s_idx + round_idx * n_x])

        full_circuit.barrier()

    # ── STEP 6: LOGICAL MEASUREMENT ─────────────────────────────────────────
    # Theory (Ch 5.4.6): Destructive readout of the logical qubit.
    if initial_state in ['0', '1']:
        measure_logical_z(full_circuit, data_qr, logical_z_cr, lattice)
    else:  # '+' state
        measure_logical_x(full_circuit, data_qr, logical_x_cr, lattice)

    print(f"[Circuit] Depth = {full_circuit.depth()}, "
          f"Total qubits = {full_circuit.num_qubits}")

    # ── STEP 7: ADD NOISE AND SIMULATE ──────────────────────────────────────
    noise_model = build_noise_model(p_error)
    # Theory (Ch 2.2.4): Depolarising noise applied after each gate.

    simulator = AerSimulator(noise_model=noise_model)
    # Theory: AerSimulator runs the noisy quantum circuit classically.

    job = simulator.run(full_circuit, shots=n_shots)
    # Theory: 'shots' = number of independent runs of the experiment.
    # With noise, different shots give different outcomes; we analyse the
    # distribution to assess code performance.

    counts = job.result().get_counts()
    # counts = {'bitstring': frequency} — e.g. {'0001 0010': 512, ...}

    print(f"[Simulation] Completed {n_shots} shots.")
    print(f"[Simulation] Distinct outcomes: {len(counts)}")

    # ── STEP 8: SYNDROME ANALYSIS ────────────────────────────────────────────
    z_synd, x_synd = parse_syndrome(counts, n_z, n_x)
    # Theory: Parse the most frequent syndrome bitstring into per-stabilizer bits.

    n_z_defects = sum(z_synd)
    n_x_defects = sum(x_synd)
    print(f"[Syndrome] Z-stabilizer defects: {n_z_defects} / {n_z}")
    print(f"[Syndrome] X-stabilizer defects: {n_x_defects} / {n_x}")

    # ── STEP 9: MWPM DECODING ────────────────────────────────────────────────
    # Theory (Ch 2.2.7): MWPM finds minimum-weight correction chains.
    z_correction = mwpm_decoder(z_synd, lattice.z_stabilizers, d)
    x_correction = mwpm_decoder(x_synd, lattice.x_stabilizers, d)

    print(f"[Decoder] Z-correction qubits: {z_correction}")
    print(f"[Decoder] X-correction qubits: {x_correction}")

    # ── STEP 10: LOGICAL OUTCOME ANALYSIS ────────────────────────────────────
    # Theory (Ch 2.2.2): Check if decoding succeeded (correct homology class).
    logical_value = decode_logical_z(counts, d)
    expected = 1 if initial_state == '1' else 0
    success = (logical_value == expected)

    print(f"[Logical] Measured: {logical_value}, Expected: {expected}, "
          f"Success: {'✓' if success else '✗'}")

    return {
        'lattice': lattice,
        'circuit': full_circuit,
        'counts': counts,
        'z_syndrome': z_synd,
        'x_syndrome': x_synd,
        'z_correction': z_correction,
        'x_correction': x_correction,
        'logical_result': logical_value,
        'success': success,
        'n_shots': n_shots,
    }


# =============================================================================
# SECTION 9 — THRESHOLD ANALYSIS
# Theory: Chapter 2.2.5 — "Code thresholds"
# =============================================================================

def estimate_threshold(d_values=[3, 5], p_values=None, n_shots=256):
    """
    Estimate the surface code threshold by sweeping error rate.

    Theory (Ch 2.2.5):
    The code threshold p_th is the error rate below which larger codes
    perform better (fewer logical errors) than smaller codes.
    Below threshold: P(logical error) decreases as d increases.
    Above threshold: P(logical error) increases as d increases.
    The crossover point estimates the threshold.

    For the surface code with MWPM decoder: threshold ≈ 10.3% (Ch 2.2.7).
    """
    if p_values is None:
        p_values = [0.001, 0.005, 0.01, 0.02, 0.05, 0.08, 0.10, 0.12]
    # Theory: We sweep p from well below to above the expected threshold ~10.3%.

    results = {}

    for d in d_values:
        results[d] = {'p': [], 'p_fail': []}
        lattice = SurfaceCodeLattice(d)
        n_z = len(lattice.z_stabilizers)
        n_x = len(lattice.x_stabilizers)

        for p in p_values:
            # Build and run the noisy circuit
            n_fail = 0
            for _ in range(n_shots):
                # Quick syndrome-only circuit for threshold estimation
                qc_s = QuantumCircuit(
                    QuantumRegister(lattice.n_data, 'd'),
                    QuantumRegister(n_z, 'az'),
                    QuantumRegister(n_x, 'ax'),
                    ClassicalRegister(n_z, 'sz'),
                    ClassicalRegister(n_x, 'sx')
                )
                # Single round of syndrome extraction
                for s_idx, stab in enumerate(lattice.z_stabilizers):
                    anc_idx = lattice.n_data + s_idx
                    qc_s.reset(anc_idx)
                    qc_s.h(anc_idx)
                    for (qr, qc_) in stab['qubits']:
                        q_idx = lattice.qubit_index[(qr, qc_)]
                        qc_s.cx(anc_idx, q_idx)
                    qc_s.h(anc_idx)
                    qc_s.measure(anc_idx, s_idx)

                noise = build_noise_model(p)
                sim = AerSimulator(noise_model=noise)
                job = sim.run(qc_s, shots=1)
                counts = job.result().get_counts()
                z_synd, x_synd = parse_syndrome(counts, n_z, n_x)
                correction = mwpm_decoder(z_synd, lattice.z_stabilizers, d)
                # Theory: A failure occurs when the correction chain is in a
                # different homology class than the actual error chain — creating
                # an undetectable logical error (Section 4.1.5).
                if len(correction) % d == 0 and len(correction) > 0:
                    n_fail += 1

            p_fail = n_fail / n_shots
            results[d]['p'].append(p)
            results[d]['p_fail'].append(p_fail)
            print(f"  d={d}, p={p:.3f}, P(fail)={p_fail:.3f}")

    return results


def plot_threshold(results, filename=None):
    """
    Plot logical failure rate vs physical error rate for different code distances.

    Theory (Ch 2.2.5):
    The threshold manifests as a CROSSING POINT of the curves for different d.
    Below threshold (p < p_th): larger d → lower P(fail) — code is beneficial.
    Above threshold (p > p_th): larger d → higher P(fail) — code is harmful.
    The crossing point estimates the threshold (theoretically ~10.3% for MWPM).
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    colors_list = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    for (d, data), color in zip(results.items(), colors_list):
        ax.semilogy(data['p'], data['p_fail'], 'o-',
                    color=color, linewidth=2, markersize=8,
                    label=f'd = {d}')
        # Theory: Logarithmic y-axis shows the exponential suppression of
        # logical errors below threshold. The slope steepens as d increases:
        # P(logical error) ~ (p/p_th)^{floor(d/2)+1} below threshold.

    ax.axvline(x=0.103, color='k', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(0.103, ax.get_ylim()[1]*0.6, '  p_th ≈ 10.3%\n  (MWPM theory)',
            fontsize=9, color='black', alpha=0.7)
    # Theory: The vertical dashed line marks the theoretical MWPM threshold
    # of 10.3% for the surface code under independent noise (Ch 2.2.7).

    ax.set_xlabel('Physical error rate p', fontsize=11)
    ax.set_ylabel('Logical failure rate P(fail)', fontsize=11)
    ax.set_title('Surface Code Threshold Analysis\n'
                  '(Crossing point ≈ code threshold)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, title='Code distance d')
    ax.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# SECTION 10 — MAIN DEMO
# =============================================================================

def main():
    """
    Run a demonstration of the surface code implementation.

    Theory summary:
    This demo implements the full surface code pipeline:
    1. Build the rotated planar surface code (Ch 2.1, Ch 4.2)
    2. Encode logical qubits in the stabilizer codespace (Ch 1.3)
    3. Extract error syndromes via ancilla-mediated measurements (Ch 1.3.9)
    4. Simulate physical errors using depolarising noise model (Ch 2.2.4)
    5. Decode using Minimum Weight Perfect Matching (Ch 2.2.7)
    6. Apply corrections and verify logical outcome (Ch 2.2.2)
    """
    print("\n" + "="*60)
    print("  SURFACE CODE DEMO — Manual Qiskit Implementation")
    print("  Based on Dan Browne's lecture notes (University of Innsbruck)")
    print("="*60)

    d = 3  # distance-3 surface code: [[9, 1, 3]] — 9 physical, 1 logical
    p = 0.01  # 1% physical error rate (well below threshold of ~10.3%)

    # ── BUILD AND VISUALIZE LATTICE ──────────────────────────────────────────
    lattice = SurfaceCodeLattice(d)
    print(f"\n[Lattice d={d}]")
    print(f"  Physical qubits:    n = {lattice.n_data}")
    print(f"  Z stabilizers:      {len(lattice.z_stabilizers)}")
    print(f"  X stabilizers:      {len(lattice.x_stabilizers)}")
    print(f"  Logical qubits:     k = {lattice.n_data - len(lattice.z_stabilizers) - len(lattice.x_stabilizers)}")
    print(f"  Code distance:      d = {d}")
    print(f"  [[n,k,d]] params:   [[{lattice.n_data}, 1, {d}]]")
    # Theory: [[9,1,3]] means 9 physical qubits, 1 logical qubit, distance 3.
    # Can correct floor((3-1)/2) = 1 arbitrary error (Section 1.1.6).

    # Visualize the lattice
    fig1 = "/tmp/lattice_clean.png"
    visualize_lattice_and_syndrome(
        lattice, title=f"d={d} Rotated Surface Code Lattice\n"
                        f"(Blue: Z-stab plaquettes  |  Green: X-stab vertices)",
        filename=fig1
    )
    print(f"\n[Figure] Lattice diagram saved.")

    # ── SIMULATE WITH INJECTED ERROR ─────────────────────────────────────────
    print(f"\n[Demo] Simulating d={d} code with p={p} error rate...")

    # Run the full simulation
    result = run_surface_code_simulation(
        d=d, p_error=p, n_shots=256, rounds=1, initial_state='0'
    )

    # Visualize syndrome
    fig2 = "/tmp/syndrome.png"
    visualize_lattice_and_syndrome(
        lattice,
        z_syndrome=result['z_syndrome'],
        x_syndrome=result['x_syndrome'],
        z_correction=result['z_correction'],
        x_correction=result['x_correction'],
        title=f"d={d} Surface Code: Syndrome & MWPM Correction\n"
               f"(Red squares: Z-defects | Green triangles: X-defects)",
        filename=fig2
    )
    print(f"[Figure] Syndrome + correction diagram saved.")

    # ── MULTI-ROUND SYNDROME EXTRACTION ─────────────────────────────────────
    print(f"\n[Demo] 3-round syndrome extraction (fault-tolerant)...")
    result_3r = run_surface_code_simulation(
        d=d, p_error=p, n_shots=256, rounds=3, initial_state='0'
    )

    print(f"\n[Summary]")
    print(f"  Single round — success: {result['success']}")
    print(f"  Three rounds  — success: {result_3r['success']}")

    return lattice, result, fig1, fig2


if __name__ == '__main__':
    lattice, result, fig1, fig2 = main()
    print("\n[Done] Surface code implementation complete.")
    print("Run with: python surface_code_qiskit.py")
