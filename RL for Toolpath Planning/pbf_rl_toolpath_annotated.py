"""
ANNOTATED RESEARCH REPRODUCTION

This module maps the paper methodology to executable components:

    Procedural target masks
        -> Geometric or thermal RL environment
        -> CNN-based DQN/PPO policy
        -> Discrete movement/source action
        -> Environment and FE state update
        -> Composite manufacturing reward

Important interpretation notes
------------------------------
- The target mask represents the region that should be processed.
- The phase map records cells that have already crossed the effective melt threshold.
- The source channel stores both laser position and laser on/off state.
- In the thermal environment, every agent action advances a transient Q4 FE solve.
- The reward is primarily an efficiency-and-coverage objective; thermal quality
  improvements emerge indirectly unless additional thermal penalties are added.
- This code is suitable for methodology demonstration and extension, not direct
  machine deployment.

"""


"""
Reference implementation of the methodology in:
"Learning-Based Toolpath Planning for Bed-Based Additive Manufacturing"

Core components
---------------
1. Procedural generation of training/validation/test masks.
2. Geometric grid environment.
3. Thermal grid environment coupled to a structured Q4 finite-element heat solver.
4. Five discrete actions: up, down, left, right, source toggle.
5. Composite reward: fill, spill, revisit, and step terms.
6. Optional Stable-Baselines3 DQN/PPO training helpers.
7. Zigzag baseline and evaluation utilities.

This is a research reproduction scaffold, not an exact copy of the authors' code.
The paper uses a simplified 2-D, single-layer thermal benchmark; this code follows
the same scope and exposes parameters so it can be extended.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple
import argparse
import math
import random
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import sparse
from scipy.sparse.linalg import factorized


# ---------------------------------------------------------------------------
# Optional Gymnasium dependency
# ---------------------------------------------------------------------------
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False

    class _FallbackEnv:
        metadata: Dict[str, object] = {}

    class _FallbackDiscrete:
        def __init__(self, n: int):
            self.n = int(n)

        def sample(self) -> int:
            return int(np.random.randint(self.n))

    class _FallbackBox:
        def __init__(self, low, high, shape, dtype=np.float32):
            self.low = low
            self.high = high
            self.shape = tuple(shape)
            self.dtype = dtype

    class _FallbackSpaces:
        Discrete = _FallbackDiscrete
        Box = _FallbackBox

    class _FallbackGym:
        Env = _FallbackEnv

    gym = _FallbackGym()
    spaces = _FallbackSpaces()


Split = Literal["train", "validation", "test"]
EnvKind = Literal["geometric", "thermal"]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def seed_everything(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Procedural mask generation
# ---------------------------------------------------------------------------
@dataclass
class MaskSample:
    mask: np.ndarray
    name: str
    size_factor: float
    rotation_deg: float


class ProceduralMaskGenerator:
    """
    Generates centered binary masks with different families for train/validation/test.

    Paper-inspired split:
      train      -> letters
      validation -> symbols
      test       -> geometric shapes

    To keep masks meaningful on a 10x10 grid, the default minimum size is 0.45.
    """

    TRAIN_TOKENS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    VALIDATION_TOKENS = ["+", "#", "*", "&", "?", "=", "%", "@", "$", "!"]
    TEST_SHAPES = ["circle", "ellipse", "ngon", "ring", "square", "triangle"]
# This initializes the geometry generator. 
    def __init__(
        self,
        n: int = 10,  # 10 × 10 cells
        min_size_factor: float = 0.45,
        max_size_factor: float = 1.0,   # These determine how large the generated character can appear inside the image.
        allow_rotation: bool = True,   # This allows the generated character or symbol to be randomly rotated.
    ):
        if n < 6:
            raise ValueError("n must be at least 6 for meaningful procedural masks.")   
            # The code rejects grids smaller than 6 × 6 because letters and shapes would become too coarse to recognize.
            # These lines save the input settings as class attributes.
        self.n = int(n)
        self.min_size_factor = float(min_size_factor)
        self.max_size_factor = float(max_size_factor)
        self.allow_rotation = bool(allow_rotation)

# This method selects a font for rendering letters and symbols.
# The @staticmethod means that the function does not require access to self.
# It can be called as: font = ProceduralMaskGenerator._font(50)
# The code tries three common fonts in order. If the first font exists, it is returned. If it is not installed, Pillow raises OSError, and the code tries the next font.
    @staticmethod
    def _font(size: int) -> ImageFont.ImageFont:
        for name in (
            "DejaVuSans-Bold.ttf",
            "Arial.ttf",
            "LiberationSans-Bold.ttf",
        ):
            try:
                return ImageFont.truetype(name, size=size)
            except OSError:
                pass
        return ImageFont.load_default()

# 1. This function converts a letter or symbol into a binary grid mask. 
# Input: token: letter or symbol, such as "A" or "+".
# size_factor: controls its size.
# rotation_deg: rotation angle in degrees.
# Output: a Boolean NumPy array of shape (n, n) 

    def _render_token(
        self,
        token: str,
        size_factor: float,
        rotation_deg: float,
    ) -> np.ndarray:
        hi = max(128, self.n * 16)   # defines the temporary image resolution, for n=10, =160
        canvas = Image.new("L", (hi, hi), 0)   # The token is first rendered at high resolution before being reduced to 10 × 10. This produces smoother and more recognizable masks. This creates a grayscale image. "L" means one-channel grayscale. (hi, hi) is the image size. 0 means the background is black.
        
        draw = ImageDraw.Draw(canvas)  # This creates a Pillow drawing object that can place text onto the canvas.

        font_size = max(10, int(hi * 0.72 * size_factor))  # 160 × 0.72 × 0.8 = 92, The max(10, ...) guarantees that the font is never smaller than 10 pixels.
        font = self._font(font_size)
        bbox = draw.textbbox((0, 0), token, font=font)   # This calculates the bounding box around the rendered token. A bounding box usually has the form:(left, top, right, bottom)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]   # token_width, token_height
        x = (hi - tw) / 2 - bbox[0]  # These equations calculate the position required to place the token in the center of the image.
        y = (hi - th) / 2 - bbox[1]
        draw.text((x, y), token, fill=255, font=font)   # This draws the character on the black canvas. fill=255 means white. Background remains 0,

        if self.allow_rotation and abs(rotation_deg) > 1e-12:  # Rotation is performed only when: rotation is enabled, and the angle is not effectively zero.
            canvas = canvas.rotate(
                rotation_deg,
                resample=Image.Resampling.BICUBIC,
                expand=False,
                fillcolor=0,
            )

        # Resize to the environment grid. The high-resolution image is reduced to the final environment resolution. 
        # LANCZOS is a high-quality downsampling method.
        # 160 × 160 image → 10 × 10 image
        small = canvas.resize((self.n, self.n), Image.Resampling.LANCZOS)
        arr = np.asarray(small, dtype=np.float32) / 255.0
        mask = arr >= 0.30   # Every cell with intensity at least 0.30 becomes True. final mask may look like: 
        return self._ensure_valid(mask)  # This sends the binary mask to another function that checks whether the generated shape contains enough active cells.
        # For example, if a small or heavily rotated letter almost disappears during resizing, _ensure_valid() prevents an empty or unusable target.

# Instead of drawing a letter like "A" or symbol like "+", it draws a geometric shape such as:  circle ellipse ring square triangle n-gon
# returns: True = printable target region  False = background
    def _render_shape(
        self,
        shape: str,   # the type of shape to draw
        size_factor: float,
        rotation_deg: float,
    ) -> np.ndarray:
        
        hi = max(128, self.n * 16)
        image = Image.new("L", (hi, hi), 0)  # L= grayscale, 0 = black background
        draw = ImageDraw.Draw(image)  # Creates a drawing object so we can draw shapes on the image.

        radius = 0.42 * hi * size_factor  # This controls the size of the shape.
        cx = cy = hi / 2  # This sets the center of the image.
        box = [cx - radius, cy - radius, cx + radius, cy + radius]  # bounding box around the shape

        # Draw the requested shape
        if shape == "circle":
            draw.ellipse(box, fill=255)  # circle
        elif shape == "ellipse":
            draw.ellipse(
                [cx - radius, cy - 0.62 * radius, cx + radius, cy + 0.62 * radius],  # otherwise draw a horizontally stretched ellipse
                fill=255,
            )
        elif shape == "ring":
            draw.ellipse(box, fill=255)
            inner = radius * 0.52
            draw.ellipse(
                [cx - inner, cy - inner, cx + inner, cy + inner],
                fill=0,
            )
        elif shape == "square":
            draw.rectangle(box, fill=255)
        elif shape == "triangle":
            pts = [
                (cx, cy - radius),
                (cx - 0.88 * radius, cy + 0.78 * radius),
                (cx + 0.88 * radius, cy + 0.78 * radius),
            ]
            draw.polygon(pts, fill=255)
        elif shape == "ngon":  # This creates a regular polygon with 7 sides.
            sides = 7
            pts = []
            for k in range(sides):
                a = -math.pi / 2 + 2 * math.pi * k / sides
                pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
            draw.polygon(pts, fill=255)
        else:
            raise ValueError(f"Unknown test shape: {shape}")

        # Optionally rotate the shape
        if self.allow_rotation and shape not in {"circle", "ring"}:
            image = image.rotate(
                rotation_deg,
                resample=Image.Resampling.BICUBIC,
                expand=False,
                fillcolor=0,
            )
        # Resize to the final grid resolution
        small = image.resize((self.n, self.n), Image.Resampling.LANCZOS)
        arr = np.asarray(small, dtype=np.float32) / 255.0
        mask = arr >= 0.30
        return self._ensure_valid(mask)

    # Ensure the mask is valid. It prevents cases where: the object becomes too small, rotation makes it nearly disappear, resizing causes the target to have too few active pixels.
    def _ensure_valid(self, mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(mask, dtype=bool)
        # Guarantee a nonempty target and avoid one-cell artifacts.
        if mask.sum() < max(3, self.n // 2):   # counts how many cells belong to the target geometry. The minimum acceptable number is:
            c = self.n // 2
            mask[max(0, c - 1): min(self.n, c + 2),
                 max(0, c - 1): min(self.n, c + 2)] = True
        return mask  
# For n = 10 and c = 5, the selected rows and columns are: 4:7. Creates a central 3 × 3 white block. This is only used when the original generated geometry is too small.

# This method generates one complete target sample. It decides:
# which geometry to use;
# how large it should be;
# how much it should rotate;
# which data split it belongs to.
# Returns a MaskSample containing both the mask and its metadata.
    def sample(
        self,
        split: Split,
        rng: np.random.Generator,  # used to randomly select: shape or token, size, rotation
        fixed_name: Optional[str] = None,  # forces a specific geometry.
    ) -> MaskSample:

        size = float(rng.uniform(self.min_size_factor, self.max_size_factor))  # selects a random decimal value between the minimum and maximum size factors.
        rotation = float(rng.uniform(0.0, 360.0)) if self.allow_rotation else 0.0

        if split == "train":  # Training targets use capital letters.
            token = fixed_name or str(rng.choice(self.TRAIN_TOKENS))
            mask = self._render_token(token, size, rotation)
            name = f"letter_{token}"
        elif split == "validation":  # Validation targets use symbols.
            token = fixed_name or str(rng.choice(self.VALIDATION_TOKENS))
            mask = self._render_token(token, size, rotation)
            name = f"symbol_{token}"
        elif split == "test":
            shape = fixed_name or str(rng.choice(self.TEST_SHAPES))
            # The manuscript uses zero rotation for held-out test shapes.
            mask = self._render_shape(shape, size, 0.0)
            rotation = 0.0
            name = shape
        else:
            raise ValueError(f"Unsupported split: {split}")

        return MaskSample(mask=mask, name=name, size_factor=size, rotation_deg=rotation)


# ---------------------------------------------------------------------------
# Structured Q4 finite-element transient heat solver
# This class solves a simplified two-dimensional transient heat-conduction problem for the powder bed after every reinforcement-learning action.
# The RL agent specifies: row, col, source_on
# 1. Locates the laser in the selected grid cell.
# 2. Distributes the Gaussian heat input to nearby finite-element nodes.
# 3. Solves the transient heat equation for one time step.
# 4. Returns an n × n temperature map to the RL environment.
# The governing equation is approximately ρc∂t/∂T−∇⋅(κ∇T)+h(T−T∞)=q src
# ---------------------------------------------------------------------------

# The @dataclass decorator creates a container for the thermal and numerical parameters.
@dataclass
class ThermalParameters:
    domain_size_m: float = 1.0e-3   # printable region is 1 mm wide and high
    conductivity: float = 113.0  # Thermal conductivity
    volumetric_heat_capacity: float = 2.403e6  # ρc = It represents how much energy is required to raise the material temperature.
    effective_heat_transfer: float = 0.0  # This controls the heat-loss term. Since it is zero, effective convection or out-of-plane heat loss is disabled. 
    ambient_temperature: float = 300.0
    boundary_temperature: float = 300.0
    initial_temperature: float = 300.0
    melt_threshold: float = 850.0  # considered processed or melted when its temperature exceeds: 850K
    source_power_per_depth: float = 3.0e6  # This is the 2D heat-source magnitude. It is interpreted as laser power per unit out-of-plane depth in the 2-D model.
    gaussian_width_m: float = 5.0e-6  # Gaussian source width: 5 um
    time_step_s: float = 1.0e-5  # The solver advances by: 10 us


class StructuredQ4ThermalSolver:
    """
    2-D Q4 finite-element heat solver with one padded element layer on each side.

    Interior printable region:
        n x n elements

    Full FE region:
        (n+2) x (n+2) elements
        (n+3) x (n+3) nodes

    The outer node boundary is held at a fixed temperature.
    """

    def __init__(self, n: int, params: Optional[ThermalParameters] = None):
        self.n = int(n)
        self.params = params or ThermalParameters()
        self.nel = self.n + 2   # padded elements 10+2
        self.nnode_side = self.n + 3  # A mesh with n + 2 elements needs n + 3 nodes
        self.ndof = self.nnode_side ** 2  # 13*13 = 169
        self.dx = self.params.domain_size_m / self.n   # For a 1 mm printable domain and n = 10. = 0.1 mm
        self.dy = self.dx

        # Build the FE model
        self.node_xy = self._build_node_coordinates()
        
        # Create the physical coordinates of all nodes
        # mass: thermal-capacity matrix M;
        # stiffness: heat-conduction matrix K;
        # sink: effective heat-loss matrix H;
        # ambient_load: ambient-temperature contribution.
        self.mass, self.stiffness, self.sink, self.ambient_load = self._assemble()
        self._prepare_time_step()  # Constructs and factorizes the backward-Euler system matrix
        self.reset()  # Initializes every node to the initial temperature

    # Convert grid indices to node IDs
    def _node_id(self, i: int, j: int) -> int:
        return i * self.nnode_side + j   # (2, 4) will convert to 2x13+4 = 30. This is called row-major indexing.

    # Build nodal coordinates
    def _build_node_coordinates(self) -> np.ndarray:
        coords = np.zeros((self.ndof, 2), dtype=np.float64)
        for i in range(self.nnode_side):
            for j in range(self.nnode_side):
                idx = self._node_id(i, j)
                coords[idx] = (j * self.dx, i * self.dy)
        return coords

    # Q4 shape functions
#     This function evaluates the Q4 shape functions in the reference element:
# −1≤ξ≤1,−1≤η≤1.
#    The four shape functions are N1=1/4 (1−ξ)(1−η),...

    @staticmethod
    def _shape(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = 0.25 * np.array([
            (1 - xi) * (1 - eta),
            (1 + xi) * (1 - eta),
            (1 + xi) * (1 + eta),
            (1 - xi) * (1 + eta),
        ], dtype=np.float64)
        dN_dxi = 0.25 * np.array([
            -(1 - eta),
             (1 - eta),
             (1 + eta),
            -(1 + eta),
        ], dtype=np.float64)
        dN_deta = 0.25 * np.array([
            -(1 - xi),
            -(1 + xi),
             (1 + xi),
             (1 - xi),
        ], dtype=np.float64)
        return N, dN_dxi, dN_deta

    # This calculates the four-element matrices and load vector.
    def _local_matrices(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p = self.params
        Me = np.zeros((4, 4), dtype=np.float64)
        Ke = np.zeros((4, 4), dtype=np.float64)
        He = np.zeros((4, 4), dtype=np.float64)
        fe_ambient = np.zeros(4, dtype=np.float64)

# The code uses a 2 × 2 Gauss quadrature rule.
# The four integration points are: (−1/sqrt(3),−1/sqrt(3)), ... The nested loops evaluate the matrices at all four points.
# This converts the reference coordinates (ξ,η) into physical coordinates (x,y).
# For a rectangular element:
# J=[[Δx/2 0] [0 Δy/2]]

        gauss = [-1 / math.sqrt(3), 1 / math.sqrt(3)]
        J = np.array([[self.dx / 2, 0.0], [0.0, self.dy / 2]])   
        detJ = np.linalg.det(J)  # The determinant scales the integration area dΩ=det(J)dξdη.
        invJ = np.linalg.inv(J)  # The inverse Jacobian converts reference derivatives into physical derivatives.

# Thermal mass matrix The thermal mass matrix is Me=∫Ω ρcN^NdΩ
# The conductivity matrix is Ke=∫Ωe κBTBdΩ
# Effective heat-loss matrix. This corresponds to:He =∫Ωe hNT NdΩ
        for xi in gauss:
            for eta in gauss:
                N, dN_dxi, dN_deta = self._shape(xi, eta)
                grad_ref = np.vstack([dN_dxi, dN_deta])
                grad = invJ @ grad_ref
                Me += p.volumetric_heat_capacity * np.outer(N, N) * detJ
                Ke += p.conductivity * (grad.T @ grad) * detJ
                He += p.effective_heat_transfer * np.outer(N, N) * detJ
                fe_ambient += (
                    p.effective_heat_transfer
                    * p.ambient_temperature
                    * N
                    * detJ
                )
        return Me, Ke, He, fe_ambient

# Assemble global matrices. This combines every local element matrix into one global system.
# This corresponds to:
# node 4 ---- node 3
#   |            |
# node 1 ---- node 2
    
    def _assemble(self):
        rows: List[int] = []
        cols: List[int] = []
        mvals: List[float] = []
        kvals: List[float] = []
        hvals: List[float] = []
        ambient = np.zeros(self.ndof, dtype=np.float64)

        Me, Ke, He, fea = self._local_matrices()

        for ei in range(self.nel):
            for ej in range(self.nel):
                nodes = [
                    self._node_id(ei, ej),
                    self._node_id(ei, ej + 1),
                    self._node_id(ei + 1, ej + 1),
                    self._node_id(ei + 1, ej),
                ]
                for a, A in enumerate(nodes):
                    ambient[A] += fea[a]
                    for b, B in enumerate(nodes):   # The local matrix entries are added to the corresponding global locations
                        rows.append(A)
                        cols.append(B)
                        mvals.append(Me[a, b])
                        kvals.append(Ke[a, b])
                        hvals.append(He[a, b])

        shape = (self.ndof, self.ndof)
        M = sparse.coo_matrix((mvals, (rows, cols)), shape=shape).tocsr()   # Sparse storage is used because most entries in an FE matrix are zero
        K = sparse.coo_matrix((kvals, (rows, cols)), shape=shape).tocsr()   
        H = sparse.coo_matrix((hvals, (rows, cols)), shape=shape).tocsr()
        return M, K, H, ambient

# Prepare backward-Euler time integration
# The continuous FE equation is:MT˙+(K+H)T=fsrc+f∞
# Using backward Euler:T˙≈Tn+1−Tn/Δt
# Therefore:(M/Δt+K+H)Tn+1=M/Δt *Tn+fsrc n+1+f∞
    def _prepare_time_step(self) -> None:
        p = self.params
        boundary = []
        free = []
        for i in range(self.nnode_side):    # outermost node rows and columns are fixed-temperature boundaries.
            for j in range(self.nnode_side):
                idx = self._node_id(i, j)
                if i in (0, self.nnode_side - 1) or j in (0, self.nnode_side - 1):   
                    boundary.append(idx)
                else:
                    free.append(idx)

        self.boundary = np.asarray(boundary, dtype=np.int64)
        self.free = np.asarray(free, dtype=np.int64)

        self.A = self.mass / p.time_step_s + self.stiffness + self.sink    # M/Δt+K+H
        self.Mdt = self.mass / p.time_step_s   # =M/Δt  

        # Partition the matrices
        Aff = self.A[self.free][:, self.free].tocsc()
        self.Afb = self.A[self.free][:, self.boundary].tocsr()
        self.Mff = self.Mdt[self.free][:, self.free].tocsr()
        self.Mfb = self.Mdt[self.free][:, self.boundary].tocsr()
        self.solve_free = factorized(Aff)

        # Reset the thermal field.  Every node starts at 300 K. outer boundary is explicitly fixed at 300 K.
    def reset(self) -> None:
        self.temperature = np.full(
            self.ndof,
            self.params.initial_temperature,
            dtype=np.float64,
        )
        self.temperature[self.boundary] = self.params.boundary_temperature
        self.peak_temperature = float(self.params.initial_temperature)

        # Convert RL grid position to physical position
    def _source_position_xy(self, row: int, col: int) -> Tuple[float, float]:
        # One padded element precedes the printable region.
        x = (col + 1.5) * self.dx   # 1.0 accounts for the padded element, and 0.5 places the source at the center of the printable element.
        y = (row + 1.5) * self.dy
        return x, y

    # Gaussian source distribution oUTPUT IS A nodal heat-load vector
    def _source_load(self, row: int, col: int, source_on: bool) -> np.ndarray:
        load = np.zeros(self.ndof, dtype=np.float64)   # when the laser is off, all nodal loads are zero
        if not source_on:
            return load

    # Distance from source to every node r2. 
        p = self.params
        x0, y0 = self._source_position_xy(row, col)
        r2 = (
            (self.node_xy[:, 0] - x0) ** 2
            + (self.node_xy[:, 1] - y0) ** 2
        )
        # Avoid numerical underflow for sigma much smaller than grid spacing.
        sigma_eff = max(p.gaussian_width_m, 0.20 * min(self.dx, self.dy))   # for n=10, dx=100 um, sigma eff=0.2*100 um=20 um
        w = np.exp(-0.5 * r2 / (sigma_eff ** 2))   # gaussian nodal weights
        w_sum = float(w.sum())
        if w_sum <= 0.0 or not np.isfinite(w_sum):  # If  Gaussian weights become zero/invalid,  code assigns the source to nearest node:
            nearest = int(np.argmin(r2))
            w[nearest] = 1.0
            w_sum = 1.0

        # Energy-conserving normalization
        # Discrete energy-conserving nodal distribution.
        load[:] = p.source_power_per_depth * w / w_sum
        return load

        # advance one thermal time step
    def step(self, row: int, col: int, source_on: bool) -> np.ndarray:
        p = self.params
        source = self._source_load(row, col, source_on)   # create the current source vector

        Tb = np.full(                     # boundary temp are fixed
            len(self.boundary),
            p.boundary_temperature,
            dtype=np.float64,
        )
        Tf_old = self.temperature[self.free]       # Previous free and boundary temperatures are extracted
        Tb_old = self.temperature[self.boundary]

    # Build the right-hand side
    # b=Mff/Δt*Tfn+Mfb/Δt*Tbn+fsrc,f +f∞,f −Afb*Tbn+1
        rhs = (
            self.Mff @ Tf_old
            + self.Mfb @ Tb_old
            + source[self.free]
            + self.ambient_load[self.free]
            - self.Afb @ Tb
        )

    # Solve for new free-node temperatures:  Aff*Tfn+1 = b
        Tf_new = self.solve_free(rhs)
        self.temperature[self.free] = Tf_new
        self.temperature[self.boundary] = Tb
    # Record peak temperature. This stores the maximum temperature reached over the entire toolpath, not only the current step.
        self.peak_temperature = max(
            self.peak_temperature,
            float(np.max(self.temperature)),
        )
        return self.cell_temperature()

    # Convert nodal temperatures to RL-cell temperatures. 
    # The RL environment uses one temperature per printable cell rather than nodal temperatures.
    # For a bilinear Q4 element, averaging the four nodal values is equivalent to evaluating the interpolated temperature at the element center: mean of T1 + T2 +T3+T4
    def cell_temperature(self) -> np.ndarray:
        out = np.zeros((self.n, self.n), dtype=np.float64)   
        # Printable elements are offset by one padded element.
        # For every printable element, the four corner-node IDs are collected.
        for i in range(self.n):
            for j in range(self.n):
                ei, ej = i + 1, j + 1
                nodes = [
                    self._node_id(ei, ej),
                    self._node_id(ei, ej + 1),
                    self._node_id(ei + 1, ej + 1),
                    self._node_id(ei + 1, ej),
                ]
                out[i, j] = float(np.mean(self.temperature[nodes]))
        return out

# ---------------------------------------------------------------------------
# Reinforcement-learning environments
# PBFBaseEnv defines the common reinforcement-learning environment logic used by both:
# the geometric PBF environment;
# the finite-element thermal PBF environment.

# It manages:
# the action space;
# laser movement;
# laser on/off state;
# target-mask generation;
# episode reset;
# reward calculation;
# completion checking;
# evaluation information;
# visualization.
# ---------------------------------------------------------------------------
@dataclass
class RewardWeights:
    fill: float = 1.5  # Positive reward for newly processing a cell that belongs to the target geometry
    spill: float = -0.15  # Penalty for processing a cell outside the target
    refill: float = -0.15  # Penalty for applying the source at a cell that had already been processed
    step: float = -1.5   # A penalty applied at every environment step. This encourages the agent to complete the target using fewer actions

# The reward is approximately:rt =1.5Nfill−0.15Nspill−0.15Nrefill−1.5−0.15Ninvalid

# This class inherits from Gymnasium's environment interface.
# It is a base class, meaning it contains shared functionality but is not intended to represent the full geometric or thermal environment by itself.
    
class PBFBaseEnv(gym.Env):
    """
    Common environment logic.

    Actions
    -------
    0: up
    1: down
    2: left
    3: right
    4: source toggle
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}    # tell Gymnasium that env can produce rgb_array images. Nominal speed 10 fps

    ACTION_TO_DELTA = {
        0: (-1, 0),   # row -1, move up
        1: (1, 0),    
        2: (0, -1),   # column -1, move left
        3: (0, 1),
    }

    def __init__(
        self,
        n: int = 10,
        split: Split = "train",
        max_steps: int = 200,
        reward_weights: Optional[RewardWeights] = None,
        seed: int = 0,
        fixed_mask_name: Optional[str] = None,
    ):
        super().__init__()
        self.n = int(n)
        self.split = split   # dataset split: train, validation, test
        self.max_steps = int(max_steps)
        self.reward_weights = reward_weights or RewardWeights() # use the supplied reward weights when provided; otherwise use the default values.
        self.fixed_mask_name = fixed_mask_name  # When this is none, a random target is generated, when specified, force a target like "A"
        self.rng = seed_everything(seed)   # random number, seed affects shape selection, size, rotation, initial laser position
        self.mask_generator = ProceduralMaskGenerator(n=n)  # This creates the object responsible for producing the binary target masks

        self.action_space = spaces.Discrete(5)  # This tells Gymnasium that the valid actions are: 0,1,2,3,4
        self.path: List[Tuple[int, int, bool]] = []  # Toolpath history, every recorded point contains (row, column, source_on)
        self._episode_seed = seed

    def _sample_start(self) -> Tuple[int, int]:  # this selects the starting grid cell
        if self.split == "train":   
            return (
                int(self.rng.integers(0, self.n)),  # The initial row and column are random
                int(self.rng.integers(0, self.n)),
            )
        return 0, 0   # for validation and testing, at fixed top left starting cell

    # Source-position observation channel: the current source position; whether the source is on or off
    def _source_channel(self) -> np.ndarray:
        channel = np.zeros((self.n, self.n), dtype=np.float32)   # initially all cells are zero
        # +1 means source on; -1 retains position while source is off.
        channel[self.row, self.col] = 1.0 if self.source_on else -1.0  
        return channel
        
    # move the laser. update the source position
    def _move(self, action: int) -> bool:
        if action not in self.ACTION_TO_DELTA:   # Action 4 is not a movement action, so it is treated as valid without changing position.
            return True
        dr, dc = self.ACTION_TO_DELTA[action]  # convert action to displacement
        nr, nc = self.row + dr, self.col + dc
        valid = 0 <= nr < self.n and 0 <= nc < self.n  # check grid boundary
        if valid:
            self.row, self.col = nr, nc
        return valid

    # Reset the common env state: It is called by the geometric and thermal subclasses
    def _base_reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = seed_everything(seed)
            self._episode_seed = seed

        # generate the target mask
        sample = self.mask_generator.sample(
            self.split,
            self.rng,
            fixed_name=self.fixed_mask_name,
        )
        self.mask = sample.mask.astype(bool)
        self.mask_name = sample.name
        self.size_factor = sample.size_factor
        self.rotation_deg = sample.rotation_deg

        self.phase = np.zeros((self.n, self.n), dtype=bool)  # initialize the phase map
        self.row, self.col = self._sample_start()  # initialize laser location
        self.source_on = False  # source begins off. the agent must use action 4 to turn it on
        self.steps = 0
        self.path = [(self.row, self.col, self.source_on)]
        self.last_reward_terms: Dict[str, float] = {}

    def _reward(
        self,
        old_phase: np.ndarray,
        moved_validly: bool,
    ) -> float:
        # This compares the previous and current phase maps and calculates the reward.
        newly = self.phase & ~old_phase  # This Boolean expression identifies cells that are processed now and were not processed before
        filled = int(np.count_nonzero(newly & self.mask))  # This counts newly processed cells that belong to the target.
        spilled = int(np.count_nonzero(newly & ~self.mask)) # background cells
        refill = int(self.source_on and old_phase[self.row, self.col])  # 1: the soruce is on and current source cell was processed before the action. 0: discourages the agent from repeatedly heating the same location
        invalid = int(not moved_validly)  # 1 for an invalid movement, 0 for a valid movement

        w = self.reward_weights
        reward = (
            w.fill * filled
            + w.spill * spilled
            + w.refill * refill
            + w.step
            + w.spill * invalid
        )
        self.last_reward_terms = {
            "filled": float(filled),
            "spilled": float(spilled),
            "refill": float(refill),
            "invalid_move": float(invalid),
            "step": 1.0,
        }
        return float(reward)

    def _done(self) -> bool:
        return bool(np.all(self.phase[self.mask]))

    def _info(self) -> Dict[str, object]:
        target_count = max(1, int(self.mask.sum()))
        filled_count = int(np.count_nonzero(self.phase & self.mask))
        return {
            "mask_name": self.mask_name,
            "coverage": filled_count / target_count,
            "filled_target_cells": filled_count,
            "target_cells": target_count,
            "steps": self.steps,
            "source_on": self.source_on,
            "reward_terms": dict(self.last_reward_terms),
        }

    def render(self):
        # RGB visualization without requiring Matplotlib.
        rgb = np.zeros((self.n, self.n, 3), dtype=np.uint8)
        rgb[self.mask] = (220, 220, 220)
        rgb[self.phase & self.mask] = (120, 80, 200)
        rgb[self.phase & ~self.mask] = (220, 80, 80)
        rgb[self.row, self.col] = (255, 215, 0) if self.source_on else (0, 180, 255)
        return np.kron(rgb, np.ones((24, 24, 1), dtype=np.uint8))


class GeometricPBFEnv(PBFBaseEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3, self.n, self.n),
            dtype=np.float32,
        )

    def _observation(self) -> np.ndarray:
        return np.stack(
            [
                self.mask.astype(np.float32),
                self.phase.astype(np.float32),
                self._source_channel(),
            ],
            axis=0,
        ).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options=None):
        if GYMNASIUM_AVAILABLE:
            super().reset(seed=seed)
        self._base_reset(seed=seed)
        return self._observation(), self._info()

    def step(self, action: int):
        if not 0 <= int(action) < 5:
            raise ValueError(f"Invalid action {action}")

        old_phase = self.phase.copy()
        moved_validly = True

        if int(action) == 4:
            self.source_on = not self.source_on
        else:
            moved_validly = self._move(int(action))

        if self.source_on:
            self.phase[self.row, self.col] = True

        self.steps += 1
        self.path.append((self.row, self.col, self.source_on))
        reward = self._reward(old_phase, moved_validly)
        terminated = self._done()
        truncated = self.steps >= self.max_steps and not terminated
        return self._observation(), reward, terminated, truncated, self._info()

# What ThermalPBFEnv does

# ThermalPBFEnv is the reinforcement-learning environment that combines:

# laser motion+laser on/off control+finite-element thermal simulation+phase-change tracking

# At every RL step, the environment:
# receives an action;
# moves the laser or toggles it on/off;
# advances the finite-element thermal model;
# identifies cells above the melting threshold;
# updates the processed-region map;
# computes the reward;
# returns the next observation.

# It inherits movement, mask generation, reward calculation, completion checking, and logging from
    
class ThermalPBFEnv(PBFBaseEnv):
    def __init__(
        self,
        thermal_params: Optional[ThermalParameters] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.thermal_params = thermal_params or ThermalParameters()
        self.solver = StructuredQ4ThermalSolver(self.n, self.thermal_params)
        self.temperature = np.full(
            (self.n, self.n),
            self.thermal_params.initial_temperature,
            dtype=np.float64,
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=2.0,
            shape=(4, self.n, self.n),
            dtype=np.float32,
        )

    def _normalized_temperature(self) -> np.ndarray:
        p = self.thermal_params
        scale = max(1.0, p.melt_threshold - p.initial_temperature)
        normalized = (self.temperature - p.initial_temperature) / scale
        return np.clip(normalized, 0.0, 2.0).astype(np.float32)

    def _observation(self) -> np.ndarray:
        return np.stack(
            [
                self.mask.astype(np.float32),
                self._source_channel(),
                self._normalized_temperature(),
                self.phase.astype(np.float32),
            ],
            axis=0,
        ).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options=None):
        if GYMNASIUM_AVAILABLE:
            super().reset(seed=seed)
        self._base_reset(seed=seed)
        self.solver.reset()
        self.temperature = self.solver.cell_temperature()
        return self._observation(), self._info()

    def step(self, action: int):
        if not 0 <= int(action) < 5:
            raise ValueError(f"Invalid action {action}")

        old_phase = self.phase.copy()
        moved_validly = True

        if int(action) == 4:
            self.source_on = not self.source_on
        else:
            moved_validly = self._move(int(action))

        self.temperature = self.solver.step(
            self.row,
            self.col,
            self.source_on,
        )
        newly_hot = self.temperature > self.thermal_params.melt_threshold
        self.phase |= newly_hot

        self.steps += 1
        self.path.append((self.row, self.col, self.source_on))
        reward = self._reward(old_phase, moved_validly)
        terminated = self._done()
        truncated = self.steps >= self.max_steps and not terminated
        info = self._info()
        info["peak_temperature"] = float(self.solver.peak_temperature)
        info["mean_temperature"] = float(np.mean(self.temperature))
        return self._observation(), reward, terminated, truncated, info

    def render(self):
        p = self.thermal_params
        norm = np.clip(
            (self.temperature - p.initial_temperature)
            / max(1.0, p.melt_threshold - p.initial_temperature),
            0.0,
            1.5,
        )
        heat = np.zeros((self.n, self.n, 3), dtype=np.float32)
        heat[..., 0] = np.clip(norm, 0, 1)
        heat[..., 1] = np.clip(1.2 - np.abs(norm - 0.7), 0, 1)
        heat[..., 2] = np.clip(1.0 - norm, 0, 1)
        heat[~self.mask] *= 0.35
        heat[self.row, self.col] = (1.0, 1.0, 1.0)
        rgb = (255 * np.clip(heat, 0, 1)).astype(np.uint8)
        return np.kron(rgb, np.ones((24, 24, 1), dtype=np.uint8))


# ---------------------------------------------------------------------------
# Baselines and evaluation
# ---------------------------------------------------------------------------
def _toggle_to(env: PBFBaseEnv, desired: bool) -> Tuple[float, bool, bool]:
    total_reward = 0.0
    terminated = truncated = False
    if env.source_on != desired:
        _, r, terminated, truncated, _ = env.step(4)
        total_reward += r
    return total_reward, terminated, truncated


def run_zigzag_episode(env: PBFBaseEnv) -> Dict[str, float]:
    """
    Serpentine row-by-row baseline.

    It turns the source on over target cells and off elsewhere. Movement to the
    first row is completed with the source off.
    """
    _, info = env.reset()
    total_reward = 0.0
    terminated = truncated = False

    order: List[Tuple[int, int]] = []
    for r in range(env.n):
        cols = range(env.n) if r % 2 == 0 else range(env.n - 1, -1, -1)
        for c in cols:
            order.append((r, c))

    def move_one(action: int):
        nonlocal total_reward, terminated, truncated
        _, r, terminated, truncated, _ = env.step(action)
        total_reward += r

    # Navigate to the first ordered cell with the source off.
    total_reward += _toggle_to(env, False)[0]
    target_r, target_c = order[0]
    while env.row > target_r and not (terminated or truncated):
        move_one(0)
    while env.row < target_r and not (terminated or truncated):
        move_one(1)
    while env.col > target_c and not (terminated or truncated):
        move_one(2)
    while env.col < target_c and not (terminated or truncated):
        move_one(3)

    for idx, (r, c) in enumerate(order):
        if terminated or truncated:
            break

        desired = bool(env.mask[r, c] and not env.phase[r, c])
        rt, terminated, truncated = _toggle_to(env, desired)
        total_reward += rt
        if terminated or truncated:
            break

        if idx + 1 < len(order):
            nr, nc = order[idx + 1]
            if nr < r:
                action = 0
            elif nr > r:
                action = 1
            elif nc < c:
                action = 2
            else:
                action = 3
            move_one(action)

    info = env._info()
    return {
        "reward": float(total_reward),
        "steps": float(env.steps),
        "coverage": float(info["coverage"]),
        "speed": float(info["filled_target_cells"] / max(1, env.steps)),
        "peak_temperature": float(
            getattr(getattr(env, "solver", None), "peak_temperature", np.nan)
        ),
    }


def evaluate_policy_callable(
    policy: Callable[[np.ndarray], int],
    env_factory: Callable[[], PBFBaseEnv],
    episodes: int = 50,
) -> Dict[str, float]:
    rewards, lengths, speeds, peaks, coverages = [], [], [], [], []

    for ep in range(episodes):
        env = env_factory()
        obs, _ = env.reset(seed=10_000 + ep)
        total_reward = 0.0

        for _ in range(env.max_steps):
            action = int(policy(obs))
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        rewards.append(total_reward)
        lengths.append(env.steps)
        coverages.append(float(info["coverage"]))
        speeds.append(float(info["filled_target_cells"] / max(1, env.steps)))
        peaks.append(float(info.get("peak_temperature", np.nan)))

    def mean_ci95(x: Sequence[float]) -> Tuple[float, float]:
        arr = np.asarray(x, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return float("nan"), float("nan")
        mean = float(np.mean(arr))
        if arr.size == 1:
            return mean, 0.0
        half = 1.96 * float(np.std(arr, ddof=1)) / math.sqrt(arr.size)
        return mean, half

    output = {}
    for key, values in {
        "reward": rewards,
        "episode_length": lengths,
        "speed": speeds,
        "coverage": coverages,
        "peak_temperature": peaks,
    }.items():
        mean, ci = mean_ci95(values)
        output[f"{key}_mean"] = mean
        output[f"{key}_ci95_half_width"] = ci
    return output


# ---------------------------------------------------------------------------
# Stable-Baselines3 integration
# ---------------------------------------------------------------------------
def require_sb3():
    if not GYMNASIUM_AVAILABLE:
        raise ImportError(
            "Gymnasium is required for training. Install with:\n"
            "pip install gymnasium stable-baselines3"
        )
    try:
        import torch
        from stable_baselines3 import DQN, PPO
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    except ImportError as exc:
        raise ImportError(
            "Stable-Baselines3 and PyTorch are required for training. Install with:\n"
            "pip install torch gymnasium stable-baselines3"
        ) from exc
    return torch, DQN, PPO, BaseFeaturesExtractor


def build_tiny_cnn_extractor():
    torch, _, _, BaseFeaturesExtractor = require_sb3()
    import torch.nn as nn

    class TinyCNN(BaseFeaturesExtractor):
        def __init__(self, observation_space, features_dim: int = 256):
            super().__init__(observation_space, features_dim)
            channels = int(observation_space.shape[0])
            self.cnn = nn.Sequential(
                nn.Conv2d(channels, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                sample = torch.as_tensor(
                    observation_space.sample()[None],
                    dtype=torch.float32,
                )
                n_flat = int(self.cnn(sample).shape[1])
            self.linear = nn.Sequential(
                nn.Linear(n_flat, features_dim),
                nn.ReLU(),
            )

        def forward(self, observations):
            return self.linear(self.cnn(observations))

    return TinyCNN


def make_env(
    env_kind: EnvKind,
    n: int,
    split: Split,
    seed: int,
    fixed_mask_name: Optional[str] = None,
) -> PBFBaseEnv:
    common = dict(
        n=n,
        split=split,
        seed=seed,
        fixed_mask_name=fixed_mask_name,
        max_steps=max(200, 2 * n * n),
    )
    if env_kind == "geometric":
        return GeometricPBFEnv(**common)
    if env_kind == "thermal":
        return ThermalPBFEnv(**common)
    raise ValueError(f"Unknown environment kind: {env_kind}")


def train_sb3(
    algorithm: Literal["dqn", "ppo"],
    env_kind: EnvKind,
    total_timesteps: int,
    n: int = 10,
    seed: int = 0,
    output_path: str = "trained_agent",
):
    _, DQN, PPO, _ = require_sb3()
    TinyCNN = build_tiny_cnn_extractor()

    env = make_env(env_kind, n=n, split="train", seed=seed)

    if algorithm == "dqn":
        policy_kwargs = dict(
            features_extractor_class=TinyCNN,
            features_extractor_kwargs=dict(features_dim=512),
            net_arch=[32],
            normalize_images=False,
        )
        model = DQN(
            "CnnPolicy",
            env,
            learning_rate=9.0e-5,
            gamma=0.995,
            buffer_size=200_000,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            target_update_interval=2_000,
            learning_starts=2_000,
            batch_size=256,
            train_freq=4,
            gradient_steps=1,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=seed,
        )
    elif algorithm == "ppo":
        policy_kwargs = dict(
            features_extractor_class=TinyCNN,
            features_extractor_kwargs=dict(features_dim=256),
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
            normalize_images=False,
        )
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=4.7e-4,
            gamma=0.995,
            clip_range=0.1,
            gae_lambda=0.8,
            n_steps=2048,
            batch_size=256,
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=seed,
        )
    else:
        raise ValueError("algorithm must be 'dqn' or 'ppo'")

    model.learn(total_timesteps=int(total_timesteps))
    model.save(output_path)
    return model


def evaluate_sb3_model(
    model_path: str,
    algorithm: Literal["dqn", "ppo"],
    env_kind: EnvKind,
    n: int = 10,
    episodes: int = 50,
    seed: int = 123,
) -> Dict[str, float]:
    _, DQN, PPO, _ = require_sb3()
    model_cls = DQN if algorithm == "dqn" else PPO
    model = model_cls.load(model_path)

    def policy(obs: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return evaluate_policy_callable(
        policy,
        lambda: make_env(env_kind, n=n, split="test", seed=seed),
        episodes=episodes,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_episode(env: PBFBaseEnv, title: Optional[str] = None):
    import matplotlib.pyplot as plt

    path = np.asarray([(r, c) for r, c, _ in env.path], dtype=float)
    source_on = np.asarray([on for _, _, on in env.path], dtype=bool)

    fig, axes = plt.subplots(1, 3 if isinstance(env, ThermalPBFEnv) else 2, figsize=(12, 4))
    axes = np.atleast_1d(axes)

    axes[0].imshow(env.mask, origin="upper", cmap="gray_r")
    axes[0].set_title("Target mask")

    axes[1].imshow(env.phase, origin="upper", cmap="Purples", vmin=0, vmax=1)
    if len(path):
        axes[1].plot(path[:, 1], path[:, 0], linewidth=1.2)
        if source_on.any():
            axes[1].scatter(
                path[source_on, 1],
                path[source_on, 0],
                s=8,
                marker=".",
            )
    axes[1].set_title("Processed phase and path")

    if isinstance(env, ThermalPBFEnv):
        im = axes[2].imshow(env.temperature, origin="upper")
        axes[2].set_title(
            f"Temperature [K]\npeak={env.solver.peak_temperature:.1f}"
        )
        fig.colorbar(im, ax=axes[2], shrink=0.8)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="Run a short environment smoke test.")
    smoke.add_argument("--env", choices=["geometric", "thermal"], default="thermal")
    smoke.add_argument("--n", type=int, default=10)
    smoke.add_argument("--steps", type=int, default=20)
    smoke.add_argument("--seed", type=int, default=0)

    zig = sub.add_parser("zigzag", help="Run the deterministic zigzag baseline.")
    zig.add_argument("--env", choices=["geometric", "thermal"], default="thermal")
    zig.add_argument("--n", type=int, default=10)
    zig.add_argument("--seed", type=int, default=0)

    train = sub.add_parser("train", help="Train DQN or PPO using Stable-Baselines3.")
    train.add_argument("--algo", choices=["dqn", "ppo"], required=True)
    train.add_argument("--env", choices=["geometric", "thermal"], required=True)
    train.add_argument("--steps", type=int, default=100_000)
    train.add_argument("--n", type=int, default=10)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--output", type=str, default="trained_agent")

    args = parser.parse_args()

    if args.command == "smoke":
        env = make_env(args.env, n=args.n, split="test", seed=args.seed)
        obs, info = env.reset(seed=args.seed)
        total = 0.0
        for _ in range(args.steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            if terminated or truncated:
                break
        print({
            "observation_shape": obs.shape,
            "reward": total,
            **info,
        })

    elif args.command == "zigzag":
        env = make_env(args.env, n=args.n, split="test", seed=args.seed)
        print(run_zigzag_episode(env))

    elif args.command == "train":
        train_sb3(
            algorithm=args.algo,
            env_kind=args.env,
            total_timesteps=args.steps,
            n=args.n,
            seed=args.seed,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()
