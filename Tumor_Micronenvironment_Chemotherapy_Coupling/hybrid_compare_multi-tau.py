import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, distance
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
import random
import os
import time
import sys
import functools
print = functools.partial(print, flush=True)

# Output directory
output_dir = "/home/wfarz001/PhD_candidacy_exam/hybrid_compare_multi-tau"
os.makedirs(output_dir, exist_ok=True)



class DiffusionReactionModel:
    """Diffusion-Reaction model for drug transport"""
    
    def __init__(self, domain_size=100, grid_spacing=2.0, D0=1e-5, K_met=2.0e-4/60, 
                 lambda_14=2.5e-4, phi_0=0.1, tau_cycle=24*3600, tau_decay=600*60):
        """
        Initialize diffusion-reaction model
        
        Parameters:
        domain_size: Domain size in mm
        grid_spacing: Grid spacing in mm
        D0: Drug diffusion coefficient 
        K_met: Drug decomposition rate 
        lambda_14: Drug consumption rate coefficient
        phi_0: Half-saturation concentration
        tau_cycle: Dosing cycle time (s)
        tau_decay: Decay time constant (s)
        """
        self.domain_size = domain_size
        self.dx = grid_spacing
        self.D0 = D0
        self.K_met = K_met
        self.lambda_14 = lambda_14
        self.phi_0 = phi_0
        self.tau_cycle = tau_cycle
        self.tau_decay = tau_decay
        
        # Grid setup
        self.nx = int(domain_size / grid_spacing) + 1
        self.ny = int(domain_size / grid_spacing) + 1
        
        # Create coordinate grids
        x = np.linspace(-domain_size/2, domain_size/2, self.nx)
        y = np.linspace(-domain_size/2, domain_size/2, self.ny)
        self.X, self.Y = np.meshgrid(x, y)
        
        # Initialize drug concentration field
        self.phi = np.zeros((self.ny, self.nx))
        
        # Initialize cell density and ECM density fields
        self.n_density = np.zeros((self.ny, self.nx))
        self.rho_ECM = np.ones((self.ny, self.nx)) * 0.9
        
        # Create circular domain mask
        self.domain_mask = (self.X**2 + self.Y**2) <= (domain_size/2)**2
        
        # Time stepping
        self.current_time = 0
        
        # Unit volume for cell density calculation
        self.delta_V = self.dx**2  # 2D unit area 
    
    def S_t(self, t, C0=1.0):
        """Periodic dosing function at tumor boundary"""
        cycle_time = t % self.tau_cycle
        return C0 / (1 + np.exp((cycle_time - self.tau_decay)/(self.tau_decay/10)))
    
    def apply_boundary_conditions(self, phi_new):
        """Apply boundary conditions - periodic dosing at domain boundary"""
        boundary_value = self.S_t(self.current_time)
        
        # Apply boundary conditions at circular boundary
        for i in range(self.ny):
            for j in range(self.nx):
                if self.domain_mask[i, j]:
                    # Check if point is near boundary
                    r = np.sqrt(self.X[i, j]**2 + self.Y[i, j]**2)
                    if r > (self.domain_size/2 - self.dx):
                        phi_new[i, j] = boundary_value
                else:
                    phi_new[i, j] = 0  # Outside domain
        
        return phi_new
    
    def diffusion_step(self, dt_step):
        """Perform one diffusion-reaction step"""
        phi_new = self.phi.copy()
        
        # Apply diffusion-reaction equation using finite differences
        for i in range(1, self.ny-1):
            for j in range(1, self.nx-1):
                if self.domain_mask[i, j]:
                    # Laplacian operator
                    laplacian = (self.phi[i+1, j] + self.phi[i-1, j] + 
                               self.phi[i, j+1] + self.phi[i, j-1] - 
                               4*self.phi[i, j]) / (self.dx**2)
                    
                    # Consumption term - updated with lambda_0 from CA model
                    consumption = self.lambda_14 * self.n_density[i, j] * \
                                self.phi[i, j] / (self.phi[i, j] + self.phi_0)
                    
                    # Update equation: f_(n+1) gets updated from f_n, D_n and ?_0
                    phi_new[i, j] = self.phi[i, j] + dt_step * (
                        self.D0 * laplacian - 
                        self.K_met * self.phi[i, j] - 
                        consumption
                    )
                    
                    # Ensure non-negative concentrations
                    phi_new[i, j] = max(0, phi_new[i, j])
        
        # Apply boundary conditions
        phi_new = self.apply_boundary_conditions(phi_new)
        self.phi = phi_new
        self.current_time += dt_step
    
    def update_cell_density(self, voronoi_model):
        """
        Update cell density field from Voronoi CA model using proper mapping
        
        """
        self.n_density.fill(0)
        
        # For each grid point, find neighboring Voronoi cells and count tumor cells
        for i in range(self.ny):
            for j in range(self.nx):
                if self.domain_mask[i, j]:
                    # Get grid point coordinates
                    grid_x = self.X[i, j]
                    grid_y = self.Y[i, j]
                    grid_point = np.array([grid_x, grid_y])
                    
                    # Find Voronoi cells within influence radius of this grid point
                    influence_radius = self.dx * 1.5  # Slightly larger than grid spacing
                    distances = np.linalg.norm(voronoi_model.points - grid_point, axis=1)
                    nearby_cells = np.where(distances <= influence_radius)[0]
                    
                    # Count tumor cells (n_m = 1 for tumor cells, 0 for ECM)
                    tumor_cell_count = 0
                    for cell_idx in nearby_cells:
                        cell_type = voronoi_model.cell_types[cell_idx]
                        # n_m = 1 if it's a tumor cell (not ECM or degraded ECM)
                        if cell_type not in [voronoi_model.ECM, voronoi_model.DEGRADED_ECM]:
                            tumor_cell_count += 1
                    
                    # Apply the formula: n_(i,j)
                    self.n_density[i, j] = tumor_cell_count / self.delta_V
    
    def update_ecm_density(self, voronoi_model):
        """Update ECM density field from Voronoi CA model with proper mapping"""
        self.rho_ECM.fill(0.9)
        
        # For each grid point, interpolate ECM density from nearby Voronoi cells
        for i in range(self.ny):
            for j in range(self.nx):
                if self.domain_mask[i, j]:
                    grid_x = self.X[i, j]
                    grid_y = self.Y[i, j]
                    grid_point = np.array([grid_x, grid_y])
                    
                    # Find nearby Voronoi cells
                    influence_radius = self.dx * 1.5
                    distances = np.linalg.norm(voronoi_model.points - grid_point, axis=1)
                    nearby_cells = np.where(distances <= influence_radius)[0]
                    
                    if len(nearby_cells) > 0:
                        # Weight-averaged ECM density based on distance
                        weights = 1.0 / (distances[nearby_cells] + 1e-10)
                        weighted_ecm = np.sum(weights * voronoi_model.ecm_density[nearby_cells])
                        total_weight = np.sum(weights)
                        self.rho_ECM[i, j] = weighted_ecm / total_weight
    
    def get_drug_concentration_at_point(self, point):
        """Get drug concentration at a specific point using bilinear interpolation"""
        # Convert point to grid coordinates
        x_coord = (point[0] + self.domain_size/2) / self.dx
        y_coord = (point[1] + self.domain_size/2) / self.dx
        
        # Get integer grid indices
        i = int(y_coord)
        j = int(x_coord)
        
        # Check bounds
        if i < 0 or i >= self.ny-1 or j < 0 or j >= self.nx-1:
            return 0.0
        
        # Bilinear interpolation weights
        di = y_coord - i
        dj = x_coord - j
        
        # Interpolate
        concentration = (1-di)*(1-dj)*self.phi[i, j] + \
                       (1-di)*dj*self.phi[i, j+1] + \
                       di*(1-dj)*self.phi[i+1, j] + \
                       di*dj*self.phi[i+1, j+1]
        
        return concentration
        
######################################################### Another Class for CA Model #######################################################

class HybridVoronoiTumorCA:
    """Simplified Hybrid Voronoi-based CA model focusing on core mechanics"""
    
    def __init__(self, radius=50, n_seeds=1200, P_gamma=0.6):
        """Initialize hybrid model with all parameters"""
        self.radius = radius
        self.n_seeds = n_seeds
        self.P_gamma = P_gamma  # Drug effectiveness parameter
        
        # Model parameters from Table 1
        self.p0 = 0.192       # Base probability of division
        self.a = 0.12         # Base necrotic thickness 
        self.b = 0.08         # Base proliferative thickness 
        self.gamma = 0.05     # Mutation rate for invasive cells 
        self.A_i = 2          # Adhesion value
        self.mu = 3           # Mobility of invasive cells 
        self.chi = 0.15       # ECM degradation ability
        
        # Cell type constants
        self.NECROTIC = 0
        self.QUIESCENT = 1  
        self.PROLIFERATIVE = 2
        self.INVASIVE = 3
        self.ECM = 4
        self.DEGRADED_ECM = 5
        
        # Generate Voronoi diagram
        self.points = self.generate_seeds()
        self.vor = Voronoi(self.points)
        
        # Initialize cell types and ECM density
        self.cell_types = np.zeros(len(self.points), dtype=int)
        self.ecm_density = np.ones(len(self.points)) * 0.9
        self.initialize_tumor_with_target_invasive()
        
        # Drug concentration at each cell location
        self.drug_concentration = np.zeros(len(self.points))

        # Simulation tracking
        self.day = 0
        self.radius_history = []
        self.invasive_radius_history = []
        self.time_history = []
        
        # Calculate initial tumor centroid
        self.update_tumor_centroid()

        # Record initial state
        self.radius_history.append(self.calculate_proliferative_radius())
        self.invasive_radius_history.append(self.calculate_invasive_radius())
        self.time_history.append(0)
    
    def generate_seeds(self):
        """Generate seed points within circular domain"""
        points = []
        while len(points) < self.n_seeds:
            x = random.uniform(-self.radius * 1.2, self.radius * 1.2)
            y = random.uniform(-self.radius * 1.2, self.radius * 1.2)
            if np.sqrt(x**2 + y**2) <= self.radius:
                points.append([x, y])
        return np.array(points)
    
    def initialize_tumor_with_target_invasive(self):
        """Initialize tumor to have approximately 142 invasive cells"""
        self.cell_types.fill(self.ECM)
        self.ecm_density.fill(0.3)
        
        tumor_cells = []
        for i, point in enumerate(self.points):
            dist_from_center = np.linalg.norm(point)
            if dist_from_center < 12:  # Core region
                tumor_cells.append(i)
        
        random.shuffle(tumor_cells)
        n_tumor = len(tumor_cells)
        
        if n_tumor > 0:
            n_invasive = min(142, int(n_tumor * 0.25))
            n_proliferative = int(n_tumor * 0.45)
            n_quiescent = int(n_tumor * 0.20)
            n_necrotic = n_tumor - n_invasive - n_proliferative - n_quiescent
            
            idx = 0
            # Assign cell types
            for _ in range(n_invasive):
                if idx < len(tumor_cells):
                    self.cell_types[tumor_cells[idx]] = self.INVASIVE
                    self.ecm_density[tumor_cells[idx]] = 0.3
                    idx += 1
            
            for _ in range(n_proliferative):
                if idx < len(tumor_cells):
                    self.cell_types[tumor_cells[idx]] = self.PROLIFERATIVE
                    self.ecm_density[tumor_cells[idx]] = 0.3
                    idx += 1
            
            for _ in range(n_quiescent):
                if idx < len(tumor_cells):
                    self.cell_types[tumor_cells[idx]] = self.QUIESCENT
                    self.ecm_density[tumor_cells[idx]] = 0.3
                    idx += 1
            
            for i in range(idx, len(tumor_cells)):
                self.cell_types[tumor_cells[i]] = self.NECROTIC
                self.ecm_density[tumor_cells[i]] = 0.3
        
        # Create degraded ECM around tumor
        for i, point in enumerate(self.points):
            if self.cell_types[i] == self.ECM:
                dist_from_center = np.linalg.norm(point)
                if 12 <= dist_from_center < 20:
                    if random.random() < 0.6:
                        self.cell_types[i] = self.DEGRADED_ECM
                        self.ecm_density[i] = 0.3
    
    def update_tumor_centroid(self):
        """Update tumor centroid based on current tumor cell positions"""
        tumor_indices = np.where((self.cell_types != self.ECM) & 
                                (self.cell_types != self.DEGRADED_ECM))[0]
        
        if len(tumor_indices) > 0:
            tumor_points = self.points[tumor_indices]
            self.tumor_centroid = np.mean(tumor_points, axis=0)
        else:
            self.tumor_centroid = np.array([0, 0])
    
    def get_neighbors(self, cell_idx, radius=4.0):
        """Get neighboring cell indices within specified radius"""
        distances = distance.cdist([self.points[cell_idx]], self.points).flatten()
        return np.where((distances > 0) & (distances < radius))[0]

    def calculate_proliferative_radius(self):
        """Calculate current proliferative tumor radius in mm"""
        proliferative_indices = np.where(self.cell_types == self.PROLIFERATIVE)[0]
        
        if len(proliferative_indices) == 0:
            return 0
        
        proliferative_points = self.points[proliferative_indices]
        distances = np.linalg.norm(proliferative_points - self.tumor_centroid, axis=1)
        return np.max(distances)
        
    def calculate_invasive_radius(self):
        """Calculate current invasive tumor radius in mm"""
        invasive_indices = np.where(self.cell_types == self.INVASIVE)[0]
        
        if len(invasive_indices) == 0:
            return 0
        
        invasive_points = self.points[invasive_indices]
        distances = np.linalg.norm(invasive_points - self.tumor_centroid, axis=1)
        return np.max(distances)
        
    def calculate_Lmax(self, cell_idx):
        """Calculate L_max - distance from tumor centroid to domain boundary"""
        cell_pos = self.points[cell_idx]
        direction = cell_pos - self.tumor_centroid
        direction_norm = np.linalg.norm(direction)
        
        if direction_norm < 1e-6:
            return self.radius - np.linalg.norm(self.tumor_centroid)
        
        direction_unit = direction / direction_norm
        
        # Find intersection with circular boundary
        a = np.dot(direction_unit, direction_unit)
        b = 2 * np.dot(self.tumor_centroid, direction_unit)
        c = np.dot(self.tumor_centroid, self.tumor_centroid) - self.radius**2
        
        discriminant = b**2 - 4*a*c
        if discriminant < 0:
            return self.radius
        
        t = (-b + np.sqrt(discriminant)) / (2*a)
        L_max = np.linalg.norm(self.tumor_centroid + t * direction_unit - self.tumor_centroid)
        
        return max(L_max, 1.0)
    
    def calculate_rho_ECM(self, cell_idx):
        """Calculate local ECM density around a cell"""
        neighbors = self.get_neighbors(cell_idx, radius=6.0)
        if len(neighbors) == 0:
            return 0.0
        
        neighbor_densities = [self.ecm_density[j] for j in neighbors]
        return np.mean(neighbor_densities)
    
    def calculate_P_gamma_phi(self, drug_concentration):
        """Calculate drug effect factor P_?,f"""
        return 1 - (1 - self.P_gamma) * drug_concentration
    
    def calculate_Pdiv(self, cell_idx):
        """Calculate division probability with drug effect"""
        # Distance from tumor centroid to dividing cell
        r = np.linalg.norm(self.points[cell_idx] - self.tumor_centroid)
        
        # Calculate L_max
        L_max = self.calculate_Lmax(cell_idx)
        
        # Calculate local ECM density
        rho_ECM = self.calculate_rho_ECM(cell_idx)
        
        # Get drug concentration at cell location
        drug_conc = self.drug_concentration[cell_idx]
        
        # Calculate drug effect
        P_gamma_phi = self.calculate_P_gamma_phi(drug_conc)
        
        # Apply the division probability formula with drug effect
        if L_max > 0:
            term1 = max(0, 1 - (r / L_max))
            term2 = max(0, 1 - rho_ECM)
            P_div = self.p0 * P_gamma_phi * (term1 + term2) / 2
        else:
            P_div = 0
        
        return np.clip(P_div, 0, 1)
    
    def update_drug_concentrations(self, diffusion_model):
        """Update drug concentrations at cell locations from diffusion model"""
        for i, point in enumerate(self.points):
            self.drug_concentration[i] = diffusion_model.get_drug_concentration_at_point(point)
    
    def get_nutrient_concentration(self, cell_idx):
        """Calculate nutrient concentration at cell location"""
        cell_pos = self.points[cell_idx]
        dist_to_boundary = self.radius - np.linalg.norm(cell_pos)
        
        # Base nutrient level based on distance to boundary (oxygen diffusion)
        base_nutrient = max(0, dist_to_boundary / self.radius)
        
        # Reduce nutrient based on local tumor cell density
        neighbors = self.get_neighbors(cell_idx, radius=8.0)
        tumor_neighbors = sum(1 for j in neighbors 
                            if self.cell_types[j] not in [self.ECM, self.DEGRADED_ECM])
        
        nutrient_consumption = min(0.8, tumor_neighbors * 0.05)  # Max 80% reduction
        
        return max(0.1, base_nutrient - nutrient_consumption)  # Minimum 10% nutrient
    
    def select_proliferative_cells_to_divide(self, N_cells):
        """Select N_cells proliferative cells to divide based on division probabilities"""
        proliferative_indices = np.where(self.cell_types == self.PROLIFERATIVE)[0]
        
        if len(proliferative_indices) == 0:
            return []
        
        # Calculate division probabilities for all proliferative cells
        division_probs = []
        for idx in proliferative_indices:
            P_div = self.calculate_Pdiv(idx)
            division_probs.append(P_div)
        
        division_probs = np.array(division_probs)
        
        # Select N_cells based on probabilities (weighted random selection)
        if np.sum(division_probs) > 0:
            # Normalize probabilities
            normalized_probs = division_probs / np.sum(division_probs)
            
            # Select N_cells (with replacement if necessary)
            N_select = min(N_cells, len(proliferative_indices))
            selected_indices = np.random.choice(
                proliferative_indices, 
                size=N_select, 
                replace=False, 
                p=normalized_probs
            )
            return selected_indices.tolist()
        else:
            return []
    
    def apply_proliferative_division_with_selection(self, selected_cells):
        """Enhanced division with all rules including invasive mutation (Rule 4)"""
        new_cell_types = self.cell_types.copy()
        new_ecm_density = self.ecm_density.copy()
        
        divisions_occurred = 0
        newly_invasive_cells = []
        
        for i in selected_cells:
            if self.cell_types[i] == self.PROLIFERATIVE:
                neighbors = self.get_neighbors(i, radius=5.0)
                available_neighbors = [j for j in neighbors 
                                     if self.cell_types[j] in [self.ECM, self.DEGRADED_ECM]]
                
                if available_neighbors:
                    daughter_idx = random.choice(available_neighbors)
                    
                    # Rule 4: Daughter cell can become invasive with probability 
                    if random.random() < self.gamma:  #  0.05
                        # Additional condition: number of neighbors < A_i
                        daughter_neighbors = self.get_neighbors(daughter_idx, radius=5.0)
                        tumor_neighbors = sum(1 for j in daughter_neighbors 
                                            if self.cell_types[j] != self.ECM and 
                                               self.cell_types[j] != self.DEGRADED_ECM)
                        
                        if tumor_neighbors < self.A_i:
                            new_cell_types[daughter_idx] = self.INVASIVE
                            newly_invasive_cells.append(daughter_idx)
                        else:
                            new_cell_types[daughter_idx] = self.PROLIFERATIVE
                    else:
                        new_cell_types[daughter_idx] = self.PROLIFERATIVE
                    
                    new_ecm_density[daughter_idx] = 0.0
                    divisions_occurred += 1
        
        # Update the state first
        self.cell_types = new_cell_types
        self.ecm_density = new_ecm_density
        
        # Apply invasive cell migration rule to newly created invasive cells
        for invasive_idx in newly_invasive_cells:
            self.process_invasive_cell_migration(invasive_idx)
        
        return divisions_occurred
    
    def process_invasive_cell_migration(self, cell_idx):
        """Process invasive cell migration with ECM degradation"""
        # Random number of attempts 
        m = random.randint(0, self.mu)  
        
        # Get neighboring ECM cells
        neighbors = self.get_neighbors(cell_idx, radius=6.0)
        ecm_neighbors = [j for j in neighbors 
                        if self.cell_types[j] in [self.ECM, self.DEGRADED_ECM]]
        
        if not ecm_neighbors:
            return False  # No migration occurred
    
    def calculate_Lt(self, cell_idx):
        """Calculate L_t: distance from centroid to closest edge cell for given cell"""
        cell_pos = self.points[cell_idx]
        
        # Get all edge cells (proliferative cells with ECM neighbors)
        edge_cells = []
        for i in range(len(self.points)):
            if self.cell_types[i] == self.PROLIFERATIVE:
                neighbors = self.get_neighbors(i, radius=5.0)
                if any(self.cell_types[j] in [self.ECM, self.DEGRADED_ECM] for j in neighbors):
                    edge_cells.append(i)
        
        if not edge_cells:
            return np.linalg.norm(cell_pos - self.tumor_centroid)
        
        # Find closest edge cell to our cell
        edge_positions = self.points[edge_cells]
        distances = np.linalg.norm(edge_positions - cell_pos, axis=1)
        closest_edge_idx = edge_cells[np.argmin(distances)]
        closest_edge_pos = self.points[closest_edge_idx]
        
        # Distance from centroid to closest edge cell
        return np.linalg.norm(closest_edge_pos - self.tumor_centroid)
    
    def distance_to_tumor_edge(self, cell_idx):
        """Calculate distance from cell to nearest tumor edge"""
        cell_pos = self.points[cell_idx]
        
        # Get all edge cells (proliferative cells with ECM neighbors)
        edge_cells = []
        for i in range(len(self.points)):
            if self.cell_types[i] == self.PROLIFERATIVE:
                neighbors = self.get_neighbors(i, radius=5.0)
                if any(self.cell_types[j] in [self.ECM, self.DEGRADED_ECM] for j in neighbors):
                    edge_cells.append(i)
        
        if not edge_cells:
            return float('inf')
        
        # Find closest edge cell
        edge_positions = self.points[edge_cells]
        distances = np.linalg.norm(edge_positions - cell_pos, axis=1)
        return np.min(distances)
    
    def should_become_necrotic(self, cell_idx):
        """Rule 1: Check if quiescent cell should become necrotic"""
        # Calculate  d=2 for 2D
        L_t = self.calculate_Lt(cell_idx)
        delta_n = self.a * (L_t ** 0.5)  # (2-1)/2 = 0.5
        
        # Distance from tumor edge
        distance_to_edge = self.distance_to_tumor_edge(cell_idx)
        
        return distance_to_edge > delta_n
    
    def should_become_quiescent(self, cell_idx):
        """Rule 3: Check if proliferative cell should become quiescent"""
        # Condition 1: Distance 
        L_t = self.calculate_Lt(cell_idx)
        delta_p = self.b * (L_t ** 0.5)  # (2-1)/2 = 0.5
        distance_to_edge = self.distance_to_tumor_edge(cell_idx)
        
        # Condition 2: No space for daughter cells
        neighbors = self.get_neighbors(cell_idx, radius=5.0)
        available_space = sum(1 for j in neighbors 
                            if self.cell_types[j] in [self.ECM, self.DEGRADED_ECM])
        
        return (distance_to_edge > delta_p) or (available_space == 0)
    
    def apply_ca_rules(self):
        """Apply all CA rules for cell state transitions"""
        # Make copies for simultaneous updates
        new_cell_types = self.cell_types.copy()
        new_ecm_density = self.ecm_density.copy()
        
        # Process all cells for rules 1 and 3
        for i in range(len(self.points)):
            current_type = self.cell_types[i]
            
            # Rule 1: Quiescent to Necrotic
            if current_type == self.QUIESCENT:
                if self.should_become_necrotic(i):
                    new_cell_types[i] = self.NECROTIC
            
            # Rule 3: Proliferative to Quiescent 
            elif current_type == self.PROLIFERATIVE:
                if self.should_become_quiescent(i):
                    new_cell_types[i] = self.QUIESCENT
        
        # Update the state
        self.cell_types = new_cell_types
        self.ecm_density = new_ecm_density
        return sum(1 for i in range(len(self.points)) 
                  if self.cell_types[i] != new_cell_types[i])
        
        # Sort by nutrient concentration (highest first) to maximize nutrient concentration
        ecm_neighbors.sort(key=lambda j: -self.get_nutrient_concentration(j))
        
        # Select the best target (highest nutrient concentration)
        target_idx = ecm_neighbors[0]
        
        # Make m attempts to degrade the target ECM cell
        for attempt in range(m):
            # Degrade ECM 
            delta_p = random.uniform(0, self.chi)  # 0.15
            self.ecm_density[target_idx] = max(0, self.ecm_density[target_idx] - delta_p)
            
            # If ECM fully degraded  migrate there
            if self.ecm_density[target_idx] <= 0:
                # Move invasive cell to new location
                self.cell_types[target_idx] = self.INVASIVE
                self.cell_types[cell_idx] = self.DEGRADED_ECM  # Leave degraded path
                self.ecm_density[target_idx] = 0
                self.ecm_density[cell_idx] = 0.4  # Set degraded ECM density
                return True  # Migration occurred
        
        return False  # No migration occurred

##################################### Hybrid Simulator Class with Coupling Process and Data Exchange ######################################

class HybridTumorSimulator:
    """
    CORRECTED IMPLEMENTATION: Couples CA and diffusion models using proper structure:
    - Within each N_p step: First N_1 diffusion steps, then N_a CA steps, then data exchange
    - (N_1 * N_p) * N_d = N_t (Total time-steps in one dosing period)
    - N_a * N_p = N_o (Total number of proliferative cells to divide)
    """
    
    def __init__(self, ca_model, diffusion_model, N_1=10, N_p=75, N_d=1, N_a=5):
        """
        Initialize hybrid simulator with CORRECTED time-scale formulation
        
        Parameters:
        ca_model: HybridVoronoiTumorCA instance
        diffusion_model: DiffusionReactionModel instance
        N_1: Number of diffusion sub-steps per coupling step
        N_p: Number of coupling steps per day
        N_d: Number of days per dosing period (default=1 for daily dosing)
        N_a: Number of CA sub-steps per coupling step
        """
        self.ca_model = ca_model
        self.diffusion_model = diffusion_model
        self.N_1 = N_1  # Diffusion sub-steps per coupling step
        self.N_p = N_p  # Coupling steps per day
        self.N_d = N_d  # Days per dosing period
        self.N_a = N_a  # CA sub-steps per coupling step
        
        # Calculate derived parameters
        self.N_t = (N_1 * N_p) * N_d  # Total diffusion time-steps in one dosing period
        self.N_o = N_a * N_p  # Total CA steps per day
        
        # Time step calculations
        self.dt_diffusion = (86400 * N_d) / self.N_t  # seconds per diffusion step
        self.dt_coupling = 86400 / N_p  # seconds per coupling step
        
        # Add daily division rate parameter to CA model if not present
        if not hasattr(self.ca_model, 'daily_division_rate'):
            self.ca_model.daily_division_rate = 0.1  # 10% of proliferative cells divide per day
        
        print(f"Hybrid Simulator Configuration:")
        print(f"  N_1 (diffusion substeps per coupling): {N_1}")
        print(f"  N_p (coupling steps per day): {N_p}")
        print(f"  N_a (CA substeps per coupling): {N_a}")
        print(f"  N_0 (total proliferative cells to divide/day): {self.N_o}")
        print(f"  Diffusion time step: {self.dt_diffusion:.1f} seconds ({self.dt_diffusion/60:.1f} minutes)")
        print(f"  Coupling time step: {self.dt_coupling:.1f} seconds ({self.dt_coupling/60:.1f} minutes)")
    
    def simulate_day(self):
        """
        Simulate one complete dosing period (N_d days)
        Following the main loop structure from pseudocode
        """
        total_divisions = 0

        # CORRECTED: Calculate N_s ONCE at the beginning of the day, not per coupling step
        proliferative_count = np.sum(self.ca_model.cell_types == self.ca_model.PROLIFERATIVE)
        # N_s should be total proliferative cells to select across ALL coupling steps in the day
        N_s = int(proliferative_count * self.ca_model.daily_division_rate)
        
        # CORRECTED: Select ALL N_s proliferative cells at the beginning
        if N_s > 0 and proliferative_count > 0:
            selected_cells_for_day = self.ca_model.select_proliferative_cells_to_divide(N_s)
            # Distribute selected cells across coupling steps
            cells_per_coupling = max(1, len(selected_cells_for_day) // self.N_p)
        else:
            selected_cells_for_day = []
            cells_per_coupling = 0
        
        # Main loop: Do i=1, N_p (coupling steps per day)
        for coupling_step in range(self.N_p):
            
            # === DIFFUSION-REACTION SUB-PROCESS ===
            # Do i=1, N_1 (diffusion substeps)
            
            for diffusion_substep in range(self.N_1):
                self.diffusion_model.update_ecm_density(self.ca_model)
                self.diffusion_model.update_cell_density(self.ca_model)
                # Perform one diffusion-reaction step
                self.diffusion_model.diffusion_step(self.dt_diffusion)
            
            # === CA MODEL SUB-PROCESS ===
            # CORRECTED: Get cells for this coupling step from pre-selected cells
            start_idx = coupling_step * cells_per_coupling
            end_idx = min((coupling_step + 1) * cells_per_coupling, len(selected_cells_for_day))
            selected_cells_this_coupling = selected_cells_for_day[start_idx:end_idx] if selected_cells_for_day else []
            
            # Do i=1, N_a (CA substeps per coupling)
            
            for ca_substep in range(self.N_a):
                    # Update drug concentrations from diffusion model
                    self.ca_model.update_drug_concentrations(self.diffusion_model)
                    
                    # CORRECTED: Apply cell selection and division within CA substeps
                    if selected_cells_this_coupling and ca_substep < len(selected_cells_this_coupling):
                       # Select N_s proliferative cells to divide based on P_div probabilities
                       # This should happen within the CA substeps
                       cell_to_divide = [selected_cells_this_coupling[ca_substep]]
                       divisions = self.ca_model.apply_proliferative_division_with_selection(cell_to_divide)
                       total_divisions += divisions
                    
                    # Update all types of cells (CA rules)
                    self.ca_model.apply_ca_rules()
                    
                    # Update tumor centroid
                    self.ca_model.update_tumor_centroid()
            
            # === BIDIRECTIONAL DATA EXCHANGE ===
            # Calculate cell density and map to finite difference grid
            self.diffusion_model.update_cell_density(self.ca_model)
            
            # Update ECM density
            self.diffusion_model.update_ecm_density(self.ca_model)


        # Update day counter and history
        self.ca_model.day += 1
        radius = self.ca_model.calculate_proliferative_radius()
        invasive_radius=self.ca_model.calculate_invasive_radius()
        self.ca_model.radius_history.append(radius)
        self.ca_model.invasive_radius_history.append(invasive_radius)
        self.ca_model.time_history.append(self.ca_model.day)
        
        return total_divisions
        
    def simulate(self, days=120):
        """Run hybrid simulation for specified days"""
        print(f"\n=== Starting CORRECTED Hybrid Simulation ===")
        print(f"Configuration:")
        print(f"  - Drug effectiveness P_gamma = {self.ca_model.P_gamma}")
        print(f"  - Simulation duration = {days} days")
        print(f"  - Total diffusion steps per day: {self.N_1 * self.N_p}")
        print(f"  - Total CA steps per day: {self.N_o}")
        print(f"  - Initial proliferative radius: {self.ca_model.radius_history[0]:.2f} mm")
        print(f"  - Initial invasive cells: {np.sum(self.ca_model.cell_types == self.ca_model.INVASIVE)}")
        
        for day in range(days):
            divisions = self.simulate_day()
            
            # Progress reporting
            if day % 20 == 0 or day == days - 1:
                current_invasive = np.sum(self.ca_model.cell_types == self.ca_model.INVASIVE)
                current_prolif = np.sum(self.ca_model.cell_types == self.ca_model.PROLIFERATIVE)
                avg_drug_conc = np.mean(self.ca_model.drug_concentration)
                max_drug_conc = np.max(self.ca_model.drug_concentration)
                
                print(f"Day {day:3d}: Radius = {self.ca_model.radius_history[-1]:.2f} mm, "
                      f"Divisions = {divisions:3d}, Invasive = {current_invasive:3d}, "
                      f"Prolif = {current_prolif:3d}, Drug(avg/max) = {avg_drug_conc:.4f}/{max_drug_conc:.4f}")
        
        final_invasive = np.sum(self.ca_model.cell_types == self.ca_model.INVASIVE)
        final_prolif = np.sum(self.ca_model.cell_types == self.ca_model.PROLIFERATIVE)
        
        print(f"\n=== Simulation Complete ===")
        print(f"Final proliferative radius: {self.ca_model.radius_history[-1]:.2f} mm")
        print(f"Final Invasive radius: {self.ca_model.invasive_radius_history[-1]:.2f} mm")
        print(f"Final number of invasive cells: {final_invasive}")
        print(f"Final number of proliferative cells: {final_prolif}")
        print(f"Total tumor cells: {final_invasive + final_prolif}")
    
    def plot_results(self):
        """Plot comprehensive simulation results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Tumor growth in cm vs days (matching paper format)
        radius_cm = [r / 10.0 for r in self.ca_model.radius_history]  # Convert mm to cm
        ax1.plot(self.ca_model.time_history, radius_cm, 'b-o', linewidth=2, markersize=4)
        ax1.set_xlabel('Time (days)', fontsize=12)
        ax1.set_ylabel('Proliferative Tumor Radius (cm)', fontsize=12)
        ax1.set_title(f'Tumor Growth with Drug Concentration (P_gamma = {self.ca_model.P_gamma})', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, max(self.ca_model.time_history))
        
        # Plot 2: Drug concentration field
        im2 = ax2.imshow(self.diffusion_model.phi, extent=[-50, 50, -50, 50], 
                        origin='lower', cmap='viridis', vmin=0)
        ax2.set_title('Drug Concentration Field phi(x,y)', fontsize=14)
        ax2.set_xlabel('Distance (mm)', fontsize=12)
        ax2.set_ylabel('Distance (mm)', fontsize=12)
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label('Drug Concentration', fontsize=11)
        
        # Add tumor boundary circle
        circle2 = plt.Circle((0, 0), self.ca_model.calculate_proliferative_radius(), 
                           fill=False, color='white', linewidth=2, linestyle='--')
        ax2.add_patch(circle2)
        
        # Plot 3: Cell density field
        im3 = ax3.imshow(self.diffusion_model.n_density, extent=[-50, 50, -50, 50], 
                        origin='lower', cmap='plasma', vmin=0)
        ax3.set_title('Cell Density Field n(x,y)', fontsize=14)
        ax3.set_xlabel('Distance (mm)', fontsize=12)
        ax3.set_ylabel('Distance (mm)', fontsize=12)
        cbar3 = plt.colorbar(im3, ax=ax3)
        cbar3.set_label('Cell Density (cells/mm_2)', fontsize=11)
        
        # Plot 4: Final tumor state
        cell_configs = [
            (self.ca_model.PROLIFERATIVE, 'red', 'Proliferative', 30),
            (self.ca_model.INVASIVE, 'green', 'Invasive', 25),
            (self.ca_model.QUIESCENT, 'yellow', 'Quiescent', 20),
            (self.ca_model.NECROTIC, 'black', 'Necrotic', 15),
        ]
        
        for cell_type, color, name, size in cell_configs:
            indices = np.where(self.ca_model.cell_types == cell_type)[0]
            if len(indices) > 0:
                points = self.ca_model.points[indices]
                ax4.scatter(points[:, 0], points[:, 1], c=color, s=size, 
                          alpha=0.7, label=f'{name} ({len(indices)})', edgecolors='black', linewidth=0.3)
        
        # Add domain boundary
        circle4 = plt.Circle((0, 0), self.ca_model.radius, fill=False, color='black', linewidth=2)
        ax4.add_patch(circle4)
        ax4.set_xlim(-55, 55)
        ax4.set_ylim(-55, 55)
        ax4.set_aspect('equal')
        ax4.set_title(f'Final Tumor State (Day {self.ca_model.day})', fontsize=14)
        ax4.set_xlabel('Distance (mm)', fontsize=12)
        ax4.set_ylabel('Distance (mm)', fontsize=12)
        ax4.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plot_results_(P_gamma = {self.ca_model.P_gamma}).png", dpi=300)
        return fig
    
    def plot_detailed_analysis(self):
        """Plot detailed analysis of simulation results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Cell population dynamics
        days = self.ca_model.time_history
        invasive_count = []
        prolif_count = []
        quiescent_count = []
        necrotic_count = []
        
        # We only have final counts, so this is simplified
        # In a full implementation, you'd track these over time
        final_invasive = np.sum(self.ca_model.cell_types == self.ca_model.INVASIVE)
        final_prolif = np.sum(self.ca_model.cell_types == self.ca_model.PROLIFERATIVE)
        final_quiescent = np.sum(self.ca_model.cell_types == self.ca_model.QUIESCENT)
        final_necrotic = np.sum(self.ca_model.cell_types == self.ca_model.NECROTIC)
        
        ax1.plot(days, [final_invasive] * len(days), 'r-', label=f'Invasive ({final_invasive})', linewidth=2)
        ax1.plot(days, [final_prolif] * len(days), 'g-', label=f'Proliferative ({final_prolif})', linewidth=2)
        ax1.plot(days, [final_quiescent] * len(days), 'orange', label=f'Quiescent ({final_quiescent})', linewidth=2)
        ax1.plot(days, [final_necrotic] * len(days), 'black', label=f'Necrotic ({final_necrotic})', linewidth=2)
        ax1.set_xlabel('Time (days)', fontsize=12)
        ax1.set_ylabel('Cell Count', fontsize=12)
        ax1.set_title('Cell Population Dynamics', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Drug concentration profile along x-axis
        center_idx = self.diffusion_model.ny // 2
        x_coords = np.linspace(-50, 50, self.diffusion_model.nx)
        drug_profile = self.diffusion_model.phi[center_idx, :]
        
        ax2.plot(x_coords, drug_profile, 'b-', linewidth=2)
        ax2.set_xlabel('Distance from center (mm)', fontsize=12)
        ax2.set_ylabel('Drug Concentration', fontsize=12)
        ax2.set_title('Drug Concentration Profile (y=0)', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Add vertical lines for tumor boundary
        tumor_radius = self.ca_model.calculate_proliferative_radius()
        ax2.axvline(x=tumor_radius, color='red', linestyle='--', alpha=0.7, label='Tumor boundary')
        ax2.axvline(x=-tumor_radius, color='red', linestyle='--', alpha=0.7)
        ax2.legend()
        
        # Plot 3: ECM density field
        im3 = ax3.imshow(self.diffusion_model.rho_ECM, extent=[-50, 50, -50, 50], 
                        origin='lower', cmap='Blues', vmin=0, vmax=1)
        ax3.set_title('ECM Density Field rho_ECM(x,y)', fontsize=14)
        ax3.set_xlabel('Distance (mm)', fontsize=12)
        ax3.set_ylabel('Distance (mm)', fontsize=12)
        cbar3 = plt.colorbar(im3, ax=ax3)
        cbar3.set_label('ECM Density', fontsize=11)
        
        # Plot 4: Growth rate analysis
        if len(self.ca_model.radius_history) > 1:
            growth_rates = []
            for i in range(1, len(self.ca_model.radius_history)):
                dr = self.ca_model.radius_history[i] - self.ca_model.radius_history[i-1]
                dt = self.ca_model.time_history[i] - self.ca_model.time_history[i-1]
                growth_rates.append(dr/dt if dt > 0 else 0)
            
            ax4.plot(self.ca_model.time_history[1:], growth_rates, 'g-o', linewidth=2, markersize=3)
            ax4.set_xlabel('Time (days)', fontsize=12)
            ax4.set_ylabel('Growth Rate (mm/day)', fontsize=12)
            ax4.set_title('Tumor Growth Rate', fontsize=14)
            ax4.grid(True, alpha=0.3)
            ax4.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/plot_detailed_results_(P_gamma = {self.ca_model.P_gamma}).png", dpi=300)
        return fig


############################### Function for single P_gamma value ##################################################

def run_hybrid_tumor_simulation(P_gamma=0.6, days=120, N_p=75):
    """
    Run hybrid tumor simulation with drug treatment
    
    Parameters:
    P_gamma: Drug effectiveness parameter (0 to 1)
    days: Simulation duration in days
    N_p: Number of sub-steps per day (50-100 recommended)
    """
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    print("=== Initializing Hybrid Tumor Model ===")
    print(f"Configuration:")
    print(f"  - Drug effectiveness P_gamma = {P_gamma}")
    print(f"  - Simulation duration = {days} days")
    print(f"  - Quasi-parallel coupling steps = {N_p} per day")
    
    # Create CA model
    ca_model = HybridVoronoiTumorCA(radius=50, n_seeds=1200, P_gamma=P_gamma)
    
    # Create diffusion model with proper parameters
    diffusion_model = DiffusionReactionModel(
        domain_size=100,        # mm (matches paper)
        grid_spacing=2.0,       # mm (good resolution)
        D0=1e-5,               #  (drug diffusion coefficient)
        K_met=2.0e-4/60,       # (drug decomposition rate)
        lambda_14=2.5e-4,      # drug consumption rate coefficient
        phi_0=0.1,             # half-saturation concentration
        tau_cycle=24*3600,     # 24 hours dosing cycle (seconds)
        tau_decay=60*600      # 1 day decay time (seconds)
    )
    
    # Create hybrid simulator
    simulator = HybridTumorSimulator(ca_model, diffusion_model, N_1=10, N_p=75, N_d=1, N_a=5)
    
    # Run simulation
    simulator.simulate(days=days)
    
    # Plot results
    print("\n=== Generating Plots ===")
    fig1 = simulator.plot_results()
    #fig1.suptitle(f'Hybrid Tumor-Drug Model Results (P_gamma = {P_gamma}, {days} days)', fontsize=16)
    
    fig2 = simulator.plot_detailed_analysis()
    #fig2.suptitle(f'Detailed Analysis (P_gamma = {P_gamma})', fontsize=16)
    
    #plt.show()
    
    return simulator
    
#################################################################################

def compare_dosing_regimens(days=120, N_p=50, P_gamma=0.05):
    """
    Compare different drug dosing regimens and parameters
    
    Parameters:
    days: Simulation duration in days
    N_p: Number of sub-steps per day
    P_gamma: Drug effectiveness parameter
    """
    
    # Define the 4 cases
    cases = {
        'Case 1': {
            'tau_cycle': 1 * 24 * 3600,      # 1 day in seconds
            'tau_decay': 600 * 60,           # 600 minutes in seconds
            'lambda_14': 2.5e-4,             # Default value
            'label': 'tau_cycle=1d, tau_decay=600min'
        },
        'Case 2': {
            'tau_cycle': 2 * 24 * 3600,      # 2 days in seconds
            'tau_decay': 1400 * 60,          # 1400 minutes in seconds
            'lambda_14': 2.5e-4,             # Default value
            'label': 'tau_cycle=2d, tau_decay=1d'
        },
        'Case 3': {
            'tau_cycle': 1 * 24 * 3600,      # 1 day in seconds
            'tau_decay': 1800 * 60,          # 1800 minutes in seconds
            'lambda_14': 2.5e-4,             # Default value
            'label': 'tau_cycle=1d,tau_decay=1800min'
        },
        'Case 4': {
            'tau_cycle': 1 * 24 * 3600,      # 1 day in seconds
            'tau_decay': 1800 * 60,          # 1800 minutes in seconds
            'lambda_14': 2.5e-5,             # Reduced by factor of 10
            'label': 'tau_cycle=1d, tau_decay=1800min, lamda_14=2.5e-5'
        }
    }
    
    results = {}
    simulators = {}
    
    print("=== Comparing Drug Dosing Regimens ===")
    print(f"Configuration: P_gamma = {P_gamma}, Duration = {days} days, N_p = {N_p}")
    print()
    
    # Run simulations for each case
    for case_name, params in cases.items():
        print(f"Running {case_name}: {params['label']}")
        
        # Set random seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        
        # Create CA model
        ca_model = HybridVoronoiTumorCA(radius=50, n_seeds=1200, P_gamma=P_gamma)
        
        # Create diffusion model with case-specific parameters
        diffusion_model = DiffusionReactionModel(
            domain_size=100,
            grid_spacing=2.0,
            D0=1e-5,
            K_met=2.0e-4/60,
            lambda_14=params['lambda_14'],
            phi_0=0.1,
            tau_cycle=params['tau_cycle'],
            tau_decay=params['tau_decay']
        )
        
        # Create and run simulator
        simulator = HybridTumorSimulator(ca_model, diffusion_model, N_1=10, N_p=50, N_d=1, N_a=5)
        simulator.simulate(days=days)
        
        # Store results
        results[case_name] = {
            'radius_history': simulator.ca_model.radius_history.copy(),
            'time_history': simulator.ca_model.time_history.copy(),
            'final_invasive': np.sum(simulator.ca_model.cell_types == simulator.ca_model.INVASIVE),
            'final_prolif': np.sum(simulator.ca_model.cell_types == simulator.ca_model.PROLIFERATIVE),
            'final_quiescent': np.sum(simulator.ca_model.cell_types == simulator.ca_model.QUIESCENT),
            'final_necrotic': np.sum(simulator.ca_model.cell_types == simulator.ca_model.NECROTIC),
            'final_degraded_ecm': np.sum(simulator.ca_model.cell_types == simulator.ca_model.DEGRADED_ECM),
            'label': params['label']
        }
        simulators[case_name] = simulator
        
        print(f"  Final radius: {simulator.ca_model.radius_history[-1]:.2f} mm")
        print(f"  Final invasive cells: {results[case_name]['final_invasive']}")
        print(f"  Final proliferative cells: {results[case_name]['final_prolif']}")
        print(f"  Final degraded ECM: {results[case_name]['final_degraded_ecm']}")
        print()
    
    # Create simplified comparison plot
    fig = plt.figure(figsize=(16, 10))
    
    # Define colors for cases
    case_colors = ['blue', 'red', 'green', 'purple']
    case_names = list(cases.keys())
    
    # Plot 1: Growth curves comparison
    ax1 = plt.subplot(2, 3, (1, 2))  # Span two columns
    for i, case_name in enumerate(case_names):
        radius_cm = [r / 10.0 for r in results[case_name]['radius_history']]
        ax1.plot(results[case_name]['time_history'], radius_cm, 
                color=case_colors[i], linewidth=3, label=case_name, 
                marker='o', markersize=4, markevery=10)
    
    ax1.set_xlabel('Time (days)', fontsize=14)
    ax1.set_ylabel('Proliferative Tumor Radius (cm)', fontsize=14)
    ax1.set_title('Tumor Growth Comparison', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plots 2-5: Final tumor states for each case
    for i, case_name in enumerate(case_names):
        ax = plt.subplot(2, 3, 3 + i)
        simulator = simulators[case_name]
        
        # Plot degraded ECM (green, small)
        degraded_ecm_indices = np.where(simulator.ca_model.cell_types == simulator.ca_model.DEGRADED_ECM)[0]
        if len(degraded_ecm_indices) > 0:
            points = simulator.ca_model.points[degraded_ecm_indices]
            ax.scatter(points[:, 0], points[:, 1], c='green', s=15, 
                      alpha=0.7, label=f'Degraded ECM ({len(degraded_ecm_indices)})', 
                      edgecolors='black', linewidth=0.3)
        
        # Plot proliferative cells (red, large)
        prolif_indices = np.where(simulator.ca_model.cell_types == simulator.ca_model.PROLIFERATIVE)[0]
        if len(prolif_indices) > 0:
            points = simulator.ca_model.points[prolif_indices]
            ax.scatter(points[:, 0], points[:, 1], c='red', s=40, 
                      alpha=0.9, label=f'Proliferative ({len(prolif_indices)})', 
                      edgecolors='black', linewidth=0.4)
        
        # Plot invasive cells (yellow, medium)
        invasive_indices = np.where(simulator.ca_model.cell_types == simulator.ca_model.INVASIVE)[0]
        if len(invasive_indices) > 0:
            points = simulator.ca_model.points[invasive_indices]
            ax.scatter(points[:, 0], points[:, 1], c='yellow', s=25, 
                      alpha=0.9, label=f'Invasive ({len(invasive_indices)})', 
                      edgecolors='black', linewidth=0.4)
        
        # Add domain boundary
        circle = plt.Circle((0, 0), simulator.ca_model.radius, 
                           fill=False, color='black', linewidth=2)
        ax.add_patch(circle)
        
        # Add proliferative boundary
        prolif_radius = simulator.ca_model.calculate_proliferative_radius()
        prolif_circle = plt.Circle((0, 0), prolif_radius, 
                                  fill=False, color='red', linewidth=1.5, linestyle='--', alpha=0.8)
        ax.add_patch(prolif_circle)
        
        ax.set_xlim(-55, 55)
        ax.set_ylim(-55, 55)
        ax.set_aspect('equal')
        ax.set_title(f'{case_name} (Day {days})\nRadius = {prolif_radius:.2f} mm', fontsize=12)
        ax.set_xlabel('Distance (mm)', fontsize=10)
        ax.set_ylabel('Distance (mm)', fontsize=10)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    
    # Overall title and layout
    
    #fig.suptitle(f'Drug Dosing Regimen Comparison (P_? = {P_gamma}, {days} days)', 
                #fontsize=18, y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    plt.savefig(f"{output_dir}/plot_detailed_results_different_tau.png", dpi=300)
    
    # Print simplified summary
    print("=== Summary ===")
    print("Case\t\tFinal Radius (cm)\tInvasive\tProliferative\tDegraded ECM")
    print("-" * 70)
    for case_name in case_names:
        final_radius_cm = results[case_name]['radius_history'][-1] / 10.0
        invasive = results[case_name]['final_invasive']
        prolif = results[case_name]['final_prolif']
        degraded = results[case_name]['final_degraded_ecm']
        
        print(f"{case_name}\t\t{final_radius_cm:.2f}\t\t\t{invasive}\t\t{prolif}\t\t{degraded}")
    
    #plt.show()
    
    return results, simulators

    
if __name__ == "__main__":
    
    start_time = time.time()
    
    results, simulators = compare_dosing_regimens(days=120, N_p=50, P_gamma=0.05)
     
    elapsed_seconds = time.time() - start_time
    elapsed_minutes = elapsed_seconds / 60
    elapsed_hours = elapsed_minutes / 60
    
    print("The simulation is completed")
    print(f"- {elapsed_hours:.2f} hours")
    

