import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


def calc_boundary_area(x, y, show=True):
    """
    Use convex hull to calculate area of wind farm boundary and optionally plot
    """
    points = np.column_stack((x, y))
    hull = ConvexHull(points)

    if show:
        plt.scatter(x, y, color="blue", label="Turbines")
        # Plot boundary
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], "r-")

        # Close the hull
        plt.plot(
            [points[hull.vertices[0], 0], points[hull.vertices[-1], 0]],
            [points[hull.vertices[0], 1], points[hull.vertices[-1], 1]],
            "r-",
        )
        plt.xlabel("x-coordinate")
        plt.ylabel("y-coordinate")
        plt.title(f"Wind Farm Boundary (Area = {hull.volume:.2f})")
        plt.legend()
        plt.axis("equal")
        plt.show()

    return hull.volume


def blockage_metrics(
    xy, wind_dir_deg_from_north, D, grid_res=151, L_inf_factor=20.0, plot=False
):
    """
    Compute per-turbine Blockage Ratio (BR) and Blockage Distance (BD)
    for a given 2D wind farm layout and wind direction.

    Parameters
    ----------
    xy : (N, 2) array
        Turbine coordinates in meters (X east, Y north).
    wind_dir_deg_from_north : float
        Meteorological wind direction in degrees (0 = from north, 90 = from east).
    D : float
        Rotor diameter (meters). All turbines assumed identical (as in the paper).
    grid_res : int
        Number of samples per axis for disk discretization (odd number recommended).
        Effective number of points on the rotor ≈ π*(grid_res/2)^2 / 4.
    L_inf_factor : float
        L∞ as a multiple of D (paper uses 20D).

    Returns
    -------
    BR : (N,) array
        Blockage ratio for each turbine (0..1).
    BD : (N,) array
        Blockage distance for each turbine (meters).
    BR_farm : float
        Mean BR across turbines.
    BD_farm : float
        Mean BD across turbines.
    """

    xy = np.asarray(xy, dtype=float)
    N = xy.shape[0]
    R = 0.5 * D

    # Default blockage distance for turbines which aren't blocked
    L_inf = L_inf_factor * D

    # Change direction into radians and define downwind and crosswind directions
    theta = np.deg2rad(wind_dir_deg_from_north)
    a_hat = np.array([-np.sin(theta), -np.cos(theta)])
    n_hat = np.array([np.cos(theta), -np.sin(theta)])

    # map turbine coordinates to downwind and crosswind vectors
    x_down = xy @ a_hat
    x_cross = xy @ n_hat

    # Creating a mesh grid for a rotor disk in the (u,z) plane:
    # u=horizontal crosswind offset, z=vertical offset (offsets from the centre of disk)
    # Keeping only points inside the rotor disk
    lin = np.linspace(-R, R, grid_res)
    uu, zz = np.meshgrid(lin, lin, indexing="xy")
    mask = (uu * uu + zz * zz) <= (R * R)
    u_pts = uu[mask]
    z_pts = zz[mask]
    M = u_pts.size

    # Initializing arrays to store blockage ratio and blockage distance for each turbine
    BR = np.zeros(N)
    BD = np.zeros(N)

    # For all turbines
    for i in range(N):
        # Initialize the grid for our turbine disk
        u_i = u_pts
        # z_i = z_pts

        # calculate the downstream distance between this turbine and all others
        delta_down = x_down[i] - x_down  # (N,)
        upstream_mask = delta_down > 0.0
        # if a turbine is not blocked
        if not np.any(upstream_mask):
            BR[i] = 0.0
            BD[i] = 1.0
            continue

        # For upstream turbines, calculate the crosswind and downwind distance between rotor centres
        cross_dist = x_cross[i] - x_cross[upstream_mask]
        down_dist = delta_down[upstream_mask]

        ### Initialize some variables:
        # boolean array representing which grid cells of the rotor grid are blocked by any upstream turbine
        blocked = np.zeros(M, dtype=bool)

        # array with default value of L_inf containing the distance to the nearest upstream turbine
        L_point = np.full(M, L_inf, dtype=float)

        ### Looping over each upstream turbine and calculating blockage on all rotor grid cells
        for j in range(cross_dist.size):
            dx = u_i - cross_dist[j]  # horizontal distance to upstream turbine centre
            dz = z_pts  # vertical offset
            d = np.sqrt(dx**2 + dz**2)  # actual straight line /euclidean distance
            hits = (
                d <= R
            )  # boolean array of rotor grid cells that are blocked by this upstream turbine

            if np.any(hits):
                blocked[hits] = True
                # In the scenario that a grid point is blocked by multiple turbines, we select the nearest one
                # This is done by selecting the minimum distance from either the turbine in the previous iteration, or the
                # Distance to the turbine in the current iteration
                L_point[hits] = np.minimum(L_point[hits], down_dist[j])

        # Calculating the ratios over the disk
        BR[i] = blocked.mean()

        # "blocked" is a boolean representing which grid cells are blocked and L_point represents the distance from each blocked point to the nearest blocking turbine
        # If there is no grid cells blocked, we default to L_inf and the total metric is a weighted sum dependent on how many grid cells are blocked
        BD[i] = (blocked * L_point + (~blocked) * L_inf).mean() / L_inf

    # Quantities on a farm level
    BR_farm = BR.mean()
    BD_farm = BD.mean()

    if plot:

        def plot_metric(metric, title, cmap="viridis"):
            fig, ax = plt.subplots(figsize=(8, 8))
            sc = ax.scatter(
                xy[:, 0], xy[:, 1], c=metric, s=100, cmap=cmap, edgecolor="k"
            )
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label("Metric value", fontsize=12)

            # Wind arrow: points FROM where wind is coming
            arrow_length = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])) * 0.1
            ax.arrow(
                np.mean(xy[:, 0]),
                np.mean(xy[:, 1]),
                -arrow_length * np.sin(theta),
                -arrow_length * np.cos(theta),
                head_width=arrow_length * 0.2,
                head_length=arrow_length * 0.2,
                fc="r",
                ec="r",
                linewidth=2,
            )
            ax.text(
                np.mean(xy[:, 0]),
                np.mean(xy[:, 1]) + arrow_length * 0.15,
                "Wind",
                color="r",
                ha="center",
            )

            pad = max(250, 0.1 * max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1])))
            x_min, x_max = xy[:, 0].min() - pad, xy[:, 0].max() + pad
            y_min, y_max = xy[:, 1].min() - pad, xy[:, 1].max() + pad
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_aspect("equal")
            ax.set_title(title)
            plt.show()

        plot_metric(
            BR,
            f"Blockage Ratio per Turbine: Farm Average = {BR_farm:.2f}",
            cmap="viridis",
        )
        plot_metric(
            BD,
            f"Blockage Distance per Turbine [m]: Farm Average = {BD_farm:.2f}",
            cmap="plasma",
        )

    return BR, BD, BR_farm, BD_farm


def farm_length_width(x, y, wind_dir_deg_from_north, D, plot=False):
    """
    Compute farm length and width in wind and crosswind directions, normalized by diameter.

    """
    xy = np.column_stack((x, y))
    # N = len(xy)
    theta = np.deg2rad(wind_dir_deg_from_north)
    # Wind direction unit vector
    wind_vec = np.array([-np.sin(theta), -np.cos(theta)])
    cross_vec = np.array([np.cos(theta), -np.sin(theta)])

    # Project all points onto wind and crosswind axes
    proj_wind = xy @ wind_vec
    proj_cross = xy @ cross_vec

    length = np.round((proj_wind.max() - proj_wind.min()) / D)
    width = np.round((proj_cross.max() - proj_cross.min()) / D)

    # Optional plotting
    if plot:

        def plot_turbine_layout(x, y, wind_dir_deg_from_north, title="Turbine Layout"):
            plt.figure(figsize=(6, 6))
            plt.scatter(x, y, s=100, c="b", label="Turbines")
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.axis("equal")
            plt.grid(True)
            plt.title(title)

            # Wind direction arrow
            center_x = np.mean(x)
            center_y = np.mean(y)
            arrow_length = max(np.ptp(x), np.ptp(y)) * 0.2
            theta = np.deg2rad(wind_dir_deg_from_north)
            dx = -arrow_length * np.sin(theta)
            dy = -arrow_length * np.cos(theta)
            plt.arrow(
                center_x,
                center_y,
                dx,
                dy,
                head_width=arrow_length * 0.15,
                head_length=arrow_length * 0.15,
                fc="r",
                ec="r",
                linewidth=2,
            )
            plt.text(
                center_x + dx * 1.1,
                center_y + dy * 1.1,
                "Wind",
                color="r",
                ha="center",
                va="center",
            )
            plt.legend()
            plt.show()

        plot_turbine_layout(
            x,
            y,
            wind_dir_deg_from_north=wind_dir_deg_from_north,
            title=f"D= {D}\n Length={length}D\nWidth={width}D",
        )

    return length, width
