import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # MUST be before importing pyplot
import matplotlib.pyplot as plt
from multiprocessing import Pool
from matplotlib.colors import LogNorm
from pathlib import Path
from multiprocessing import Pool
from FuncDatabase import *


_fields = [
    'alpha_rho',     # 03: Conservative Density 
    'rho',           # 04: Primitive Density 
    'rho_u',         # 05: X-Momentum (Conservative) 
    'rho_v',         # 06: Y-Momentum (Conservative)
    'u',             # 07: X-Velocity (Primitive)
    'v',             # 08: Y-Velocity (Primitive)
    'E',             # 09: Total Energy
    'p',             # 10: Pressure (derived from EoS)
    'alpha',         # 11: Volume Fraction 
    'c',             # 12: Sound Speed 
    'ib_markers'     # 13: Immersed Boundary Markers (0 fluid, >1 solid)
]


def inspect_contents(db):
    def walk(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f'DATASET {obj.name}:: ({obj.shape}) \tmax: {obj[()].max()}, min: {obj[()].min()}')
        elif isinstance(obj, h5py.Group):
            print(f'GROUP {name}')
        elif isinstance(obj, h5py.Datatype):
            print(f'NAMEDTYPE {name}, {obj.dtype}')
    db.visititems(walk)
    return


def extract_line(x_cc,y_cc,data_field,x_val,y_range):
    '''Extracts a line of data for a constant x value over a given y range.'''

    # Finds the nearest x index to the specified x value
    x_idx = (np.abs(x_cc - x_val)).argmin()

    # Id index range for y values
    y_indices = np.where((y_cc >= y_range[0]) & (y_cc <= y_range[1]))[0]

    extracted_data = data_field[x_idx, y_indices]
    extracted_y = y_cc[y_indices]
    
    return extracted_y, extracted_data


def mfcpp(snapshot: int, Main_plot: bool = True, ExtractLine: bool = True,
          BL_Analysis: bool = False, ZoomState: str = "Base", Verbose: bool = False,
          Use_STL_Mask: bool = False, stl_profile: dict = None,
          STL_Mask_Color: str = 'dimgray',
          slice_plane: str = None, slice_coord: float = None, slice_index: int = None):
        
    run_root = Path(__file__).parent.parent / "Runs-Folder" / "Hyp-Lam-Flatplate"
    db_dir   = run_root / "silo_hdf5" / "p0"
    db_path  = db_dir / f"{snapshot}.silo"

    if not db_path.exists():
        print(f"  [SKIP] Snapshot {snapshot}: file not found ({db_path})")
        return

    print(f"Started reading snapshot {snapshot}...")

    #inspect_contents(h5py.File(db_path, "r"))
    #exit()

    # Read data (auto-detects 2D/3D; for 3D, slice_plane must be provided)
    xx, yy, fields_data = Silo_Read(db_path,
                                     slice_plane=slice_plane,
                                     slice_coord=slice_coord,
                                     slice_index=slice_index)
    # Extract 1-D coordinate arrays from the meshgrid (works for any slice plane)
    x_cc = xx[:, 0]
    y_cc = yy[0, :]

    print(f'SNAPSHOT {snapshot} ' + 10*'==')
    if Verbose:
        for f in fields_data:
            print(f'FIELD {f} ... min: {fields_data[f].min(): .4e}'
                f' ... max: {fields_data[f].max(): .4e}')

    print(f'SNAPSHOT {snapshot} ' + 10*'==' + '\n')


    if ExtractLine:
        # --- START LINE EXTRACTION ---
        # Example: Extract Pressure at X = 0.5 for Y between 0.1 and 0.9
        target_x = [0.22, .25, .35, .38]  
        target_y_range = (0.02539, 0.05)
        target_value = 'rho'
        line_vals = np.zeros((len(target_x), len(np.where((y_cc >= target_y_range[0]) & (y_cc <= target_y_range[1]))[0])))
        U_Uinf = np.zeros((len(target_x), len(np.where((y_cc >= target_y_range[0]) & (y_cc <= target_y_range[1]))[0])))
        

        for num, tx in enumerate(target_x):
            line_y, line_vals[num] = extract_line(x_cc, y_cc, fields_data[target_value], tx, target_y_range)
            if BL_Analysis:
                y_wall_norm, U_Uinf[num], tau_wall, C_f = BL_extract_line(x_cc, y_cc, fields_data['u'], fields_data['Temperature'], tx, target_y_range)

        fig, ax = plt.subplots(1, len(target_x), figsize=(10*len(target_x), 10))
        for numtx, tx in enumerate(target_x):
            ax[numtx].plot(line_vals[numtx], line_y, label=f'{target_value} at x={tx}',linewidth=3)
            ax[numtx].set_ylabel('Y Dist (m)', fontsize=20)
            ax[numtx].set_xlabel(target_value, fontsize=20)
            ax[numtx].set_xlim(0, 0.08)
            ax[numtx].set_ylim(0.0254, 0.05)
            ax[numtx].set_title(f'x = {tx} m', fontsize=20)  # ← Individual subplot title
        plt.suptitle(f'Line Extraction Snapshot {snapshot}', fontsize=24)  # ← Overall figure title
        plt.tight_layout()
        plt.savefig(f'Line_Images/line_extract_{snapshot}_{target_value}.png')
        plt.close()

        if BL_Analysis:
            fig, ax = plt.subplots(1, len(target_x), figsize=(10*len(target_x), 10))
            for numtx, tx in enumerate(target_x):
                ax[numtx].plot(U_Uinf[numtx], y_wall_norm, label=f'BL Profile at x={tx}',linewidth=3)
                ax[numtx].set_ylabel('Y Dist (m)', fontsize=20)
                ax[numtx].set_xlabel('U/U_inf', fontsize=20)
                ax[numtx].set_xlim(0, 1.1)
                ax[numtx].set_ylim(0.0254, 0.05)
                ax[numtx].set_title(f'x = {tx} m', fontsize=20)  # ← Individual subplot title
            plt.suptitle(f'Boundary Layer Profile Snapshot {snapshot}', fontsize=24)
            plt.tight_layout()
            plt.savefig(f'BL_Images/BL_profile_{snapshot}.png')
            plt.close()

        # --- END LINE EXTRACTION ---



    print(f'SNAPSHOT {snapshot} ' + 10*'==' + '\n')


    aspect_ratio = (
        (y_cc.max() - y_cc.min()) /
        (x_cc.max() - x_cc.min())
    )

    # ------------------------------------------------------------------ #
    #  BUILD GEOMETRY MASK — STL or ib_markers                          #
    # ------------------------------------------------------------------ #
    if Use_STL_Mask and stl_profile is not None:
        print(f"  Building STL geometry mask...")
        geometry_mask = build_stl_geometry_mask(xx, yy, stl_profile)
    else:
        # Original ib_markers mask
        geometry_mask = np.where(np.abs(fields_data['ib_markers']) > 0, 1, np.nan)


    if Main_plot:
        Multi = False
        Zoom = True

        
        if Multi:
            print("Made it to Plotting...")
            
            cmap = Colormaps()

            fig, ax = plt.subplots(2, 2, figsize=(30, 30))

            # We store the "mappable" object returned by pcolormesh to pass to the colorbar later
            plots = []

            # 1. Density
            p0 = ax[0, 0].pcolormesh(
                xx, yy, fields_data['rho'],
                vmin=0, vmax=0.08,
                cmap='viridis', shading='gouraud'
            )
            plots.append(p0)

            # 2. Vorticity
            p1 = ax[0, 1].pcolormesh(
                xx, yy, fields_data['$Vorticity/Vorticity_{char}$'],
                vmin=-250, vmax=250,
                #norm=LogNorm(vmin=fields_data['Temperature'].min(), vmax=fields_data['Temperature'].max()),
                cmap=cmap["Blue2Red"], shading='gouraud'
            )
            plots.append(p1)

            # 3. Mach Number
            p2 = ax[1, 0].pcolormesh(
                xx, yy, fields_data['Mach'],
                vmin=fields_data['Mach'].min(), vmax=7,
                cmap='turbo', shading='gouraud'
            )
            plots.append(p2)

            # 4. Schlieren
            p3 = ax[1, 1].pcolormesh(
                xx, yy, fields_data['$|∇ρ|/(ρ∞/L_{char})$'],
                vmin=0, vmax= 1500, 
                cmap='gray_r', shading='gouraud'
            )
            plots.append(p3)

            # Titles for each subplot
            titles = ['Density (kg/m^3)', '$Vorticity/Vorticity_{char}$', 'Mach Number', '$|∇ρ|/(ρ∞/L_{char})$']

            print("Finished plotting, now applying formatting and saving...")

            # Apply formatting and colorbars
            #  Zip 'plots' here so we use the correct mappable for each axis
            for i, (a, title, p) in enumerate(zip(ax.flat, titles, plots)):
                cbar = fig.colorbar(p, ax=a) # Use the explicit plot object 'p'
                cbar.ax.tick_params(labelsize=20)
                a.tick_params(labelsize=20)
                a.set_xlabel('Distance from Leading Edge (m)', fontsize=24)
                a.set_title(title, fontsize=28)

            if Zoom:
                # x-ranges define the region of interest; y-range is auto-derived
                # from the domain midpoint so it works for any simulation domain size.
                # A square physical window (x_width == y_height) fills the axes
                # correctly when set_aspect('equal') is active.
                y_mid = 0.5 * (y_cc.min() + y_cc.max())

                if ZoomState == "Base":
                    xmin, xmax = x_cc.min(), x_cc.max()
                    Img_extra = ""
                elif ZoomState == "Sec1":
                    xmin, xmax = -0.05, 0.15
                    Img_extra = "_Sec1"
                elif ZoomState == "Sec2":
                    xmin, xmax = 0.4, 1.0
                    Img_extra = "_Sec2"
                else:
                    xmin, xmax = x_cc.min(), x_cc.max()
                    Img_extra = ""

                x_half = 0.5 * (xmax - xmin)
                ymin = max(y_mid - x_half, y_cc.min())
                ymax = min(y_mid + x_half, y_cc.max())

                for a in ax.flat:
                    a.set_xlim(xmin, xmax)
                    a.set_ylim(ymin, ymax)
                    a.set_aspect('equal')  # enforces equal physical scale in x and y

            plt.tight_layout()

            # Geometry Mask - Kept as contourf for sharp edges
            ax[0,0].contourf(xx, yy, geometry_mask, colors='white'); ax[0,1].contourf(xx, yy, geometry_mask, colors='grey')
            ax[1,0].contourf(xx, yy, geometry_mask, colors='white'); ax[1,1].contourf(xx, yy, geometry_mask, colors='black')

            # Snapshot Logic
            if snapshot < 10:
                sn_str = f'0000{snapshot}'
            elif snapshot < 100:
                sn_str = f'000{snapshot}'
            elif snapshot < 1000:
                sn_str = f'00{snapshot}'
            elif snapshot < 10000:
                sn_str = f'0{snapshot}'
            elif snapshot < 100000:
                sn_str = f'{snapshot}'
            else:        
                sn_str = f'{snapshot}'

            plt.savefig(f'Images{Img_extra}/fig_{sn_str}.png')
            plt.close()

            print(f"Finished processing snapshot {snapshot}.\n\n")

        else:

            # 1. Use pcolormesh with gouraud shading for smooth gradient
            # Determine zoom limits first so we can size the figure correctly
            if Zoom:
                y_mid = 0.5 * (y_cc.min() + y_cc.max())

                if ZoomState == "Base":
                    xmin, xmax = x_cc.min(), x_cc.max()
                    Img_extra = ""
                elif ZoomState == "Sec1":
                    xmin, xmax = -0.05, 0.15
                    Img_extra = "_Sec1"
                elif ZoomState == "Sec2":
                    xmin, xmax = 0.4, 1.0
                    Img_extra = "_Sec2"
                else:
                    xmin, xmax = x_cc.min(), x_cc.max()
                    Img_extra = ""

                x_half = 0.5 * (xmax - xmin)
                ymin = max(y_mid - x_half, y_cc.min())
                ymax = min(y_mid + x_half, y_cc.max())
            else:
                xmin, xmax = x_cc.min(), x_cc.max()
                ymin, ymax = y_cc.min(), y_cc.max()
                Img_extra = ""

            # Size figure to match physical aspect ratio so colorbar stays proportional
            fig_width = 20
            domain_aspect = (ymax - ymin) / (xmax - xmin)
            fig_height = max(fig_width * domain_aspect + 1.5, 4)  # +1.5 for labels/title
            fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

            p = ax.pcolormesh(
                xx, yy, fields_data['Mach'],
                vmin=0, vmax=7.7,
                cmap='turbo', 
                shading='gouraud'
            )

            #ax.contourf(xx, yy, geometry_mask, colors='black')
        
            # 2. Pass the plot object 'p' to the colorbar
            cbar = fig.colorbar(p, ax=ax)
            cbar.ax.tick_params(labelsize=16)

            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect('equal')
            ax.set_title('Mach', fontsize=20)
            ax.tick_params(labelsize=20)
            plt.tight_layout()
            
            if snapshot < 10:
                sn_str = f'0000{snapshot}'
            elif snapshot < 100:
                sn_str = f'000{snapshot}' 
            elif snapshot < 1000:
                sn_str = f'00{snapshot}'
            elif snapshot < 10000:
                sn_str = f'0{snapshot}'
            elif snapshot < 100000:
                sn_str = f'{snapshot}'
            else:        
                sn_str = f'{snapshot}'
            plt.savefig(run_root / f'fig_{sn_str}.png', dpi=200)
            plt.close()

    return

    
if __name__ == '__main__':


    snapshots = list(range(0, 100000, 10000))  # Test with just 4 snapshots first
    print(f"Snapshots to process: {snapshots}", flush=True)
    
    ZoomState = "Sec1"
    '''
    for i, sn in enumerate(snapshots):
        print(f"[{i+1}/{len(snapshots)}] Processing snapshot {sn}...", flush=True)
        try:
            mfcpp(sn, True, False, False, ZoomState, False)
            print(f"✓ Completed {sn}", flush=True)
        except Exception as e:
            print(f"✗ Failed {sn}: {e}", flush=True)
            import traceback
            traceback.print_exc()
    
    print("\n✓ All done!", flush=True)






    snapshots = list(range(288600, 445000, 50))
    ZoomState = "Base"

    def process_snapshot(sn):
        """Wrapper for multiprocessing — each worker handles one snapshot."""
        try:
            mfcpp(sn, True, False, False, ZoomState, False)
            return (sn, True, None)  # Return success tuple
        except Exception as e:
            return (sn, False, str(e))  # Return failure tuple

    num_workers = 7
    print(f"Starting multiprocessing with {num_workers} workers for {len(snapshots)} snapshots...", flush=True)
    
    # CRITICAL: Store results to force completion
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_snapshot, snapshots, chunksize=1)
        pool.close()  # No more tasks
        pool.join()   # WAIT for all to finish
    
    # Analyze results
    successful = sum(1 for sn, success, err in results if success)
    failed = sum(1 for sn, success, err in results if not success)
    
    print("\n" + "="*70, flush=True)
    print(f"✓ Processing Complete!", flush=True)
    print(f"  Total snapshots: {len(snapshots)}", flush=True)
    print(f"  Successful: {successful}", flush=True)
    print(f"  Failed: {failed}", flush=True)
    
    # Show first few failures
    failures = [(sn, err) for sn, success, err in results if not success]
    if failures:
        print(f"\nFirst 5 failures:", flush=True)
        for sn, err in failures[:5]:
            print(f"  - Snapshot {sn}: {err}", flush=True)
    
    print("="*70, flush=True)
'''


    for sn in range(0, 100000, 10000):
        ZoomState = "Sec1"
        mfcpp(sn, True, False, False, ZoomState, False)

    exit()
