# This is a Python file used as the dashboard to execute the other scripts in Run/FuncDatabase.py

import os
import numpy as np
import matplotlib.pyplot as plt

from RunDatabase import *
from FuncDatabase import *

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


LineExtraction_Bool = True
StreamlinePlot_Bool = False

X_LOCATION = 0.05
UNIT_REYNOLDS = 4.2e6
FREESTREAM_U = 1726.0

PeleC = [
        {
            "source": script_path("..", "PeleC", "SBLI-Driver", "pelec-flatplate", "pltFlatPlate50000"),
            "label": "PeleC-FPV1-ISO",
        },
        {
            "source": script_path("..", "PeleC", "SBLI-Driver", "pelec-flatplate-Adiabatic", "pltFlatPlate50000"),
            "label": "PeleC-FPV2-ADIA",
        },
        #{
        #    "source": script_path("..", "PeleC", "SBLI-Driver", "pelec-CompRamp-Adia", "plt272030"),
        #    "label": "PeleC-CP-Adia",
        #},
    ]

MFC = [
    {
        "source": script_path("..", "Runs-Folder", "Hyp-Lam-FPV4", "silo_hdf5", "p0", "60000.silo"),
        "label": "MFC-FPV4-ADIA-Chapman_Enskog",
        "species": True,
    },
    {
        "source": script_path("..", "Runs-Folder", "Hyp-Lam-FPV5-Adia", "silo_hdf5", "p0", "60000.silo"),
        "label": "MFC-FPV5-ADIA",
        "linestyle": "--",
        "species": True,
    },
    {
        "source": script_path("..", "Runs-Folder", "Hyp-Lam-FPV5", "silo_hdf5", "p0", "60000.silo"),
        "label": "MFC-FPV5-ISO",
        "linestyle": "--",
        "species": True,
    }
]

AltSource_Velo = [
    #{
    #    "source": script_path("..", "Runs-Folder", "Hyp-Lam-FPV4", "Probe_Plots", "bl_M7p7_Tinf125K.dat"),
    #    "label": "Vincenzo Data",
    #    "columns": {"y": 0, "value": 1},
    #    "y_transform": "boundary_layer_nondim",
    #    "value_scale": FREESTREAM_U,
    #},
    {
        "source": script_path("..", "Runs-Folder", "Hyp-Lam-FPV5", "Probe_Plots", "uvel_x_0p05_check.dat"),
        "label": "Vincenzo-ADIA-Sutherlands",
        "columns": {"y": "y", "value": "V"},
    }
]

AltSource_Temp = [
    {
        "source": script_path("..", "Runs-Folder", "Hyp-Lam-FPV5", "Probe_Plots", "temp_x_0p05_check.dat"),
        "label": "Vincenzo-ADIA-Sutherlands",
        "columns": {"y": "y", "value": "T"},
    }
]


if LineExtraction_Bool == True:

    temperature_plot_path = script_path("Probe_Plots", "Temperature_profile.png")
    temperature_figure, temperature_axis = plt.subplots(figsize=(8, 6))

    result = LineExtraction(
        pelec_cases=PeleC,
        mfc_cases=MFC,
        field_key="Temperature",
        x_location=X_LOCATION,
        surface_y=0.0,
        y_max=0.005,
        field_label="Temperature [K]",
        title="Flat-Plate Surface-Normal Temperature Profiles",
        output_path=None,
        ax=temperature_axis,
    )

    if len(AltSource_Temp) > 0:
        alt_profile = load_alt_profile(
        AltSource_Temp[0],
        x_location=X_LOCATION, unit_reynolds=UNIT_REYNOLDS,
    )
        temperature_axis.plot(
            alt_profile['values'],
            alt_profile['y'],
            linewidth=2.0,
            linestyle='-',
            color='black',
            label=alt_profile['label'],
        )
        temperature_axis.legend()

    temperature_figure.tight_layout()
    temperature_figure.savefig(temperature_plot_path, dpi=200)

    
    velocity_plot_path = script_path("Probe_Plots", "UVelo_profile.png")
    velocity_figure, velocity_axis = plt.subplots(figsize=(8, 6))

    result = LineExtraction(
        pelec_cases=PeleC,
        mfc_cases=MFC,
        field_key="u",
        x_location=X_LOCATION,
        surface_y=0.0,
        y_max=0.005,
        field_label="u [m/s]",
        title="Flat-Plate Surface-Normal u Profiles",
        output_path=None,
        ax=velocity_axis,
    )

    if len(AltSource_Velo) > 0:
        alt_profile = load_alt_profile(
        AltSource_Velo[-1],
        x_location=X_LOCATION, unit_reynolds=UNIT_REYNOLDS,
    )
        
        velocity_axis.plot(
            alt_profile['values'],
            alt_profile['y'],
            linewidth=2.0,
            linestyle='-',
            color='black',
            label=alt_profile['label'],
        )
        velocity_axis.legend()

    velocity_figure.tight_layout()
    velocity_figure.savefig(velocity_plot_path, dpi=200)


if StreamlinePlot_Bool == True:

    streamline_plot_path = script_path("Probe_Plots", "Streamlines.png")
    streamline_figure, streamline_axis = plt.subplots(figsize=(10, 4))

    streamline_result = StreamlinePlot(
        pelec_cases=PeleC[-1],
        #mfc_cases=[MFC[-1]],
        color_key="speed",
        mask_key="vfrac",
        mask_threshold=0.1,
        mask_mode="below",
        title="Steady Streamlines from Velocity Field",
        output_path=None,
        xlim=(0.0, 0.2),
        ylim=(0.0, 0.04),
        density=(10.0, 10.0),
        linewidth=0.75,
        cmap="turbo",
        seed_lines=[
            {
                "count": 30,
                "x_position": 0.002,
                "y_min": 0.0,
                "y_max": 0.04,
                "spacing": "power",
                "power": 1.0,
            },
            {
                "count": 50,
                "x_position": 0.055,
                "y_min": 0.000,
                "y_max": 0.0075,
                "spacing": "power",
                "power": 3.0,
            },
            {
                "count": 50,
                "x_position": 0.095,
                "y_min": 0.000,
                "y_max": 0.01,
                "spacing": "power",
                "power": 3.0,
            },
        ],
        colorbar=True,
        colorbar_label="|V| [m/s]",
        mask_fill_color="black",
        ax=streamline_axis,
    )

    streamline_figure.tight_layout()
    streamline_figure.savefig(streamline_plot_path, dpi=300)
    