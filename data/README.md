# Data

Download the deposit from Zenodo (<https://doi.org/10.5281/zenodo.24410726>) and
place the three season folders here, keeping their names:

```text
data/
├── 23_24_N/                                  nitrogen, 2023–2024 (model development)
│   ├── trait_N_weather_yield_combined_conc.xlsx   plot-level modelling table
│   ├── fert_by_plot_daily_lb2kg.xlsx             daily fertilizer per plot (sheet "Combined")
│   ├── FAWN_report_metric_23_24.xlsx             daily weather (FAWN station, metric units)
│   └── <YYYYMMDD>_AS_S2/                         one folder per UAV flight
│       ├── ortho_by_plot/<plot>.tif              5-band multispectral tile
│       └── dem_by_plot/<plot>.tif                DEM tile
├── 24_25_K/                                  potassium, 2024–2025 (model development)
│   ├── trait_K_weather_yield_combined_conc.xlsx
│   ├── fert_by_plot_daily_lb2kg.xlsx
│   ├── FAWN_report_metric_24_25.xlsx
│   └── <YYYYMMDD>_AS_S2/...
└── 25_26_K/                                  potassium, 2025–2026 (external validation only)
    ├── trait_K_weather_yield_combined_conc.xlsx
    ├── fert_by_plot_daily_lb2kg.xlsx
    ├── FAWN_report_metric_25_26.xlsx
    └── <YYYYMMDD>_AS_S2/...
```

If the data are stored elsewhere, link them here, e.g.
`ln -s /path/to/deposit/23_24_N data/23_24_N`.

Modelling-table columns used by the code:

| Group | Columns |
|---|---|
| identifiers and dates | `Image_ID` (plot), `htp_date` (UAV flight), `conc_date` (leaf sampling), `day` (days after transplanting at the flight) |
| target | `N (%)` (2023–2024) or `K (%)` (2024–2025, 2025–2026), % dry mass |
| V spectral descriptors | `*_Average`, `*_StdDev` of 55 vegetation indices and bands |
| S structural descriptors | `Canopy_Coverage_pixel`, `Canopy_Coverage_cm_2`, `Relative_Average_Height_Top_5_cm`, `Relative_Volume_cm_3` |
| C cultivar | `cv_*` (one-hot) |
| P plot position | `Centroid_X`, `Centroid_Y` |
| T timing | days after transplanting, \|sampling − flight\| interval |
| Fagg | `N_total` or `K_total` (cumulative nutrient applied, recomputed to the sampling date by `final_runs`) |
| Wagg | weather aggregates (recomputed from the daily FAWN file by `final_runs`) |

Units: daily fertilizer amounts in `fert_by_plot_daily_lb2kg.xlsx` are in kg ha⁻¹;
the cumulative amounts `N_total` / `K_total` in the modelling tables are in
lb acre⁻¹ (1 lb acre⁻¹ = 1.12085 kg ha⁻¹), and `final_runs` keeps that unit when it
rebuilds them to the sampling date.
