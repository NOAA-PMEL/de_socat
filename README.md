# SOCAT Database Update Runbook: New Release Workflow

This runbook guides maintainers through refreshing and seeding the database tables powering the Dash application whenever a new SOCAT data release is published.

---

## 1. Prerequisites & Environment Setup

* **Database Service**: Ensure the target PostgreSQL database instance is running and accessible.


* **Connection String**: Set the `DATABASE_URL` environment variable in your working environment (if omitted, the notebooks default to `postgresql://postgres:docker@127.0.0.1:5432`).


* **Source Endpoints**: Open `constants.py` and update the VERSION_ID variable with the release tag. This will propogate to the `decimated_url` and `full_url` to target the new release endpoints on the NOAA PMEL ERDDAP server (e.g., updating the VERSION_ID suffix from `socat_v2025` to `socat_v2026`).



---

## 2. Notebook Execution Sequence

Execute the notebooks in the exact order below, as later stages depend on tables created in earlier steps.

```
[constants.py] 
       │ (endpoint update)
       ▼
1. write_expocodes.ipynb  ──► Creates/Replaces 'cruises' table
       │
       ├────────────────────────────────────────┐
       ▼                                        ▼
2. counts_dataset.ipynb                  3. cruise_traces.ipynb
   (Creates binned density dataset)         (Fetches & strides raw tracks)
                                                │
                                          You can do one or the other of
                                          these two notebooks. They both
                                          create subsamples from the decimated
                                          cruises. I think the second does
                                          a better job.
                                                │
                                                ▼
                                         4. convert_to_sample.ipynb
                                            (Applies geodesic RDP simplification)

```

### Step 1: Ingest Cruise Metadata (`write_expocodes.ipynb`)

* **Purpose**: Rebuilds the primary cruise catalog with all valid cruises in the new release.


* **Action**: Open and execute all cells in `write_expocodes.ipynb`.


* **What Happens**:
* Queries the new ERDDAP decimated dataset using `distinct()` to retrieve unique cruise entries.


* Parses metadata fields: `expocode`, `organization`, `investigators`, `platform_name`, `platform_type`, `qc_flag`, and `socat_version`.


* Replaces the `cruises` table in the PostgreSQL database.




* **Validation**: Ensure the final verification cell confirms that known expocodes exist in the database and that the row count matches the expected total for the release (e.g., >8,000–9,000 cruises).



### Step 2: Generate Spatial Density Grids (`counts_dataset.ipynb`)

* **Purpose**: Pre-aggregates global cruise observations into coordinate bins for the Dash heatmap and regional summary layers.


* **Action**: Open and execute all cells in `counts_dataset.ipynb`.


* **What Happens**:
* Downloads all observation coordinates, timestamps, and regional flags (`region_id`, `all_region_ids`) from ERDDAP.


* Bins latitude into 80 intervals and longitude into 180 intervals.


* Aggregates observation totals per grid cell and regional code.


* Calculates cell midpoints and logarithmic scale counts ($\log_{10}$) for rendering.




* **Validation**: Confirm the output table contains non-zero counts with valid centers within $[-90, 90]$ latitude and $[-180, 180]$ longitude.

## You can do one other the other of these last two.

### Step 3: Fetch & Downsample Raw Cruise Traces (`cruise_traces.ipynb`)

* **Purpose**: Fetches the individual coordinate paths for every cruise and applies a uniform stride downsampling to heavy traces.


* **Action**: Open and run `cruise_traces.ipynb`.


* **What Happens**:
* Reads all target expocodes from the seeded `cruises` database table.


* Iterates through every cruise and queries ERDDAP for timestamped latitude, longitude, and platform type.


* Applies a stride interval if the point count exceeds the target threshold (~100 points) to prevent browser rendering lag.




* **Note**: Because this step issues an individual ERDDAP query per cruise, execution will take substantial time. Ensure a stable network connection before running.



### Step 4: Geodesic Path Simplification (`convert_to_sample.ipynb`)

* **Purpose**: Optimizes track line geometries for map display by preserving significant turns while reducing straight-line segments.


* **Action**: Open and run `convert_to_sample.ipynb`.


* **What Happens**:
* Loads the cruise list from the database.


* Converts coordinate paths to ellipsoidal Vincenty `LatLon` representations.


* Applies the Ramer-Douglas-Peucker (RDP) algorithm with antimeridian longitude wrapping.


* Saves the final simplified point collections for Dash rendering.





---

## 3. Post-Run Verification Checklist

| Verification Item | Check Description | Success Criteria |
| --- | --- | --- |
| **Cruise Catalog** | Run `SELECT count(*) FROM cruises;` in PostgreSQL.

 | Count matches official release documentation (~8,000–9,500+ records).

 |
| **Binned Observations** | Check the binned count dataframe in `counts_dataset.ipynb`.

 | Filtered rows with `count > 0` are present across expected oceanic regions.

 |
| **Trajectory Counts** | Verify terminal output logs in `cruise_traces.ipynb`.

 | Each expocode prints initial size, stride, and final size without unhandled HTTP errors.

 |
| **Simplification Integrity** | Review output logs in `convert_to_sample.ipynb`.

 | Shows reduced point counts for dense tracks while retaining short tracks unchanged.

 |
| **Application Check** | Launch the Dash application locally. | Tracklines, overview density overlays, and cruise dropdown selections render without timeouts. |

---

## 4. Maintainer Notes & Common Failure Modes

* **Overwriting Active Production Tables**: Running `write_expocodes.ipynb` executes `if_exists="replace"`, dropping and recreating the `cruises` table. If the Dash app is serving live traffic, run updates against an isolated database or staging schema first, then swap tables.


* **Date Line & Polar Route Warnings**: During `convert_to_sample.ipynb`, you may observe messages such as `delta (X) exceeds limit (45)` for high-latitude polar cruises or trans-Pacific tracks. The simplification library leaves these specific problematic segments unchanged to avoid distorted horizontal lines across the map.


* **ERDDAP Rate Limiting & Interrupted Runs**: If network instability interrupts `cruise_traces.ipynb` partway through, modify the loop filter to resume from the last completed expocode rather than restarting from index 0.