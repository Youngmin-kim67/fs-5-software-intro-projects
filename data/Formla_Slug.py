
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path




plot_dir = Path(__file__).parent / "plots"
plot_dir.mkdir(exist_ok = True)  # make the folder if it is not there yet

def save_plot(file_name):
    plt.savefig(plot_dir / file_name, dpi = 150)  # dpi = 150 makes the picture sharper than the default
    print(f"Saved {plot_dir / file_name}")


def print_step(number, title):
    print()
    print(f"========== Step {number}: {title} ==========")


print_step(1, "Forward fill and print the data")
df = pl.read_parquet("~/Downloads/software-data.parquet").fill_null(strategy="forward")
print(df)




#All the givens are Motor RPM, the ratio for sproket GR = 12/41, tire radius WR = 0.2m
# velocity = mRPM * 12/41 * 2pi/60 * 0.2
# we need to change it to mile per hour
# which means we multiply v by 2.23694
# v_mile = v* 2.23694

print_step(2, "Speed (mph) vs time, and the speed at 10 s")
time = df["Time"]  # Time is already in seconds (0 ~ 164.58 s, one sample every ~0.012 s), so no conversion is needed.
#We use the equation that we got from the line 12
mRPM  = df["SME_TRQSPD_Speed"]
vms = mRPM * (12/41) *  (2 * np.pi /60) * 0.2
vmph = vms * 2.23694

df = df.with_columns(
    vms.alias("speed_ms"),
    vmph.alias ("speed_mph")
)

# We want to make a plot for speed vs time to visualize it better

plt.figure(figsize = (10,4))
plt.plot(time, vmph, label = "Speed(mph)", color = "red") # We can always change vmph to vms if we want different value. 
plt.xlabel ( "TIME")
plt.ylabel("Speed(mph)")
plt.grid(True)

# Find the speed at 10 s: take the first row at or after 10 s and mark it on the graph
row_10s = df.filter(pl.col("Time") >= 10)[0]
t_10s = row_10s["Time"][0]
mph_10s = row_10s["speed_mph"][0]
ms_10s = row_10s["speed_ms"][0]
print(f"Speed at {t_10s:.2f} s: {mph_10s:.2f} mph ({ms_10s:.2f} m/s)")
plt.plot(t_10s, mph_10s, "o", color = "blue", label = f"10 s: {mph_10s:.2f} mph")
plt.legend()
save_plot("step2_speed_vs_time.png")
# No plt.show() here: every window is shown once at the very end, so all the printed results come out first

#Step 3
print_step(3, "Time frames for accelerating, braking, and coasting")

# We save each state as a condition so we can reuse it in Step 5 and Step 6
accelerating = (pl.col("SME_THROTL_TorqueDemand") > 100) & (pl.col("ETC_STATUS_BRAKE_SENSE_VOLTAGE") <= 375)
coasting = (pl.col("SME_THROTL_TorqueDemand") <= 100) & (pl.col("ETC_STATUS_BRAKE_SENSE_VOLTAGE") <= 375) & (pl.col("speed_ms") > 0.5)
braking = pl.col("ETC_STATUS_BRAKE_SENSE_VOLTAGE") > 375

states = df.select(accelerating.alias("Accelerating"),
                   coasting.alias("Coasting"),
                   braking.alias("Braking"))

times = df["Time"].to_numpy()

# Turn a state into time frames: walk through the rows one by one, and every run of
# consecutive True rows becomes one frame, saved as (start row, end row).
# Frames shorter than min_duration are dropped, since they are mostly sensor flicker.
def find_frames(on, min_duration):
    frames = []
    dropped = 0
    start = None
    for i in range(len(on)):
        if on[i] and start is None:
            start = i  # a new frame begins on this row
        is_last_row = (i == len(on) - 1)
        if start is not None and (not on[i] or is_last_row):
            if on[i]:
                end = i      # the data ran out while the frame was still going
            else:
                end = i - 1  # the frame ended on the row before this one
            if times[end] - times[start] >= min_duration:
                frames.append((start, end))
            else:
                dropped += 1
            start = None
    return frames, dropped

def print_frames(frames):
    print("  Start (s)   End (s)   Duration (s)")
    for start, end in frames:
        print(f"  {times[start]:9.2f} {times[end]:9.2f} {times[end] - times[start]:12.2f}")

for name in ["Accelerating", "Braking", "Coasting"]:
    frames, dropped = find_frames(states[name].to_numpy(), 0.5)
    print(f"{name}: {len(frames)} time frames of 0.5 s or longer ({dropped} shorter ones dropped)")
    print_frames(frames)

fig, axs = plt.subplots(3,1, figsize = (10,8), sharex = True)

axs[0].plot(time, states["Accelerating"].to_numpy(), color = "green")
axs[0].set_title("Accelerating")
axs[1].plot(time, states["Braking"].to_numpy(), color="red")
axs[1].set_title("Braking")

axs[2].plot(time, states["Coasting"].to_numpy(), color="orange")
axs[2].set_title("Coasting")
axs[2].set_xlabel("Time (s)")

plt.tight_layout()
save_plot("step3_driving_states.png")

# Step 4

# Lap times found from the GPS track. One lap = one drive from standing still until the car stops again.
# The car drove a straight back and forth, turning around at the east end of the track.
# Each lap starts when the driver presses the throttle (start of the first Accelerating frame in Step 3)
# and ends when the speed drops below 0.5 m/s.
laps = [
    {"lap": 1, "start": 18.92, "end": 76.86},   # out to the east end and back to the west end (about 293 m)
    {"lap": 2, "start": 94.24, "end": 131.84},  # out to the east end, then stopped there without coming back (about 188 m)
]

print_step(4, "Number of laps, with start and end time")
print(f"Laps driven: {len(laps)}")
print("  Lap   Start (s)   End (s)   Duration (s)")
for lap_info in laps:
    print(f"  {lap_info['lap']:3d} {lap_info['start']:11.2f} {lap_info['end']:9.2f} {lap_info['end'] - lap_info['start']:12.2f}")

# Draw the whole run in gray, then each lap in its own color on top.
# A circle marks where each lap starts and an X marks where it ends.
lap_colors = ["blue", "orange"]

plt.figure(figsize=(12, 5))
plt.scatter(df["VDM_GPS_Longitude"], df["VDM_GPS_Latitude"], s = 2, color = "lightgray", label = "Whole run")
for lap_info in laps:
    lap_df = df.filter((pl.col("Time") >= lap_info["start"]) & (pl.col("Time") <= lap_info["end"]))
    color = lap_colors[lap_info["lap"] - 1]
    first_row = lap_df[0]
    last_row = lap_df[lap_df.height - 1]
    plt.scatter(lap_df["VDM_GPS_Longitude"], lap_df["VDM_GPS_Latitude"], s = 4, color = color,
                label = f"Lap {lap_info['lap']}: {lap_info['start']:.2f} - {lap_info['end']:.2f} s")
    plt.scatter(first_row["VDM_GPS_Longitude"], first_row["VDM_GPS_Latitude"], s = 120, marker = "o", color = color, edgecolors = "black")
    # zorder = 3 keeps the X on top, since Lap 1 ends at the same spot where Lap 2 starts
    plt.scatter(last_row["VDM_GPS_Longitude"], last_row["VDM_GPS_Latitude"], s = 120, marker = "X", color = color, edgecolors = "black", zorder = 3)
plt.ticklabel_format(useOffset = False)  # show the real latitude/longitude instead of "+3.8575e1"
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("GPS Track Layout (circle = lap start, X = lap end)")
plt.legend()
plt.grid(True)
save_plot("step4_gps_track.png")

#Step 5
print_step(5, "Max speed, max acceleration, accel time, and coast time for each lap")
# List to store metrics for each lap
lap_results = []

for lap_info in laps:
    lap_df = df.filter(
        (time >= lap_info["start"]) & (time <= lap_info["end"])
    )
    

    max_speed = lap_df["speed_mph"].max()
    
   
    # The IMU X axis is flipped: speeding up reads negative, slowing down reads positive.
    # So the largest forward acceleration is the most negative value, with the sign turned around.
    max_accel = -lap_df["VDM_X_AXIS_ACCELERATION"].min()
    
 
    time_step = lap_df["Time"][1] - lap_df["Time"][0]
    
    # Use the same conditions as Step 3 so the states are defined the same way
    accel_time = lap_df.filter(accelerating).height * time_step
    coast_time = lap_df.filter(coasting).height * time_step
    
    lap_results.append({
        "Lap": lap_info["lap"],
        "Max Speed (mph)": round(max_speed, 2),
        "Max Accel (G)": round(max_accel, 2),
        "Accel Time (s)": round(accel_time, 2),
        "Coast Time (s)": round(coast_time, 2)
    })


summary_df = pl.DataFrame(lap_results)
print(summary_df)



#Step 6,7

m = 221.4      
m_eff = 244.08 
rho = 1.225    
g = 9.81      


# Step 6: keep coasting at 5 m/s or faster, then drop any coasting frame shorter than 1 second.
# The short frames are flickers right after a real coast-down, where the speed bounces around 5 m/s.
print_step(6, "Coast-down data (5 m/s or faster, 1 s or longer)")
coast_fast = coasting & (pl.col("speed_ms") >= 5.0)
coast_on = df.select(coast_fast.alias("on"))["on"].to_numpy()
coast_frames, dropped = find_frames(coast_on, 1.0)

# Cut each coast-down out of df as its own small DataFrame
coast_parts = []
rows_kept = 0
for start, end in coast_frames:
    part = df[start:end + 1]
    coast_parts.append(part)
    rows_kept += part.height

print(f"Coast-down data: {len(coast_frames)} frames of 1 s or longer ({dropped} shorter ones dropped), {rows_kept} rows kept")
print_frames(coast_frames)


# Step 7: dv/dt = (change in speed between two readings) / (time between them), as projects.md describes.
# We do it one coast-down at a time, so we never take a difference across the gap between two coast-downs.
# Each pair of readings gives one dv/dt, so v² uses the middle of the pair to line up with it.
print_step(7, "Drag (CdA) and rolling resistance (Crr) from the coast-down")
v_squared = []
dv_dt = []
for part in coast_parts:
    v = part["speed_ms"].to_numpy()
    t = part["Time"].to_numpy()
    for i in range(len(v) - 1):
        dv_dt.append((v[i + 1] - v[i]) / (t[i + 1] - t[i]))
        v_squared.append(((v[i] + v[i + 1]) / 2) ** 2)

slope, intercept = np.polyfit(v_squared, dv_dt, 1)


CdA = (-2 * slope * m_eff) / rho
Crr = (-intercept * m_eff) / (m * g)

print(f"CdA: {CdA:.4f} m²")
print(f"Crr: {Crr:.4f}")

# Plot dv/dt against v² with the fitted line: the slope gives drag, the intercept gives rolling resistance
x_line = np.array([min(v_squared), max(v_squared)])

plt.figure(figsize = (8, 5))
plt.plot(v_squared, dv_dt, '.', color = "gray", alpha = 0.3, label = "Coast-down data")
plt.plot(x_line, slope * x_line + intercept, color = "red", label = f"Fit: dv/dt = {slope:.5f} * v² {intercept:+.4f}")
plt.xlabel("v² (m²/s²)")
plt.ylabel("dv/dt (m/s²)")
plt.title("Coast-down: dv/dt vs v²")
plt.legend()
plt.grid(True)
save_plot("step7_coastdown_fit.png")


bin_centers = []
bin_means = []
bin_errors = []
print("  v² bin (m²/s²)   points   average dv/dt (m/s²)")
for low in range(25, 125, 10):
    high = low + 10
    # Collect every dv/dt whose v² falls in this bin
    in_bin = []
    for i in range(len(v_squared)):
        if v_squared[i] >= low and v_squared[i] < high:
            in_bin.append(dv_dt[i])
    in_bin = np.array(in_bin)

    if len(in_bin) < 50:
        print(f"  {low:5d} - {high:4d} {len(in_bin):8d}   (skipped: fewer than 50 points)")
    else:
        error = in_bin.std() / np.sqrt(len(in_bin))
        bin_centers.append((low + high) / 2)
        bin_means.append(in_bin.mean())
        bin_errors.append(error)
        print(f"  {low:5d} - {high:4d} {len(in_bin):8d}   {in_bin.mean():+.3f} ± {error:.3f}")

plt.figure(figsize = (8, 5))
plt.plot(v_squared, dv_dt, '.', color = "lightgray", alpha = 0.3, label = "Coast-down data (each pair of readings)")
# errorbar draws each average as a dot with a line above and below it (the error bar)
plt.errorbar(bin_centers, bin_means, yerr = bin_errors, fmt = "o", color = "black", capsize = 4,
             label = "Average in each v² bin (± standard error)")
plt.plot(x_line, slope * x_line + intercept, color = "red", label = f"Fit: dv/dt = {slope:.5f} * v² {intercept:+.4f}")
# The intercept alone is the rolling resistance part: it is the same at every speed, so it is a flat line.
plt.plot(x_line, [intercept, intercept], color = "blue", linestyle = "--", label = f"Rolling resistance only: {intercept:.3f} m/s²")
# fill_between colors the space between two lines.
# The gap between the flat blue line and the red line is the extra slowing from drag, which grows with v².
plt.fill_between(x_line, intercept, slope * x_line + intercept, color = "red", alpha = 0.15, label = "Extra slowing from drag")
plt.ylim(-1.5, 1.0)  # zoom in around the line; dots outside this range are still used in the fit
plt.xlabel("v² (m²/s²)")
plt.ylabel("dv/dt (m/s²)")
plt.title("Coast-down: dv/dt vs v² (zoomed in, with bin averages)")
plt.legend(loc = "upper left", fontsize = 8)
plt.grid(True)
save_plot("step7_coastdown_fit_zoomed.png")



# Step 9 (extra): how hard the brake is pressed, not just on/off
print_step("10 (Extra)", "How hard the brake is pressed")


brake_col = "ETC_STATUS_BRAKE_SENSE_VOLTAGE"
# Not braking = 375 or below. The first 4 rows read 0 before the sensor starts, so skip them too.
resting = df.filter((pl.col(brake_col) > 0) & (pl.col(brake_col) <= 375))[brake_col]
rest_v = resting.median()  # the middle value, so a few odd readings don't move it
max_v = df[brake_col].max()
brake_pct = ((df[brake_col] - rest_v) / (max_v - rest_v) * 100).to_numpy()
# .to_numpy() from polars can't be changed, so np.array() makes our own copy that we can change below
brake_pct = np.array(brake_pct)
for i in range(len(brake_pct)):
    if brake_pct[i] < 0:
        brake_pct[i] = 0  # readings a little below resting (and the first 4 zero rows) count as 0 %
threshold_pct = (375 - rest_v) / (max_v - rest_v) * 100
print(f"Resting voltage: {rest_v:.0f}, highest voltage: {max_v:.0f}, Step 3 threshold 375 = {threshold_pct:.1f} %")

brake_v = df[brake_col].to_numpy()
brake_f = df["TMAIN_DATA_BRAKES_F"].to_numpy()
brake_r = df["TMAIN_DATA_BRAKES_R"].to_numpy()
# Correlation: 1 = the two always move together, 0 = no relation.
# np.corrcoef gives a 2x2 table, and [0, 1] is the number between the two columns.
print(f"Brake voltage vs front pressure: correlation {np.corrcoef(brake_v, brake_f)[0, 1]:.2f}, "
      f"vs rear pressure: {np.corrcoef(brake_v, brake_r)[0, 1]:.2f}")

# For each braking frame from Step 3: how hard, and was the car still moving?
# F/R = front pressure / rear pressure at the hardest moment (1.0 means front and rear get the same pressure)
brake_frames, dropped = find_frames(states["Braking"].to_numpy(), 0.5)
speed = df["speed_ms"].to_numpy()
moving_count = 0
print("  Start (s)   End (s)   Max (%)   Mean (%)   Start v (m/s)   End v (m/s)   F/R at max")
for start, end in brake_frames:
    pct = brake_pct[start:end + 1]
    # Find the row where the brake was pressed hardest in this frame
    hardest = start
    for i in range(start, end + 1):
        if brake_pct[i] > brake_pct[hardest]:
            hardest = i
    print(f"  {times[start]:9.2f} {times[end]:9.2f} {pct.max():9.1f} {pct.mean():10.1f} "
          f"{speed[start]:15.2f} {speed[end]:13.2f} {brake_f[hardest] / brake_r[hardest]:12.2f}")
    if speed[start] > 0.5:
        moving_count = moving_count + 1
print(f"Braking frames that start while the car is moving (over 0.5 m/s): {moving_count} of {len(brake_frames)}")
print("-> The driver mostly pressed the brake to hold the car after it had already rolled to a stop.")

fig, axs = plt.subplots(2, 1, figsize = (12, 6), sharex = True)
axs[0].plot(times, brake_pct, color = "red")
axs[0].plot([times[0], times[len(times) - 1]], [threshold_pct, threshold_pct], color = "gray", linestyle = "--",
            label = f"Step 3 threshold (375 = {threshold_pct:.1f} %)")
axs[0].set_ylabel("Brake pressed (%)")
axs[0].set_title("How hard the brake is pressed (0 % = resting, 100 % = hardest press in this log)")
axs[0].legend(loc = "upper right")
axs[0].grid(True)
axs[1].plot(times, speed, color = "black")
axs[1].set_ylabel("Speed (m/s)")
axs[1].set_xlabel("Time (s)")
axs[1].set_title("Speed (to see whether the car was moving while braking)")
axs[1].grid(True)
# Shade the Step 3 braking frames on both graphs. axvspan colors the area between two times.
for start, end in brake_frames:
    axs[0].axvspan(times[start], times[end], color = "red", alpha = 0.1)
    axs[1].axvspan(times[start], times[end], color = "red", alpha = 0.1)
plt.tight_layout()
save_plot("step10_brake_strength.png")

# Show all 7 graphs at once. The script ends when the last window is closed.
plt.show()

