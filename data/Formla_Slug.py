
import polars as pl
import matplotlib.pyplot as plt
import numpy as np


df = pl.read_parquet("~/Downloads/software-data.parquet").fill_null(strategy="forward")
print(df)




#All the givens are Motor RPM, the ratio for sproket GR = 12/41, tire radius WR = 0.2m 
# velocity = mRPM * 12/41 * 2pi/60 * 0.2
# we need to change it to mile per hour
# which means we multiply v by 2.23694
# v_mile = v* 2.23694


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
plt.show()

#Step 3

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
plt.show()

# Step 4

# Lap times found from the GPS track. One lap = one drive from standing still until the car stops again.
# The car drove a straight back and forth, turning around at the east end of the track.
# Each lap starts when the driver presses the throttle (start of the first Accelerating frame in Step 3)
# and ends when the speed drops below 0.5 m/s.
laps = [
    {"lap": 1, "start": 18.92, "end": 76.86},   # out to the east end and back to the west end (about 293 m)
    {"lap": 2, "start": 94.24, "end": 131.84},  # out to the east end, then stopped there without coming back (about 188 m)
]

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
plt.show()

#Step 5
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
plt.show()