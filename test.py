# %%
import pandas as pd

csv_file = "data.csv"
df = pd.read_csv(csv_file)
df["solve_time"] = df["cycle_2"] - df["cycle_1"]
# remove entry 4
df = df.drop(4)
df
# %%

constant_1 = 251 - 5
constant_2 = 279 - 5
constant_3 = 301 - 5

min_max_time = 1000

for contant_1 in range(210, 290):
    for constant_2 in range(210, 290):
        for constant_3 in range(299 - 5, 304 - 5):
            tts_1_df = df[df["File"] <= constant_1].copy()
            tts_2_df = df[df["File"] <= constant_2].copy()
            tts_3_df = df[df["File"] <= constant_3].copy()

            tts_1_df["max_file"] = tts_1_df["File"] + 4
            tts_2_df["max_file"] = tts_2_df["File"] + 4
            tts_3_df["max_file"] = tts_3_df["File"] + 4

            tts_1_df["steps"] = (constant_1 - tts_1_df["File"]) // 5
            tts_1_df["grabbed_file"] = constant_1 - tts_1_df["steps"] * 5
            tts_1_df["extra_steps"] = tts_1_df["max_file"] - tts_1_df["grabbed_file"]
            tts_1_df["cycle_predicted"] = (
                tts_1_df["steps"] * 2 + tts_1_df["extra_steps"] * 2 + 15
            )
            tts_1_df["final_time"] = (
                tts_1_df["cycle_predicted"] + tts_1_df["solve_time"]
            )

            tts_2_df["steps"] = (constant_2 - tts_2_df["File"]) // 5
            tts_2_df["grabbed_file"] = constant_2 - tts_2_df["steps"] * 5
            tts_2_df["extra_steps"] = tts_2_df["max_file"] - tts_2_df["grabbed_file"]
            tts_2_df["cycle_predicted"] = (
                tts_2_df["steps"] * 2 + tts_2_df["extra_steps"] * 2 + 17
            )
            tts_2_df["final_time"] = (
                tts_2_df["cycle_predicted"] + tts_2_df["solve_time"]
            )
            tts_2_df

            tts_3_df["steps"] = (constant_3 - tts_3_df["File"]) // 5
            tts_3_df["grabbed_file"] = constant_3 - tts_3_df["steps"] * 5
            tts_3_df["extra_steps"] = tts_3_df["max_file"] - tts_3_df["grabbed_file"]
            tts_3_df["cycle_predicted"] = (
                tts_3_df["steps"] * 2 + tts_3_df["extra_steps"] * 2 + 18
            )
            tts_3_df["final_time"] = (
                tts_3_df["cycle_predicted"] + tts_3_df["solve_time"]
            )
            tts_3_df

            # merge the dataframes on the "TR" column, and aggregate the "final_time" column as min
            tts_1_df = tts_1_df[["TR", "final_time"]]
            tts_2_df = tts_2_df[["TR", "final_time"]]
            tts_3_df = tts_3_df[["TR", "final_time"]]
            tts_df = pd.concat([tts_1_df, tts_2_df, tts_3_df])
            tts_df = tts_df.groupby("TR").min().reset_index()

            max_time = tts_df["final_time"].max()
            if max_time < min_max_time:
                min_max_time = max_time
                best_constants = (constant_1, constant_2, constant_3)
                print(min_max_time, best_constants)


# %%


constant_1 = 251 + 5
constant_2 = 279 + 5
constant_3 = 209 + 5

min_max_time = 1000


for contant_1 in range(210, 290):
    for constant_2 in range(210, 290):

        tts_1_df = df[df["File"] >= constant_1].copy()
        tts_2_df = df[df["File"] >= constant_2].copy()
        tts_3_df = df[df["File"] >= constant_3].copy()

        tts_1_df["max_file"] = tts_1_df["File"] + 4
        tts_2_df["max_file"] = tts_2_df["File"] + 4
        tts_3_df["max_file"] = tts_3_df["File"] + 4

        tts_1_df["steps"] = (tts_1_df["max_file"] - constant_1) // 5
        tts_1_df["grabbed_file"] = tts_1_df["steps"] * 5 + constant_1
        tts_1_df["extra_steps"] = tts_1_df["max_file"] - tts_1_df["grabbed_file"]
        tts_1_df["cycle_predicted"] = (
            tts_1_df["steps"] * 2 + tts_1_df["extra_steps"] * 2 + 15
        )
        tts_1_df["final_time"] = tts_1_df["cycle_predicted"] + tts_1_df["solve_time"]

        tts_2_df["steps"] = (tts_2_df["max_file"] - constant_2) // 5
        tts_2_df["grabbed_file"] = tts_2_df["steps"] * 5 + constant_2
        tts_2_df["extra_steps"] = tts_2_df["max_file"] - tts_2_df["grabbed_file"]
        tts_2_df["cycle_predicted"] = (
            tts_2_df["steps"] * 2 + tts_2_df["extra_steps"] * 2 + 17
        )
        tts_2_df["final_time"] = tts_2_df["cycle_predicted"] + tts_2_df["solve_time"]
        tts_2_df

        tts_3_df["steps"] = (tts_3_df["max_file"] - constant_3) // 5
        tts_3_df["grabbed_file"] = tts_3_df["steps"] * 5 + constant_3
        tts_3_df["extra_steps"] = tts_3_df["max_file"] - tts_3_df["grabbed_file"]
        tts_3_df["cycle_predicted"] = (
            tts_3_df["steps"] * 2 + tts_3_df["extra_steps"] * 2 + 18
        )
        tts_3_df["final_time"] = tts_3_df["cycle_predicted"] + tts_3_df["solve_time"]
        tts_3_df

        # merge the dataframes on the "TR" column, and aggregate the "final_time" column as min

        tts_df = pd.concat([tts_1_df, tts_2_df, tts_3_df])
        tts_df = tts_df.groupby("TR").min().reset_index()
        tts_3_df

        max_time = tts_df["final_time"].max()
        if max_time < min_max_time:
            min_max_time = max_time
            best_constants = (constant_1, constant_2, constant_3)
            print(min_max_time, best_constants)
# %%

import numpy as np

x = np.arange(1, 49, 6)
x

for factor in range(-200, 200):
    y = x * factor
    print(y)
