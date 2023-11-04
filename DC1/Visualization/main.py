import matplotlib.pyplot as plt
from celluloid import Camera
import csv

plt.rcParams["animation.ffmpeg_path"] = "/opt/homebrew/bin/ffmpeg"


def main(input_file: str):
    figure = plt.figure(figsize=(8, 6), dpi=250)
    axes = plt.axes()
    axes.set_xlim(-600, 600)
    axes.set_ylim(-600, 600)
    camera = Camera(figure)

    with open(input_file, "r") as csv_file:
        reader = csv.reader(csv_file)
        next(reader)
        step = 1
        for line in reader:
            line.pop(0)
            coords = [tuple(list(map(float, line[i:i + 2]))) for i in range(0, len(line), 2)]
            print(f"Step {step}: {coords}")
            step += 1
            for coord in coords:
                axes.scatter(coord[0], coord[1], color="red")
            camera.snap()

        print("Animation generation started")
        animation = camera.animate()
        print("Saving result")
        animation.save("output.mp4", fps=30)


if __name__ == "__main__":
    main("output.csv")
