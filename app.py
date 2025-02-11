import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect

# Flask App Initialization
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Upgrade requirements for each level
upgrade_requirements = {
    'Welfare': [0, 2000, 4500, 12000, 25000, 65000, 180000, 250000],
    'Coin Earning': [0, 800, 1500, 4500, 16000, 29000, 85000, 200000],
    'Luck': [0, 4000, 15000, 50000, 180000, 400000, 900000, 1500000],
    'Size': [0, 2000, 4500, 20000, 75000, 200000, 450000, 1000000]
}

class ProgressBarAnalyzer:
    def __init__(self, goal_crowns):
        self.goal_crowns = goal_crowns
        self.color_ranges = {
            'orange_yellow': {
                'lower': np.array([15, 100, 100]),  # Adjusted to capture more yellow
                'upper': np.array([45, 255, 255])   # Adjusted to capture more orange
            }
        }

    def load_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image from {image_path}")
        return image

    def preprocess_image(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)
        return hsv

    def create_mask(self, hsv_image, color_key='orange_yellow'):
        color_range = self.color_ranges[color_key]
        mask = cv2.inRange(hsv_image, color_range['lower'], color_range['upper'])
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def analyze_progress(self, image_path):
        image = self.load_image(image_path)
        hsv = self.preprocess_image(image)
        mask = self.create_mask(hsv)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None, None

        progress_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(progress_contour)

        total_width = image.shape[1]
        progress_percentage = (w / total_width) * 100
        crowns_donated = int((progress_percentage / 100) * self.goal_crowns)
        crowns_remaining = self.goal_crowns - crowns_donated

        return progress_percentage, crowns_donated, crowns_remaining

    def visualize_results(self, image_path):
        image = self.load_image(image_path)
        hsv = self.preprocess_image(image)
        mask = self.create_mask(hsv)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            progress_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(image, [progress_contour], -1, (0, 255, 0), 2)

        result_image_path = os.path.join(RESULT_FOLDER, 'progress_result.png')
        cv2.imwrite(result_image_path, image)

        progress_percentage, crowns_donated, crowns_remaining = self.analyze_progress(image_path)

        if progress_percentage is not None:
            plt.figure(figsize=(5, 2))
            plt.barh(0, progress_percentage, color='orange', height=0.4)
            plt.barh(0, 100, color='lightgray', alpha=0.3, height=0.4)
            plt.xlim(0, 100)
            plt.title(f'Progress: {progress_percentage:.1f}%\n'
                      f'Donated: {crowns_donated:,} crowns\n'
                      f'Remaining: {crowns_remaining:,} crowns')
            plt.axis('off')

            progress_bar_path = os.path.join(RESULT_FOLDER, 'progress_chart.png')
            plt.savefig(progress_bar_path)
            plt.close()
        else:
            progress_bar_path = None

        return result_image_path, progress_bar_path

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        upgrade_type = request.form["upgrade"]
        level = int(request.form["level"])

        goal_crowns = upgrade_requirements[upgrade_type][level]
        analyzer = ProgressBarAnalyzer(goal_crowns=goal_crowns)

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)

        progress_percentage, crowns_donated, crowns_remaining = analyzer.analyze_progress(file_path)
        result_image, progress_bar = analyzer.visualize_results(file_path)

        if progress_percentage is None:
            return "No progress bar detected!"

        return render_template("result.html",
                               progress=progress_percentage,
                               donated=crowns_donated,
                               remaining=crowns_remaining,
                               result_image=result_image,
                               progress_bar=progress_bar)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
