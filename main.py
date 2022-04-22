import cv2
import numpy as np


def add_subtitle(clip, text, height):
    cv2.putText(clip, text, (0, height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2, cv2.LINE_AA)

# helper function to change what you do based on video seconds


def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper


def main(input_video_file: str, output_video_file: str) -> None:
    # OpenCV video objects to work with)
    cap = cv2.VideoCapture(input_video_file)
    fps = int(cap.get(5))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # saving output video as .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_file, fourcc, fps,
                          (frame_width, frame_height))

    # while loop where the real work happens
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break

            if between(cap, 0, 1000):
                add_subtitle(frame, "Original", frame_height)

            elif between(cap, 1000, 2000):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                add_subtitle(frame, "Grayscale", frame_height)

            elif between(cap, 2000, 3000):
                add_subtitle(frame, "Original", frame_height)

            elif between(cap, 3000, 4000):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                add_subtitle(frame, "Grayscale", frame_height)

            elif between(cap, 4000, 6000):
                frame = cv2.GaussianBlur(frame, (5, 5), 5)
                add_subtitle(
                    frame, "Gaussian kernel size 5, makes blurry", frame_height)

            elif between(cap, 6000, 8000):
                frame = cv2.GaussianBlur(frame, (15, 15), 5)
                add_subtitle(
                    frame, "Gaussian kernel size 15, makes more blurry", frame_height)

            elif between(cap, 8000, 10000):
                frame = cv2.bilateralFilter(frame, 9, 75, 75)
                add_subtitle(
                    frame, "Bilateral size 9, blurs the image, preserves edges", frame_height)

            elif between(cap, 10000, 12000):
                frame = cv2.bilateralFilter(frame, 18, 75, 75)
                add_subtitle(
                    frame, "Bilateral size 18, blurs the image more, preserves edges", frame_height)

            elif between(cap, 12000, 13000):
                add_subtitle(frame, "Original", frame_height)

            elif between(cap, 13000, 15000):
                frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_orange = np.array([0, 100, 100])  # H-10, 100, 100
                upper_orange = np.array([19, 255, 255])  # H+10, 255, 255
                frame = cv2.inRange(frame_hsv, lower_orange, upper_orange)
                frame = np.stack([frame] * 3, axis=-1)
                add_subtitle(frame, "Detect the object", frame_height)

            elif between(cap, 15000, 17500):
                frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_orange = np.array([0, 100, 100])  # H-10, 100, 100
                upper_orange = np.array([19, 255, 255])  # H+10, 255, 255
                frame = cv2.inRange(frame_hsv, lower_orange, upper_orange)
                frame = cv2.morphologyEx(
                    frame, cv2.MORPH_CLOSE, np.ones((5, 5)))
                frame = np.stack([frame] * 3, axis=-1)
                add_subtitle(
                    frame, "Closing operation to remove reflection line", frame_height)

            elif between(cap, 17500, 20100):
                frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_orange = np.array([0, 100, 100])  # H-10, 100, 100
                upper_orange = np.array([19, 255, 255])  # H+10, 255, 255
                frame = cv2.inRange(frame_hsv, lower_orange, upper_orange)
                frame = cv2.morphologyEx(
                    frame, cv2.MORPH_CLOSE, np.ones((9, 9)))
                frame = cv2.erode(frame, np.ones((3, 3)), iterations=1)
                frame = np.stack([frame] * 3, axis=-1)
                add_subtitle(
                    frame, "Erosion", frame_height)

            elif between(cap, 20100, 21000):
                add_subtitle(frame, "Original", frame_height)

            elif between(cap, 21000, 22500):
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                grad_y = cv2.Sobel(frame, cv2.CV_8U, 0, 1,
                                   ksize=3)  # horizontal
                frame = cv2.convertScaleAbs(grad_y)
                frame = np.stack([frame] * 3, axis=-1).astype("uint8")

                # MAKE DIFFERENT COLOR
                add_subtitle(
                    frame, "Horizontal Sobel ksize=3", frame_height)

            elif between(cap, 22500, 24000):
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                grad_y = cv2.Sobel(frame, cv2.CV_8U, 0, 1,
                                   ksize=5)  # horizontal
                frame = cv2.convertScaleAbs(grad_y)
                frame = np.stack([frame] * 3, axis=-1).astype("uint8")

                add_subtitle(
                    frame, "Horizontal Sobel ksize=5", frame_height)

            elif between(cap, 24000, 25000):
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                grad_x = cv2.Sobel(frame, cv2.CV_8U, 1,
                                   0, ksize=3)  # vertical
                frame = cv2.convertScaleAbs(grad_x)
                frame = np.stack([frame] * 3, axis=-1).astype("uint8")

                add_subtitle(
                    frame, "Vertical Sobel ksize=3", frame_height)

            elif between(cap, 25000, 27100):
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                grad_x = cv2.Sobel(frame, cv2.CV_8U, 1, 0, ksize=3)
                grad_y = cv2.Sobel(frame, cv2.CV_8U, 0, 1, ksize=3)

                abs_grad_x = cv2.convertScaleAbs(grad_x)
                abs_grad_y = cv2.convertScaleAbs(grad_y)

                frame = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
                frame = np.stack([frame] * 3, axis=-1).astype("uint8")

                add_subtitle(
                    frame, "All edges", frame_height)

            elif between(cap, 27100, 29500):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rows = gray.shape[0]
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,
                                           rows / 8, param1=300, param2=15, minRadius=1, maxRadius=100)

                if circles is not None:
                    circles = np.uint16(np.around(circles))

                    for i in circles[0, :]:
                        center = (i[0], i[1])
                        # circle center
                        cv2.circle(frame, center, 1, (77, 137, 255), 3)
                        # circle outline
                        radius = i[2]
                        cv2.circle(frame, center, radius, (255, 0, 255), 2)

                add_subtitle(
                    frame, "All circles with Hough", frame_height)

            elif between(cap, 29500, 32000):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rows = gray.shape[0]
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,
                                           rows / 8, param1=300, param2=15, minRadius=80, maxRadius=100)

                if circles is not None:
                    circles = np.uint16(np.around(circles))

                    for i in circles[0, :]:
                        center = (i[0], i[1])
                        # circle center
                        cv2.circle(frame, center, 1, (77, 137, 255), 3)
                        # circle outline
                        radius = i[2]
                        cv2.circle(frame, center, radius, (255, 0, 255), 2)

                add_subtitle(
                    frame, "Only big circles with Hough", frame_height)

            elif between(cap, 32000, 35200):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rows = gray.shape[0]
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,
                                           rows / 8, param1=300, param2=15, minRadius=1, maxRadius=60)

                if circles is not None:
                    circles = np.uint16(np.around(circles))

                    for i in circles[0, :]:
                        center = (i[0], i[1])
                        # circle center
                        cv2.circle(frame, center, 1, (77, 137, 255), 3)
                        # circle outline
                        radius = i[2]
                        cv2.circle(frame, center, radius, (255, 0, 255), 2)

                add_subtitle(
                    frame, "Only small circles with Hough", frame_height)

            elif between(cap, 35200, 36500):
                add_subtitle(frame, "Original", frame_height)

            elif between(cap, 36200, 39000):
                template = cv2.imread('black.png')
                h, w = template.shape[:2]
                template = cv2.resize(template, (int(w * 0.6), int(h * 0.6)))
                h, w = template.shape[:2]
                method = eval("cv2.TM_CCOEFF")
                original_shape = frame.shape[:2]
                res = cv2.matchTemplate(frame, template, method)
                res = res - res.min()
                res = res / res.max()
                res = (res * 255).astype("uint8")
                frame = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
                frame = cv2.resize(frame, original_shape[::-1])

                add_subtitle(frame, "Template matching", frame_height)

            elif between(cap, 39000, 40000):
                add_subtitle(frame, "Original", frame_height)

            elif between(cap, 40000, 40100):
                replace_image = frame  # live image to replace with
                replace_image[:, :, :] = replace_image[:, :, :] + 3

                add_subtitle(frame, "Original", frame_height)

            elif between(cap, 40100, 40700):
                add_subtitle(frame, "Original", frame_height)

            elif between(cap, 40700, 43300):
                kernel = np.ones((3, 3), np.uint8)
                frame_hsv = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2HSV)  # BGR to HSV
                lb = np.array([90, 50, 38])  # blue mask
                ub = np.array([110, 255, 255])
                mask = cv2.inRange(frame_hsv, lb, ub)  # Create Mask
                opening = cv2.morphologyEx(
                    mask, cv2.MORPH_CLOSE, kernel)  # Morphology
                contours, _ = cv2.findContours(opening, cv2.RETR_TREE,  # Find contours
                                               cv2.CHAIN_APPROX_NONE)
                for cnt in contours:
                    (x, y, w, h) = cv2.boundingRect(cnt)
                    if w > 2 and h > 20:
                        cv2.rectangle(frame, (x, y), (x+w, y+h),
                                      (42, 255, 100), 4)
                add_subtitle(frame, "Object Tracking", frame_height)

            elif between(cap, 43300, 48000):
                kernel = np.ones((3, 3), np.uint8)
                frame_hsv = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2HSV)  # BGR to HSV
                lb = np.array([90, 50, 38])
                ub = np.array([110, 255, 255])
                mask = cv2.inRange(frame_hsv, lb, ub)  # Create a blue mask
                opening = cv2.morphologyEx(
                    mask, cv2.MORPH_CLOSE, kernel)  # Morphology
                dilate = cv2.dilate(opening, kernel, iterations=2)
                midResult1 = cv2.bitwise_and(
                    replace_image, replace_image, mask=dilate)
                invert = cv2.bitwise_not(
                    dilate, dilate, mask=None)  # Invert the mask
                midResult2 = cv2.bitwise_and(frame, frame, mask=invert)

                frame = midResult1 + midResult2
                add_subtitle(
                    frame, "Make the pen invisible", frame_height)

            elif between(cap, 48000, 54300):
                add_subtitle(
                    frame, "Make the pen visible", frame_height)

            elif between(cap, 54300, 58300):
                kernel = np.ones((3, 3), np.uint8)
                frame_hsv = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2HSV)  # BGR to HSV
                lb = np.array([90, 50, 38])
                ub = np.array([110, 255, 255])
                mask = cv2.inRange(frame_hsv, lb, ub)  # Create a blue mask
                opening = cv2.morphologyEx(
                    mask, cv2.MORPH_CLOSE, kernel)  # Morphology
                dilate = cv2.dilate(opening, kernel, iterations=2)
                midResult1 = cv2.bitwise_and(
                    replace_image, replace_image, mask=dilate)
                invert = cv2.bitwise_not(
                    dilate, dilate, mask=None)  # Invert the mask
                midResult2 = cv2.bitwise_and(frame, frame, mask=invert)
                frame = midResult1 + midResult2
                add_subtitle(
                    frame, "Make the pen invisible", frame_height)

            elif between(cap, 58300, 60000):
                add_subtitle(
                    frame, "Original", frame_height)

                # write frame that you processed to output
            out.write(frame)
            cv2.imshow('Video', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and writing object
    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main("input.mp4", "output.mp4")
