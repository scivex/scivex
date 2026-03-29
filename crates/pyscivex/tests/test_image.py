"""Tests for pyscivex image processing — image submodule."""

import pyscivex as sv


# ===========================================================================
# IMAGE CREATION & PROPERTIES
# ===========================================================================


class TestImageCreation:
    def test_zeros(self):
        img = sv.image.Image.zeros(10, 10, 3)
        assert img.width == 10
        assert img.height == 10
        assert img.channels == 3

    def test_from_data(self):
        data = list(range(12))  # 2x2 RGB
        img = sv.image.Image(data, 2, 2, 3)
        assert img.width == 2
        assert img.height == 2
        assert img.channels == 3

    def test_grayscale(self):
        data = [128] * 9  # 3x3 gray
        img = sv.image.Image(data, 3, 3, 1)
        assert img.channels == 1
        assert img.format == "gray"

    def test_dimensions(self):
        img = sv.image.Image.zeros(5, 10, 3)
        assert img.dimensions() == (5, 10)

    def test_shape(self):
        img = sv.image.Image.zeros(5, 10, 3)
        assert img.shape() == [10, 5, 3]  # [height, width, channels]

    def test_repr(self):
        img = sv.image.Image.zeros(5, 10, 3)
        r = repr(img)
        assert "Image" in r
        assert "5" in r
        assert "10" in r

    def test_to_list(self):
        data = [100, 150, 200] * 4  # 2x2 RGB
        img = sv.image.Image(data, 2, 2, 3)
        out = img.to_list()
        assert len(out) == 12
        assert out[0] == 100


# ===========================================================================
# PIXEL OPERATIONS
# ===========================================================================


class TestPixelOps:
    def test_get_pixel(self):
        data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
        img = sv.image.Image(data, 2, 2, 3)
        px = img.get_pixel(0, 0)
        assert list(px) == [10, 20, 30]

    def test_set_pixel(self):
        img = sv.image.Image.zeros(3, 3, 3)
        img.set_pixel(1, 1, [255, 0, 0])
        px = img.get_pixel(1, 1)
        assert list(px) == [255, 0, 0]

    def test_to_tensor(self):
        img = sv.image.Image.zeros(4, 4, 3)
        t = img.to_tensor()
        shape = t.shape()
        assert shape == [4, 4, 3]


# ===========================================================================
# COLOR CONVERSIONS
# ===========================================================================


class TestColor:
    def test_to_grayscale(self):
        data = [128, 128, 128] * 4  # 2x2 uniform gray RGB
        img = sv.image.Image(data, 2, 2, 3)
        gray = img.to_grayscale()
        assert gray.channels == 1

    def test_to_rgb(self):
        data = [128] * 4  # 2x2 gray
        img = sv.image.Image(data, 2, 2, 1)
        rgb = img.to_rgb()
        assert rgb.channels == 3

    def test_invert(self):
        data = [100, 150, 200] * 4
        img = sv.image.Image(data, 2, 2, 3)
        inv = img.invert()
        px = list(inv.get_pixel(0, 0))
        assert px[0] == 155  # 255 - 100
        assert px[1] == 105  # 255 - 150

    def test_rgb_hsv_roundtrip(self):
        data = [100, 150, 200] * 4
        img = sv.image.Image(data, 2, 2, 3)
        hsv = sv.image.rgb_to_hsv(img)
        assert hsv.channels == 3
        rgb_back = sv.image.hsv_to_rgb(hsv)
        assert rgb_back.channels == 3


# ===========================================================================
# TRANSFORMS
# ===========================================================================


class TestTransforms:
    def test_resize_nearest(self):
        img = sv.image.Image.zeros(4, 4, 3)
        resized = img.resize(8, 8, "nearest")
        assert resized.width == 8
        assert resized.height == 8

    def test_crop(self):
        img = sv.image.Image.zeros(10, 10, 3)
        cropped = img.crop(2, 2, 5, 5)
        assert cropped.width == 5
        assert cropped.height == 5

    def test_flip_horizontal(self):
        data = [255, 0, 0, 0, 255, 0]  # 2x1: red, green
        img = sv.image.Image(data, 2, 1, 3)
        flipped = img.flip_horizontal()
        assert list(flipped.get_pixel(0, 0)) == [0, 255, 0]
        assert list(flipped.get_pixel(0, 1)) == [255, 0, 0]

    def test_flip_vertical(self):
        img = sv.image.Image.zeros(4, 4, 3)
        flipped = img.flip_vertical()
        assert flipped.width == 4
        assert flipped.height == 4

    def test_rotate90(self):
        img = sv.image.Image.zeros(4, 6, 3)
        rotated = img.rotate90()
        assert rotated.width == 6
        assert rotated.height == 4

    def test_rotate180(self):
        img = sv.image.Image.zeros(4, 6, 3)
        rotated = img.rotate180()
        assert rotated.width == 4
        assert rotated.height == 6

    def test_rotate270(self):
        img = sv.image.Image.zeros(4, 6, 3)
        rotated = img.rotate270()
        assert rotated.width == 6
        assert rotated.height == 4

    def test_pad(self):
        img = sv.image.Image.zeros(4, 4, 3)
        padded = img.pad(1, 1, 1, 1, 0)
        assert padded.width == 6
        assert padded.height == 6


# ===========================================================================
# FILTERS
# ===========================================================================


class TestFilters:
    def test_gaussian_blur(self):
        img = sv.image.Image.zeros(10, 10, 1)
        blurred = sv.image.gaussian_blur(img, 1.0)
        assert blurred.width == 10

    def test_box_blur(self):
        img = sv.image.Image.zeros(10, 10, 1)
        blurred = sv.image.box_blur(img, 1)
        assert blurred.width == 10

    def test_sharpen(self):
        img = sv.image.Image.zeros(10, 10, 1)
        sharp = sv.image.sharpen(img)
        assert sharp.width == 10

    def test_sobel(self):
        img = sv.image.Image.zeros(10, 10, 1)
        edges = sv.image.sobel(img)
        assert edges.width == 10

    def test_sobel_xy(self):
        img = sv.image.Image.zeros(10, 10, 1)
        sx = sv.image.sobel_x(img)
        sy = sv.image.sobel_y(img)
        assert sx.width == 10
        assert sy.width == 10

    def test_median_filter(self):
        img = sv.image.Image.zeros(10, 10, 1)
        filtered = sv.image.median_filter(img, 1)
        assert filtered.width == 10


# ===========================================================================
# MORPHOLOGY
# ===========================================================================


class TestMorphology:
    def test_erode(self):
        data = [255] * 100  # 10x10 white gray
        img = sv.image.Image(data, 10, 10, 1)
        eroded = img.erode("rect", 3)
        assert eroded.width == 10

    def test_dilate(self):
        data = [0] * 100
        img = sv.image.Image(data, 10, 10, 1)
        dilated = img.dilate("rect", 3)
        assert dilated.width == 10

    def test_opening(self):
        data = [128] * 100
        img = sv.image.Image(data, 10, 10, 1)
        opened = img.opening("rect", 3)
        assert opened.width == 10

    def test_closing(self):
        data = [128] * 100
        img = sv.image.Image(data, 10, 10, 1)
        closed = img.closing("rect", 3)
        assert closed.width == 10


# ===========================================================================
# DRAWING
# ===========================================================================


class TestDrawing:
    def test_draw_line(self):
        img = sv.image.Image.zeros(10, 10, 3)
        img.draw_line(0, 0, 9, 9, [255, 0, 0])
        # Check that at least the start pixel was set
        px = img.get_pixel(0, 0)
        assert list(px) == [255, 0, 0]

    def test_draw_rect(self):
        img = sv.image.Image.zeros(10, 10, 3)
        img.draw_rect(2, 2, 5, 5, [0, 255, 0])
        assert img.width == 10

    def test_fill_rect(self):
        img = sv.image.Image.zeros(10, 10, 1)
        img.fill_rect(1, 1, 3, 3, [255])
        px = img.get_pixel(2, 2)
        assert list(px) == [255]

    def test_draw_circle(self):
        img = sv.image.Image.zeros(20, 20, 3)
        img.draw_circle(10, 10, 5, [0, 0, 255])
        assert img.width == 20


# ===========================================================================
# HISTOGRAM
# ===========================================================================


class TestHistogram:
    def test_histogram(self):
        data = [100] * 9  # 3x3 gray, all 100
        img = sv.image.Image(data, 3, 3, 1)
        h = img.histogram()
        shape = h.shape()
        assert shape[0] == 1  # 1 channel
        assert shape[1] == 256

    def test_equalize(self):
        data = list(range(100))  # 10x10 gray gradient
        img = sv.image.Image(data, 10, 10, 1)
        eq = img.equalize()
        assert eq.width == 10


# ===========================================================================
# FEATURES
# ===========================================================================


class TestFeatures:
    def test_harris_corners(self):
        # Create a simple image with a corner pattern
        data = [0] * (20 * 20)
        for r in range(5, 15):
            for c in range(5, 15):
                data[r * 20 + c] = 255
        img = sv.image.Image(data, 20, 20, 1)
        corners = sv.image.harris_corners(img, k=0.04, threshold=0.0, block_size=3)
        # Returns list of (row, col, response) tuples
        assert isinstance(corners, list)

    def test_fast_corners(self):
        # Create checkerboard pattern
        data = [0] * (20 * 20)
        for r in range(20):
            for c in range(20):
                if (r + c) % 2 == 0:
                    data[r * 20 + c] = 255
        img = sv.image.Image(data, 20, 20, 1)
        corners = sv.image.fast_corners(img, threshold=20, nonmax=True)
        assert isinstance(corners, list)


# ===========================================================================
# ORB FEATURES
# ===========================================================================


class TestOrb:
    def test_orb_detect(self):
        # Create a non-trivial image
        data = [0] * (30 * 30)
        for r in range(10, 20):
            for c in range(10, 20):
                data[r * 30 + c] = 255
        img = sv.image.Image(data, 30, 30, 1)
        features = sv.image.orb_features(img, n_features=10, fast_threshold=10)
        assert isinstance(features, list)


# ===========================================================================
# HOUGH TRANSFORMS
# ===========================================================================


class TestHough:
    def test_hough_lines(self):
        # Create a horizontal line
        data = [0] * (20 * 20)
        for c in range(20):
            data[10 * 20 + c] = 255
        img = sv.image.Image(data, 20, 20, 1)
        lines = sv.image.hough_lines(img, threshold=5)
        assert isinstance(lines, list)

    def test_hough_circles(self):
        data = [0] * (30 * 30)
        img = sv.image.Image(data, 30, 30, 1)
        circles = sv.image.hough_circles(img, min_radius=3, max_radius=10, threshold=5)
        assert isinstance(circles, list)


# ===========================================================================
# CONTOURS
# ===========================================================================


class TestContours:
    def test_find_contours(self):
        data = [0] * (10 * 10)
        for r in range(3, 7):
            for c in range(3, 7):
                data[r * 10 + c] = 255
        img = sv.image.Image(data, 10, 10, 1)
        contours = sv.image.find_contours(img, threshold=128)
        assert isinstance(contours, list)

    def test_contour_area(self):
        points = [(0, 0), (0, 4), (4, 4), (4, 0)]
        area = sv.image.contour_area(points)
        assert area > 0

    def test_contour_perimeter(self):
        points = [(0, 0), (0, 4), (4, 4), (4, 0)]
        perim = sv.image.contour_perimeter(points)
        assert perim > 0


# ===========================================================================
# SEGMENTATION
# ===========================================================================


class TestSegmentation:
    def test_connected_components(self):
        data = [0] * (10 * 10)
        # Two separate blobs
        for r in range(0, 3):
            for c in range(0, 3):
                data[r * 10 + c] = 255
        for r in range(6, 9):
            for c in range(6, 9):
                data[r * 10 + c] = 255
        img = sv.image.Image(data, 10, 10, 1)
        labels, count = img.connected_components(128)
        assert count >= 2
        assert len(labels) == 100

    def test_region_growing(self):
        data = [128] * (10 * 10)
        img = sv.image.Image(data, 10, 10, 1)
        result = img.region_growing([(5, 5)], tolerance=50)
        assert len(result) == 100


# ===========================================================================
# OPTICAL FLOW
# ===========================================================================


class TestOpticalFlow:
    def test_lucas_kanade(self):
        # Two frames — second shifted
        data1 = [0] * (10 * 10)
        data2 = [0] * (10 * 10)
        for r in range(3, 6):
            for c in range(3, 6):
                data1[r * 10 + c] = 255
        for r in range(4, 7):
            for c in range(4, 7):
                data2[r * 10 + c] = 255
        img1 = sv.image.Image(data1, 10, 10, 1)
        img2 = sv.image.Image(data2, 10, 10, 1)
        result = sv.image.lucas_kanade(img1, img2, window_size=3)
        assert "flow_x" in result
        assert "flow_y" in result
        assert result["width"] == 10
        assert result["height"] == 10


# ===========================================================================
# AUGMENTATION PIPELINE
# ===========================================================================


class TestAugmentation:
    def test_pipeline_create(self):
        pipeline = sv.image.AugmentPipeline()
        assert "AugmentPipeline" in repr(pipeline)

    def test_pipeline_apply(self):
        img = sv.image.Image.zeros(10, 10, 3)
        pipeline = sv.image.AugmentPipeline()
        pipeline.random_flip_h(0.5)
        pipeline.random_flip_v(0.5)
        out = pipeline.apply(img, seed=42)
        assert out.width == 10
        assert out.height == 10

    def test_pipeline_brightness(self):
        data = [128, 128, 128] * (10 * 10)
        img = sv.image.Image(data, 10, 10, 3)
        pipeline = sv.image.AugmentPipeline()
        pipeline.random_brightness(0.2)
        out = pipeline.apply(img, seed=42)
        assert out.width == 10

    def test_pipeline_noise(self):
        data = [128] * (10 * 10)
        img = sv.image.Image(data, 10, 10, 1)
        pipeline = sv.image.AugmentPipeline()
        pipeline.gaussian_noise(0.1)
        out = pipeline.apply(img, seed=42)
        assert out.width == 10


# ===========================================================================
# LANCZOS RESIZE
# ===========================================================================


class TestLanczos:
    def test_resize_lanczos(self):
        img = sv.image.Image.zeros(8, 8, 1)
        resized = sv.image.resize_lanczos(img, 16, 16, a=3)
        assert resized.width == 16
        assert resized.height == 16


# ===========================================================================
# FILE I/O
# ===========================================================================


class TestIO:
    def test_save_load_bmp(self, tmp_path):
        data = [128, 64, 32] * (5 * 5)
        img = sv.image.Image(data, 5, 5, 3)
        path = str(tmp_path / "test.bmp")
        img.save(path)
        loaded = sv.image.Image.open(path)
        assert loaded.width == 5
        assert loaded.height == 5
        assert loaded.channels == 3

    def test_save_load_ppm(self, tmp_path):
        data = [200, 100, 50] * (4 * 4)
        img = sv.image.Image(data, 4, 4, 3)
        path = str(tmp_path / "test.ppm")
        img.save(path)
        loaded = sv.image.Image.open(path)
        assert loaded.width == 4
        assert loaded.height == 4


# ===========================================================================
# INTEGRATION
# ===========================================================================


class TestIntegration:
    def test_all_accessible(self):
        fns = [
            # Classes
            sv.image.Image,
            sv.image.AugmentPipeline,
            # Color
            sv.image.rgb_to_hsv,
            sv.image.hsv_to_rgb,
            # Filters
            sv.image.gaussian_blur,
            sv.image.box_blur,
            sv.image.sharpen,
            sv.image.sobel,
            sv.image.sobel_x,
            sv.image.sobel_y,
            sv.image.median_filter,
            # Lanczos
            sv.image.resize_lanczos,
            # Features
            sv.image.harris_corners,
            sv.image.fast_corners,
            sv.image.orb_features,
            sv.image.match_features,
            # Hough
            sv.image.hough_lines,
            sv.image.hough_circles,
            # Contours
            sv.image.find_contours,
            sv.image.contour_area,
            sv.image.contour_perimeter,
            # Optical flow
            sv.image.lucas_kanade,
            # Segmentation
            sv.image.watershed,
        ]
        for fn in fns:
            assert fn is not None
