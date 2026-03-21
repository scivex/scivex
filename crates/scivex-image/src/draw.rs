use scivex_core::Scalar;

use crate::image::Image;

/// Draw a line using Bresenham's algorithm (mutates in-place).
///
/// # Examples
///
/// ```
/// # use scivex_image::{Image, PixelFormat};
/// # use scivex_image::draw::draw_line;
/// let mut img = Image::<u8>::new(5, 5, PixelFormat::Gray).unwrap();
/// draw_line(&mut img, 0, 2, 4, 2, &[255]);
/// assert_eq!(img.get_pixel(2, 2).unwrap(), vec![255]);
/// ```
#[allow(clippy::cast_possible_wrap)]
pub fn draw_line<T: Scalar>(
    img: &mut Image<T>,
    mut x0: isize,
    mut y0: isize,
    x1: isize,
    y1: isize,
    color: &[T],
) {
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx: isize = if x0 < x1 { 1 } else { -1 };
    let sy: isize = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    let (w, h) = img.dimensions();
    let wi = w as isize;
    let hi = h as isize;

    loop {
        if x0 >= 0 && x0 < wi && y0 >= 0 && y0 < hi {
            let _ = img.set_pixel(y0 as usize, x0 as usize, color);
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

/// Draw a rectangle outline.
///
/// # Examples
///
/// ```
/// # use scivex_image::prelude::*;
/// # use scivex_image::draw;
/// let mut img = Image::<u8>::new(5, 5, PixelFormat::Gray).unwrap();
/// draw::draw_rect(&mut img, 1, 1, 3, 3, &[200]);
/// assert_eq!(img.get_pixel(1, 1).unwrap(), vec![200]);
/// assert_eq!(img.get_pixel(2, 2).unwrap(), vec![0]); // interior empty
/// ```
#[allow(clippy::cast_possible_wrap)]
pub fn draw_rect<T: Scalar>(
    img: &mut Image<T>,
    x: isize,
    y: isize,
    w: usize,
    h: usize,
    color: &[T],
) {
    let x2 = x + w as isize - 1;
    let y2 = y + h as isize - 1;
    draw_line(img, x, y, x2, y, color); // top
    draw_line(img, x, y2, x2, y2, color); // bottom
    draw_line(img, x, y, x, y2, color); // left
    draw_line(img, x2, y, x2, y2, color); // right
}

/// Fill a rectangle.
///
/// # Examples
///
/// ```
/// # use scivex_image::prelude::*;
/// # use scivex_image::draw;
/// let mut img = Image::<u8>::new(5, 5, PixelFormat::Gray).unwrap();
/// draw::fill_rect(&mut img, 1, 1, 3, 2, &[100]);
/// assert_eq!(img.get_pixel(1, 2).unwrap(), vec![100]);
/// ```
pub fn fill_rect<T: Scalar>(
    img: &mut Image<T>,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    color: &[T],
) {
    let (img_w, img_h) = img.dimensions();
    let x_end = (x + w).min(img_w);
    let y_end = (y + h).min(img_h);

    for row in y..y_end {
        for col in x..x_end {
            let _ = img.set_pixel(row, col, color);
        }
    }
}

/// Draw a circle outline using Bresenham's circle algorithm.
///
/// # Examples
///
/// ```
/// # use scivex_image::prelude::*;
/// # use scivex_image::draw;
/// let mut img = Image::<u8>::new(11, 11, PixelFormat::Gray).unwrap();
/// draw::draw_circle(&mut img, 5, 5, 4, &[255]);
/// assert_eq!(img.get_pixel(1, 5).unwrap(), vec![255]); // top of circle
/// ```
#[allow(clippy::cast_possible_wrap)]
pub fn draw_circle<T: Scalar>(
    img: &mut Image<T>,
    cx: isize,
    cy: isize,
    radius: usize,
    color: &[T],
) {
    let r = radius as isize;
    let mut x = r;
    let mut y: isize = 0;
    let mut err = 1 - r;

    let (w, h) = img.dimensions();
    let wi = w as isize;
    let hi = h as isize;

    while x >= y {
        for &(px, py) in &[
            (cx + x, cy + y),
            (cx - x, cy + y),
            (cx + x, cy - y),
            (cx - x, cy - y),
            (cx + y, cy + x),
            (cx - y, cy + x),
            (cx + y, cy - x),
            (cx - y, cy - x),
        ] {
            if px >= 0 && px < wi && py >= 0 && py < hi {
                let _ = img.set_pixel(py as usize, px as usize, color);
            }
        }

        y += 1;
        if err <= 0 {
            err += 2 * y + 1;
        } else {
            x -= 1;
            err += 2 * (y - x) + 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::PixelFormat;

    #[test]
    fn test_draw_line_horizontal() {
        let mut img = Image::<u8>::new(5, 5, PixelFormat::Gray).unwrap();
        draw_line(&mut img, 0, 2, 4, 2, &[255]);
        for x in 0..5 {
            assert_eq!(img.get_pixel(2, x).unwrap(), vec![255]);
        }
        // Pixels not on line should be 0
        assert_eq!(img.get_pixel(0, 0).unwrap(), vec![0]);
    }

    #[test]
    fn test_fill_rect() {
        let mut img = Image::<u8>::new(5, 5, PixelFormat::Gray).unwrap();
        fill_rect(&mut img, 1, 1, 3, 2, &[100]);
        for row in 1..3 {
            for col in 1..4 {
                assert_eq!(img.get_pixel(row, col).unwrap(), vec![100]);
            }
        }
        assert_eq!(img.get_pixel(0, 0).unwrap(), vec![0]);
    }

    #[test]
    fn test_draw_rect() {
        let mut img = Image::<u8>::new(5, 5, PixelFormat::Gray).unwrap();
        draw_rect(&mut img, 1, 1, 3, 3, &[200]);
        assert_eq!(img.get_pixel(1, 1).unwrap(), vec![200]);
        assert_eq!(img.get_pixel(3, 3).unwrap(), vec![200]);
        assert_eq!(img.get_pixel(2, 2).unwrap(), vec![0]);
    }

    #[test]
    fn test_draw_circle() {
        let mut img = Image::<u8>::new(11, 11, PixelFormat::Gray).unwrap();
        draw_circle(&mut img, 5, 5, 4, &[255]);
        assert_eq!(img.get_pixel(1, 5).unwrap(), vec![255]);
        assert_eq!(img.get_pixel(5, 5).unwrap(), vec![0]);
    }
}
