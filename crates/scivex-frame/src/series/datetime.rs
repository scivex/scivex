//! DateTime column type for temporal data.
//!
//! Stores timestamps as nanoseconds since the Unix epoch (1970-01-01T00:00:00Z).
//! Provides date/time component extraction and duration arithmetic.

use std::any::Any;
use std::fmt;

use crate::dtype::DType;
use crate::error::{FrameError, Result};
use crate::series::AnySeries;

// ---------------------------------------------------------------------------
// DateTime — a single timestamp value
// ---------------------------------------------------------------------------

/// A timestamp represented as nanoseconds since the Unix epoch.
///
/// Positive values are after 1970-01-01, negative values are before.
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DateTime {
    /// Nanoseconds since 1970-01-01T00:00:00 UTC.
    nanos: i64,
}

/// Number of nanoseconds in one second.
const NANOS_PER_SEC: i64 = 1_000_000_000;
/// Number of nanoseconds in one millisecond.
const NANOS_PER_MILLI: i64 = 1_000_000;
/// Number of nanoseconds in one microsecond.
const NANOS_PER_MICRO: i64 = 1_000;
/// Number of nanoseconds in one minute.
const NANOS_PER_MIN: i64 = 60 * NANOS_PER_SEC;
/// Number of nanoseconds in one hour.
const NANOS_PER_HOUR: i64 = 60 * NANOS_PER_MIN;
/// Number of nanoseconds in one day.
const NANOS_PER_DAY: i64 = 24 * NANOS_PER_HOUR;

/// Days in each month for non-leap years.
const DAYS_IN_MONTH: [i32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

/// Whether a year is a leap year.
fn is_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Days in a given month (1-indexed) and year.
fn days_in_month(year: i32, month: u32) -> u32 {
    if month == 2 && is_leap_year(year) {
        29
    } else {
        DAYS_IN_MONTH[(month - 1) as usize] as u32
    }
}

/// Convert a civil date to days since Unix epoch.
fn date_to_epoch_days(year: i32, month: u32, day: u32) -> i64 {
    // Algorithm from Howard Hinnant's date library (public domain).
    let y = if month <= 2 {
        i64::from(year) - 1
    } else {
        i64::from(year)
    };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let m = if month <= 2 {
        i64::from(month) + 9
    } else {
        i64::from(month) - 3
    };
    let doy = (153 * m + 2) / 5 + i64::from(day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Convert days since Unix epoch to civil date (year, month, day).
fn epoch_days_to_date(days: i64) -> (i32, u32, u32) {
    // Howard Hinnant's algorithm (inverse of the above).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    #[allow(clippy::cast_possible_truncation)]
    (y as i32, m as u32, d as u32)
}

impl DateTime {
    /// Create from nanoseconds since the Unix epoch.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTime;
    /// let dt = DateTime::from_nanos(1_000_000_000);
    /// assert_eq!(dt.timestamp(), 1);
    /// ```
    #[inline]
    pub fn from_nanos(nanos: i64) -> Self {
        Self { nanos }
    }

    /// Create from seconds since the Unix epoch.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTime;
    /// let dt = DateTime::from_secs(86400);
    /// assert_eq!(dt.day(), 2); // 1970-01-02
    /// ```
    #[inline]
    pub fn from_secs(secs: i64) -> Self {
        Self {
            nanos: secs * NANOS_PER_SEC,
        }
    }

    /// Create from milliseconds since the Unix epoch.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTime;
    /// let dt = DateTime::from_millis(86_400_000);
    /// assert_eq!(dt.day(), 2);
    /// ```
    #[inline]
    pub fn from_millis(millis: i64) -> Self {
        Self {
            nanos: millis * NANOS_PER_MILLI,
        }
    }

    /// Create from a civil date and time.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTime;
    /// let dt = DateTime::from_ymd_hms(2024, 3, 15, 14, 30, 0).unwrap();
    /// assert_eq!(dt.year(), 2024);
    /// assert_eq!(dt.hour(), 14);
    /// ```
    pub fn from_ymd_hms(
        year: i32,
        month: u32,
        day: u32,
        hour: u32,
        minute: u32,
        second: u32,
    ) -> Result<Self> {
        if !(1..=12).contains(&month) {
            return Err(FrameError::InvalidArgument {
                reason: "month must be 1..=12",
            });
        }
        if day < 1 || day > days_in_month(year, month) {
            return Err(FrameError::InvalidArgument {
                reason: "day out of range for given month/year",
            });
        }
        if hour >= 24 || minute >= 60 || second >= 60 {
            return Err(FrameError::InvalidArgument {
                reason: "hour/minute/second out of range",
            });
        }
        let days = date_to_epoch_days(year, month, day);
        let time_nanos = i64::from(hour) * NANOS_PER_HOUR
            + i64::from(minute) * NANOS_PER_MIN
            + i64::from(second) * NANOS_PER_SEC;
        Ok(Self {
            nanos: days * NANOS_PER_DAY + time_nanos,
        })
    }

    /// Create from a date only (midnight).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTime;
    /// let dt = DateTime::from_ymd(2024, 12, 25).unwrap();
    /// assert_eq!(dt.month(), 12);
    /// assert_eq!(dt.day(), 25);
    /// ```
    pub fn from_ymd(year: i32, month: u32, day: u32) -> Result<Self> {
        Self::from_ymd_hms(year, month, day, 0, 0, 0)
    }

    /// The Unix epoch (1970-01-01T00:00:00Z).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTime;
    /// let dt = DateTime::epoch();
    /// assert_eq!(dt.year(), 1970);
    /// assert_eq!(dt.timestamp(), 0);
    /// ```
    pub fn epoch() -> Self {
        Self { nanos: 0 }
    }

    /// Nanoseconds since the Unix epoch.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTime;
    /// let dt = DateTime::from_secs(1);
    /// assert_eq!(dt.timestamp_nanos(), 1_000_000_000_i64);
    /// ```
    #[inline]
    pub fn timestamp_nanos(&self) -> i64 {
        self.nanos
    }

    /// Seconds since the Unix epoch (truncated toward zero).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTime;
    /// let dt = DateTime::from_secs(42);
    /// assert_eq!(dt.timestamp(), 42);
    /// ```
    #[inline]
    pub fn timestamp(&self) -> i64 {
        self.nanos.div_euclid(NANOS_PER_SEC)
    }

    /// Milliseconds since the Unix epoch.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTime;
    /// let dt = DateTime::from_millis(5000);
    /// assert_eq!(dt.timestamp_millis(), 5000);
    /// ```
    #[inline]
    pub fn timestamp_millis(&self) -> i64 {
        self.nanos.div_euclid(NANOS_PER_MILLI)
    }

    // -- Component extraction -----------------------------------------------

    /// Total days since epoch (for internal date extraction).
    fn epoch_days(self) -> i64 {
        self.nanos.div_euclid(NANOS_PER_DAY)
    }

    /// Nanoseconds within the day.
    fn day_nanos(self) -> i64 {
        self.nanos.rem_euclid(NANOS_PER_DAY)
    }

    /// Year component.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTime;
    /// let dt = DateTime::from_ymd(2024, 6, 15).unwrap();
    /// assert_eq!(dt.year(), 2024);
    /// ```
    pub fn year(&self) -> i32 {
        epoch_days_to_date(self.epoch_days()).0
    }

    /// Month component (1–12).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTime;
    /// let dt = DateTime::from_ymd(2024, 7, 4).unwrap();
    /// assert_eq!(dt.month(), 7);
    /// ```
    pub fn month(&self) -> u32 {
        epoch_days_to_date(self.epoch_days()).1
    }

    /// Day-of-month component (1–31).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTime;
    /// let dt = DateTime::from_ymd(2024, 7, 4).unwrap();
    /// assert_eq!(dt.day(), 4);
    /// ```
    pub fn day(&self) -> u32 {
        epoch_days_to_date(self.epoch_days()).2
    }

    /// Hour component (0–23).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTime;
    /// let dt = DateTime::from_ymd_hms(2024, 1, 1, 15, 0, 0).unwrap();
    /// assert_eq!(dt.hour(), 15);
    /// ```
    #[allow(clippy::cast_possible_truncation)]
    pub fn hour(&self) -> u32 {
        (self.day_nanos() / NANOS_PER_HOUR) as u32
    }

    /// Minute component (0–59).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTime;
    /// let dt = DateTime::from_ymd_hms(2024, 1, 1, 0, 45, 0).unwrap();
    /// assert_eq!(dt.minute(), 45);
    /// ```
    #[allow(clippy::cast_possible_truncation)]
    pub fn minute(&self) -> u32 {
        ((self.day_nanos() % NANOS_PER_HOUR) / NANOS_PER_MIN) as u32
    }

    /// Second component (0–59).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTime;
    /// let dt = DateTime::from_ymd_hms(2024, 1, 1, 0, 0, 30).unwrap();
    /// assert_eq!(dt.second(), 30);
    /// ```
    #[allow(clippy::cast_possible_truncation)]
    pub fn second(&self) -> u32 {
        ((self.day_nanos() % NANOS_PER_MIN) / NANOS_PER_SEC) as u32
    }

    /// Nanosecond component within the second.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTime;
    /// let dt = DateTime::from_nanos(1_000_000_500);
    /// assert_eq!(dt.nanosecond(), 500);
    /// ```
    #[allow(clippy::cast_possible_truncation)]
    pub fn nanosecond(&self) -> u32 {
        (self.day_nanos() % NANOS_PER_SEC) as u32
    }

    /// Day of week (0 = Monday, 6 = Sunday).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTime;
    /// let dt = DateTime::epoch(); // 1970-01-01 was a Thursday
    /// assert_eq!(dt.weekday(), 3);
    /// ```
    #[allow(clippy::cast_possible_truncation)]
    pub fn weekday(&self) -> u32 {
        // 1970-01-01 was a Thursday (day 3, 0-indexed from Monday).
        ((self.epoch_days() + 3).rem_euclid(7)) as u32
    }

    /// Day of year (1–366).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTime;
    /// let dt = DateTime::from_ymd(2024, 1, 1).unwrap();
    /// assert_eq!(dt.day_of_year(), 1);
    /// ```
    #[allow(clippy::cast_possible_truncation)]
    pub fn day_of_year(&self) -> u32 {
        let (year, _month, _day) = epoch_days_to_date(self.epoch_days());
        let jan1 = date_to_epoch_days(year, 1, 1);
        (self.epoch_days() - jan1) as u32 + 1
    }

    // -- Arithmetic ---------------------------------------------------------

    /// Add a [`Duration`] to this timestamp.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::{DateTime, Duration};
    /// let dt = DateTime::from_ymd(2024, 1, 1).unwrap();
    /// let later = dt.add_duration(Duration::days(31));
    /// assert_eq!(later.month(), 2);
    /// ```
    pub fn add_duration(&self, dur: Duration) -> Self {
        Self {
            nanos: self.nanos + dur.nanos,
        }
    }

    /// Subtract a [`Duration`] from this timestamp.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::{DateTime, Duration};
    /// let dt = DateTime::from_ymd(2024, 2, 1).unwrap();
    /// let earlier = dt.sub_duration(Duration::days(1));
    /// assert_eq!(earlier.month(), 1);
    /// assert_eq!(earlier.day(), 31);
    /// ```
    pub fn sub_duration(&self, dur: Duration) -> Self {
        Self {
            nanos: self.nanos - dur.nanos,
        }
    }

    /// Duration between two timestamps (self - other).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTime;
    /// let a = DateTime::from_ymd(2024, 1, 10).unwrap();
    /// let b = DateTime::from_ymd(2024, 1, 1).unwrap();
    /// assert_eq!(a.duration_since(&b).total_days(), 9);
    /// ```
    pub fn duration_since(&self, other: &DateTime) -> Duration {
        Duration {
            nanos: self.nanos - other.nanos,
        }
    }

    // -- Parsing ------------------------------------------------------------

    /// Parse from ISO 8601 format: `YYYY-MM-DD` or `YYYY-MM-DDTHH:MM:SS`.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTime;
    /// let dt = DateTime::parse("2024-06-15").unwrap();
    /// assert_eq!(dt.year(), 2024);
    /// assert_eq!(dt.month(), 6);
    /// ```
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        // Try YYYY-MM-DDTHH:MM:SS or YYYY-MM-DD HH:MM:SS
        if s.len() >= 19 {
            let date_part = &s[..10];
            let time_part = &s[11..19];
            let (year, month, day) = parse_date_part(date_part)?;
            let (hour, minute, second) = parse_time_part(time_part)?;
            return Self::from_ymd_hms(year, month, day, hour, minute, second);
        }
        // Try YYYY-MM-DD
        if s.len() == 10 {
            let (year, month, day) = parse_date_part(s)?;
            return Self::from_ymd(year, month, day);
        }
        Err(FrameError::InvalidArgument {
            reason: "cannot parse datetime; expected YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS",
        })
    }
}

/// Parse `YYYY-MM-DD` into components.
fn parse_date_part(s: &str) -> Result<(i32, u32, u32)> {
    if s.len() != 10 || s.as_bytes()[4] != b'-' || s.as_bytes()[7] != b'-' {
        return Err(FrameError::InvalidArgument {
            reason: "invalid date format; expected YYYY-MM-DD",
        });
    }
    let year: i32 = s[..4].parse().map_err(|_| FrameError::InvalidArgument {
        reason: "invalid year in date",
    })?;
    let month: u32 = s[5..7].parse().map_err(|_| FrameError::InvalidArgument {
        reason: "invalid month in date",
    })?;
    let day: u32 = s[8..10].parse().map_err(|_| FrameError::InvalidArgument {
        reason: "invalid day in date",
    })?;
    Ok((year, month, day))
}

/// Parse `HH:MM:SS` into components.
fn parse_time_part(s: &str) -> Result<(u32, u32, u32)> {
    if s.len() != 8 || s.as_bytes()[2] != b':' || s.as_bytes()[5] != b':' {
        return Err(FrameError::InvalidArgument {
            reason: "invalid time format; expected HH:MM:SS",
        });
    }
    let hour: u32 = s[..2].parse().map_err(|_| FrameError::InvalidArgument {
        reason: "invalid hour in time",
    })?;
    let minute: u32 = s[3..5].parse().map_err(|_| FrameError::InvalidArgument {
        reason: "invalid minute in time",
    })?;
    let second: u32 = s[6..8].parse().map_err(|_| FrameError::InvalidArgument {
        reason: "invalid second in time",
    })?;
    Ok((hour, minute, second))
}

impl fmt::Display for DateTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (year, month, day) = epoch_days_to_date(self.epoch_days());
        let h = self.hour();
        let m = self.minute();
        let s = self.second();
        if h == 0 && m == 0 && s == 0 && self.nanosecond() == 0 {
            write!(f, "{year:04}-{month:02}-{day:02}")
        } else {
            write!(f, "{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}")
        }
    }
}

// ---------------------------------------------------------------------------
// Duration — a time span
// ---------------------------------------------------------------------------

/// A time span represented as nanoseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Duration {
    nanos: i64,
}

impl Duration {
    /// Create from nanoseconds.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::Duration;
    /// let d = Duration::nanoseconds(1_000_000);
    /// assert_eq!(d.total_nanos(), 1_000_000);
    /// ```
    #[inline]
    pub fn nanoseconds(nanos: i64) -> Self {
        Self { nanos }
    }

    /// Create from microseconds.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::Duration;
    /// let d = Duration::microseconds(1_000_000);
    /// assert_eq!(d.total_seconds(), 1);
    /// ```
    #[inline]
    pub fn microseconds(micros: i64) -> Self {
        Self {
            nanos: micros * NANOS_PER_MICRO,
        }
    }

    /// Create from milliseconds.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::Duration;
    /// let d = Duration::milliseconds(2000);
    /// assert_eq!(d.total_seconds(), 2);
    /// ```
    #[inline]
    pub fn milliseconds(millis: i64) -> Self {
        Self {
            nanos: millis * NANOS_PER_MILLI,
        }
    }

    /// Create from seconds.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::Duration;
    /// let d = Duration::seconds(60);
    /// assert_eq!(d.total_minutes(), 1);
    /// ```
    #[inline]
    pub fn seconds(secs: i64) -> Self {
        Self {
            nanos: secs * NANOS_PER_SEC,
        }
    }

    /// Create from minutes.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::Duration;
    /// let d = Duration::minutes(90);
    /// assert_eq!(d.total_hours(), 1);
    /// ```
    #[inline]
    pub fn minutes(mins: i64) -> Self {
        Self {
            nanos: mins * NANOS_PER_MIN,
        }
    }

    /// Create from hours.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::Duration;
    /// let d = Duration::hours(24);
    /// assert_eq!(d.total_days(), 1);
    /// ```
    #[inline]
    pub fn hours(hours: i64) -> Self {
        Self {
            nanos: hours * NANOS_PER_HOUR,
        }
    }

    /// Create from days.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::Duration;
    /// let d = Duration::days(2);
    /// assert_eq!(d.total_hours(), 48);
    /// ```
    #[inline]
    pub fn days(days: i64) -> Self {
        Self {
            nanos: days * NANOS_PER_DAY,
        }
    }

    /// Zero duration.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::Duration;
    /// let d = Duration::zero();
    /// assert_eq!(d.total_nanos(), 0);
    /// ```
    #[inline]
    pub fn zero() -> Self {
        Self { nanos: 0 }
    }

    /// Total nanoseconds.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::Duration;
    /// let d = Duration::nanoseconds(500);
    /// assert_eq!(d.total_nanos(), 500);
    /// ```
    #[inline]
    pub fn total_nanos(&self) -> i64 {
        self.nanos
    }

    /// Total seconds (truncated).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::Duration;
    /// let d = Duration::seconds(90);
    /// assert_eq!(d.total_seconds(), 90);
    /// ```
    #[inline]
    pub fn total_seconds(&self) -> i64 {
        self.nanos / NANOS_PER_SEC
    }

    /// Total milliseconds (truncated).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::Duration;
    /// let d = Duration::seconds(2);
    /// assert_eq!(d.total_millis(), 2000);
    /// ```
    #[inline]
    pub fn total_millis(&self) -> i64 {
        self.nanos / NANOS_PER_MILLI
    }

    /// Total minutes (truncated).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::Duration;
    /// let d = Duration::minutes(3);
    /// assert_eq!(d.total_minutes(), 3);
    /// ```
    #[inline]
    pub fn total_minutes(&self) -> i64 {
        self.nanos / NANOS_PER_MIN
    }

    /// Total hours (truncated).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::Duration;
    /// let d = Duration::hours(2);
    /// assert_eq!(d.total_hours(), 2);
    /// ```
    #[inline]
    pub fn total_hours(&self) -> i64 {
        self.nanos / NANOS_PER_HOUR
    }

    /// Total days (truncated).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::Duration;
    /// let d = Duration::days(7);
    /// assert_eq!(d.total_days(), 7);
    /// ```
    #[inline]
    pub fn total_days(&self) -> i64 {
        self.nanos / NANOS_PER_DAY
    }

    /// Absolute value of this duration.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::Duration;
    /// let d = Duration::seconds(-90);
    /// assert_eq!(d.abs().total_seconds(), 90);
    /// ```
    pub fn abs(&self) -> Self {
        Self {
            nanos: self.nanos.abs(),
        }
    }
}

impl fmt::Display for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let total_secs = self.nanos.abs() / NANOS_PER_SEC;
        let d = total_secs / 86400;
        let h = (total_secs % 86400) / 3600;
        let m = (total_secs % 3600) / 60;
        let s = total_secs % 60;
        let sign = if self.nanos < 0 { "-" } else { "" };
        if d > 0 {
            write!(f, "{sign}{d}d {h:02}:{m:02}:{s:02}")
        } else {
            write!(f, "{sign}{h:02}:{m:02}:{s:02}")
        }
    }
}

// ---------------------------------------------------------------------------
// DateTimeSeries — a column of DateTime values
// ---------------------------------------------------------------------------

/// A named column of [`DateTime`] values with optional null tracking.
#[derive(Debug, Clone)]
pub struct DateTimeSeries {
    name: String,
    data: Vec<DateTime>,
    null_mask: Option<Vec<bool>>,
}

impl DateTimeSeries {
    /// Create a new datetime series from a vector of `DateTime` values.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::{DateTime, DateTimeSeries};
    /// let dates = vec![DateTime::from_ymd(2024, 1, 1).unwrap()];
    /// let s = DateTimeSeries::new("dt", dates);
    /// assert_eq!(s.len(), 1);
    /// ```
    pub fn new(name: impl Into<String>, data: Vec<DateTime>) -> Self {
        Self {
            name: name.into(),
            data,
            null_mask: None,
        }
    }

    /// Create from epoch-second timestamps.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTimeSeries;
    /// let s = DateTimeSeries::from_timestamps("ts", &[0, 86400]);
    /// assert_eq!(s.len(), 2);
    /// ```
    pub fn from_timestamps(name: impl Into<String>, timestamps: &[i64]) -> Self {
        Self::new(
            name,
            timestamps.iter().map(|&t| DateTime::from_secs(t)).collect(),
        )
    }

    /// Create from epoch-millisecond timestamps.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTimeSeries;
    /// let s = DateTimeSeries::from_millis("ts", &[0, 86_400_000]);
    /// assert_eq!(s.len(), 2);
    /// assert_eq!(s.get(1).unwrap().day(), 2);
    /// ```
    pub fn from_millis(name: impl Into<String>, millis: &[i64]) -> Self {
        Self::new(
            name,
            millis.iter().map(|&m| DateTime::from_millis(m)).collect(),
        )
    }

    /// Parse from string slices in ISO 8601 format.
    ///
    /// Values that fail to parse become null.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTimeSeries;
    /// let s = DateTimeSeries::parse("dates", &["2024-01-01", "invalid"]);
    /// assert!(!s.is_null_at(0));
    /// assert!(s.is_null_at(1));
    /// ```
    pub fn parse(name: impl Into<String>, values: &[&str]) -> Self {
        let mut data = Vec::with_capacity(values.len());
        let mut nulls = Vec::with_capacity(values.len());
        let mut has_nulls = false;
        for &v in values {
            if let Ok(dt) = DateTime::parse(v) {
                data.push(dt);
                nulls.push(false);
            } else {
                data.push(DateTime::epoch());
                nulls.push(true);
                has_nulls = true;
            }
        }
        Self {
            name: name.into(),
            data,
            null_mask: if has_nulls { Some(nulls) } else { None },
        }
    }

    /// Create with explicit null positions.
    ///
    /// `null_mask[i] = true` means element `i` is null.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::{DateTime, DateTimeSeries};
    /// let dates = vec![DateTime::from_ymd(2024, 1, 1).unwrap(), DateTime::epoch()];
    /// let s = DateTimeSeries::with_nulls("dt", dates, vec![false, true]).unwrap();
    /// assert_eq!(s.null_count(), 1);
    /// ```
    pub fn with_nulls(
        name: impl Into<String>,
        data: Vec<DateTime>,
        null_mask: Vec<bool>,
    ) -> Result<Self> {
        if data.len() != null_mask.len() {
            return Err(FrameError::RowCountMismatch {
                expected: data.len(),
                got: null_mask.len(),
            });
        }
        Ok(Self {
            name: name.into(),
            data,
            null_mask: Some(null_mask),
        })
    }

    /// Generate a date range from start to end (inclusive) with a step duration.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::{DateTime, DateTimeSeries, Duration};
    /// let start = DateTime::from_ymd(2024, 1, 1).unwrap();
    /// let end = DateTime::from_ymd(2024, 1, 5).unwrap();
    /// let s = DateTimeSeries::date_range("r", start, end, Duration::days(1)).unwrap();
    /// assert_eq!(s.len(), 5);
    /// ```
    pub fn date_range(
        name: impl Into<String>,
        start: DateTime,
        end: DateTime,
        step: Duration,
    ) -> Result<Self> {
        if step.nanos == 0 {
            return Err(FrameError::InvalidArgument {
                reason: "step duration must be non-zero",
            });
        }
        let mut data = Vec::new();
        let mut current = start;
        if step.nanos > 0 {
            while current.nanos <= end.nanos {
                data.push(current);
                current = current.add_duration(step);
            }
        } else {
            while current.nanos >= end.nanos {
                data.push(current);
                current = current.add_duration(step);
            }
        }
        Ok(Self::new(name, data))
    }

    // -- Accessors ----------------------------------------------------------

    /// Column name.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTimeSeries;
    /// let s = DateTimeSeries::from_timestamps("events", &[0]);
    /// assert_eq!(s.name(), "events");
    /// ```
    #[inline]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Number of elements.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTimeSeries;
    /// let s = DateTimeSeries::from_timestamps("ts", &[0, 1, 2]);
    /// assert_eq!(s.len(), 3);
    /// ```
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the series is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTimeSeries;
    /// let s = DateTimeSeries::from_timestamps("ts", &[]);
    /// assert!(s.is_empty());
    /// ```
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the element at `index`.
    ///
    /// Returns `None` if the index is out of bounds or the element is null.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::{DateTime, DateTimeSeries};
    /// let s = DateTimeSeries::from_timestamps("ts", &[0]);
    /// assert_eq!(s.get(0), Some(&DateTime::epoch()));
    /// assert!(s.get(1).is_none());
    /// ```
    pub fn get(&self, index: usize) -> Option<&DateTime> {
        if index >= self.data.len() || self.is_null_at(index) {
            return None;
        }
        Some(&self.data[index])
    }

    /// Whether the element at `index` is null.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTimeSeries;
    /// let s = DateTimeSeries::parse("dt", &["2024-01-01", "bad"]);
    /// assert!(!s.is_null_at(0));
    /// assert!(s.is_null_at(1));
    /// ```
    #[inline]
    pub fn is_null_at(&self, index: usize) -> bool {
        self.null_mask
            .as_ref()
            .is_some_and(|m| m.get(index).copied().unwrap_or(false))
    }

    /// Number of null entries.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTimeSeries;
    /// let s = DateTimeSeries::parse("dt", &["2024-01-01", "bad", "also-bad"]);
    /// assert_eq!(s.null_count(), 2);
    /// ```
    pub fn null_count(&self) -> usize {
        self.null_mask
            .as_ref()
            .map_or(0, |m| m.iter().filter(|&&v| v).count())
    }

    /// Rename in place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::DateTimeSeries;
    /// let mut s = DateTimeSeries::from_timestamps("old", &[0]);
    /// s.rename("new");
    /// assert_eq!(s.name(), "new");
    /// ```
    pub fn rename(&mut self, name: impl Into<String>) {
        self.name = name.into();
    }

    /// Underlying data as a slice.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::{DateTime, DateTimeSeries};
    /// let s = DateTimeSeries::from_timestamps("ts", &[0]);
    /// assert_eq!(s.as_slice(), &[DateTime::epoch()]);
    /// ```
    pub fn as_slice(&self) -> &[DateTime] {
        &self.data
    }

    // -- Component extraction (vectorized) ----------------------------------

    /// Extract year from each element.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::{DateTime, DateTimeSeries};
    /// let s = DateTimeSeries::new("dt", vec![DateTime::from_ymd(2024, 6, 1).unwrap()]);
    /// assert_eq!(s.year(), vec![Some(2024)]);
    /// ```
    pub fn year(&self) -> Vec<Option<i32>> {
        self.map_component(DateTime::year)
    }

    /// Extract month (1–12) from each element.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::{DateTime, DateTimeSeries};
    /// let s = DateTimeSeries::new("dt", vec![DateTime::from_ymd(2024, 6, 1).unwrap()]);
    /// assert_eq!(s.month(), vec![Some(6)]);
    /// ```
    pub fn month(&self) -> Vec<Option<u32>> {
        self.map_component(DateTime::month)
    }

    /// Extract day-of-month (1–31) from each element.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::{DateTime, DateTimeSeries};
    /// let s = DateTimeSeries::new("dt", vec![DateTime::from_ymd(2024, 6, 15).unwrap()]);
    /// assert_eq!(s.day(), vec![Some(15)]);
    /// ```
    pub fn day(&self) -> Vec<Option<u32>> {
        self.map_component(DateTime::day)
    }

    /// Extract hour (0–23) from each element.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::{DateTime, DateTimeSeries};
    /// let s = DateTimeSeries::new("dt", vec![DateTime::from_ymd_hms(2024, 1, 1, 9, 0, 0).unwrap()]);
    /// assert_eq!(s.hour(), vec![Some(9)]);
    /// ```
    pub fn hour(&self) -> Vec<Option<u32>> {
        self.map_component(DateTime::hour)
    }

    /// Extract minute (0–59) from each element.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::{DateTime, DateTimeSeries};
    /// let s = DateTimeSeries::new("dt", vec![DateTime::from_ymd_hms(2024, 1, 1, 0, 30, 0).unwrap()]);
    /// assert_eq!(s.minute(), vec![Some(30)]);
    /// ```
    pub fn minute(&self) -> Vec<Option<u32>> {
        self.map_component(DateTime::minute)
    }

    /// Extract second (0–59) from each element.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::{DateTime, DateTimeSeries};
    /// let s = DateTimeSeries::new("dt", vec![DateTime::from_ymd_hms(2024, 1, 1, 0, 0, 45).unwrap()]);
    /// assert_eq!(s.second(), vec![Some(45)]);
    /// ```
    pub fn second(&self) -> Vec<Option<u32>> {
        self.map_component(DateTime::second)
    }

    /// Extract weekday (0=Monday, 6=Sunday) from each element.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::{DateTime, DateTimeSeries};
    /// // 1970-01-01 was a Thursday (3)
    /// let s = DateTimeSeries::new("dt", vec![DateTime::epoch()]);
    /// assert_eq!(s.weekday(), vec![Some(3)]);
    /// ```
    pub fn weekday(&self) -> Vec<Option<u32>> {
        self.map_component(DateTime::weekday)
    }

    /// Helper to apply a component extraction function.
    fn map_component<R>(&self, f: impl Fn(&DateTime) -> R) -> Vec<Option<R>> {
        (0..self.data.len())
            .map(|i| {
                if self.is_null_at(i) {
                    None
                } else {
                    Some(f(&self.data[i]))
                }
            })
            .collect()
    }

    // -- Arithmetic ---------------------------------------------------------

    /// Add a duration to every element.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::{DateTime, DateTimeSeries, Duration};
    /// let s = DateTimeSeries::new("dt", vec![DateTime::from_ymd(2024, 1, 1).unwrap()]);
    /// let shifted = s.add_duration(Duration::days(10));
    /// assert_eq!(shifted.get(0).unwrap().day(), 11);
    /// ```
    pub fn add_duration(&self, dur: Duration) -> Self {
        Self {
            name: self.name.clone(),
            data: self.data.iter().map(|dt| dt.add_duration(dur)).collect(),
            null_mask: self.null_mask.clone(),
        }
    }

    /// Subtract a duration from every element.
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::{DateTime, DateTimeSeries, Duration};
    /// let s = DateTimeSeries::new("dt", vec![DateTime::from_ymd(2024, 1, 11).unwrap()]);
    /// let shifted = s.sub_duration(Duration::days(10));
    /// assert_eq!(shifted.get(0).unwrap().day(), 1);
    /// ```
    pub fn sub_duration(&self, dur: Duration) -> Self {
        Self {
            name: self.name.clone(),
            data: self.data.iter().map(|dt| dt.sub_duration(dur)).collect(),
            null_mask: self.null_mask.clone(),
        }
    }

    /// Compute the minimum timestamp (ignoring nulls).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::{DateTime, DateTimeSeries};
    /// let s = DateTimeSeries::new("dt", vec![
    ///     DateTime::from_ymd(2024, 3, 1).unwrap(),
    ///     DateTime::from_ymd(2024, 1, 1).unwrap(),
    /// ]);
    /// assert_eq!(s.min().unwrap().month(), 1);
    /// ```
    pub fn min(&self) -> Option<DateTime> {
        self.data
            .iter()
            .enumerate()
            .filter(|&(i, _)| !self.is_null_at(i))
            .map(|(_, dt)| *dt)
            .min()
    }

    /// Compute the maximum timestamp (ignoring nulls).
    ///
    /// # Examples
    ///
    /// ```
    /// # use scivex_frame::series::datetime::{DateTime, DateTimeSeries};
    /// let s = DateTimeSeries::new("dt", vec![
    ///     DateTime::from_ymd(2024, 1, 1).unwrap(),
    ///     DateTime::from_ymd(2024, 12, 31).unwrap(),
    /// ]);
    /// assert_eq!(s.max().unwrap().month(), 12);
    /// ```
    pub fn max(&self) -> Option<DateTime> {
        self.data
            .iter()
            .enumerate()
            .filter(|&(i, _)| !self.is_null_at(i))
            .map(|(_, dt)| *dt)
            .max()
    }
}

impl fmt::Display for DateTimeSeries {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DateTimeSeries({:?}, len={})",
            self.name,
            self.data.len()
        )
    }
}

// ---------------------------------------------------------------------------
// AnySeries implementation
// ---------------------------------------------------------------------------

impl AnySeries for DateTimeSeries {
    fn name(&self) -> &str {
        &self.name
    }

    fn dtype(&self) -> DType {
        DType::DateTime
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn null_count(&self) -> usize {
        self.null_count()
    }

    fn is_null(&self, index: usize) -> bool {
        self.is_null_at(index)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn AnySeries> {
        Box::new(self.clone())
    }

    fn display_value(&self, index: usize) -> String {
        if self.is_null_at(index) {
            "null".to_string()
        } else if index < self.data.len() {
            self.data[index].to_string()
        } else {
            String::new()
        }
    }

    fn hash_value(&self, index: usize) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        if self.is_null_at(index) {
            u64::MAX.hash(&mut hasher);
        } else if index < self.data.len() {
            self.data[index].timestamp_nanos().hash(&mut hasher);
        }
        hasher.finish()
    }

    fn filter_mask(&self, mask: &[bool]) -> Box<dyn AnySeries> {
        let mut data = Vec::new();
        let mut new_nulls: Option<Vec<bool>> = self.null_mask.as_ref().map(|_| Vec::new());
        for (i, &keep) in mask.iter().enumerate() {
            if keep && i < self.data.len() {
                data.push(self.data[i]);
                if let Some(ref mut nm) = new_nulls {
                    nm.push(
                        self.null_mask
                            .as_ref()
                            .expect("null_mask present when has_nulls is true")[i],
                    );
                }
            }
        }
        Box::new(DateTimeSeries {
            name: self.name.clone(),
            data,
            null_mask: new_nulls,
        })
    }

    fn take_indices(&self, indices: &[usize]) -> Box<dyn AnySeries> {
        let data: Vec<DateTime> = indices.iter().map(|&i| self.data[i]).collect();
        let null_mask = self
            .null_mask
            .as_ref()
            .map(|m| indices.iter().map(|&i| m[i]).collect());
        Box::new(DateTimeSeries {
            name: self.name.clone(),
            data,
            null_mask,
        })
    }

    fn slice(&self, offset: usize, length: usize) -> Box<dyn AnySeries> {
        let end = (offset + length).min(self.data.len());
        let data = self.data[offset..end].to_vec();
        let null_mask = self.null_mask.as_ref().map(|m| m[offset..end].to_vec());
        Box::new(DateTimeSeries {
            name: self.name.clone(),
            data,
            null_mask,
        })
    }

    fn rename_box(&self, name: &str) -> Box<dyn AnySeries> {
        let mut cloned = self.clone();
        cloned.name = name.to_string();
        Box::new(cloned)
    }

    fn drop_nulls(&self) -> Box<dyn AnySeries> {
        if self.null_mask.is_none() {
            return self.clone_box();
        }
        let mask = self
            .null_mask
            .as_ref()
            .expect("null_mask present when has_nulls is true");
        let keep: Vec<bool> = mask.iter().map(|&is_null| !is_null).collect();
        self.filter_mask(&keep)
    }

    fn null_mask_vec(&self) -> Vec<bool> {
        self.null_mask
            .clone()
            .unwrap_or_else(|| vec![false; self.data.len()])
    }

    fn null_series(&self, name: &str, len: usize) -> Box<dyn AnySeries> {
        Box::new(DateTimeSeries {
            name: name.to_string(),
            data: vec![DateTime::epoch(); len],
            null_mask: Some(vec![true; len]),
        })
    }

    fn take_optional(&self, indices: &[Option<usize>]) -> Box<dyn AnySeries> {
        let mut data = Vec::with_capacity(indices.len());
        let mut nulls = Vec::with_capacity(indices.len());
        for opt in indices {
            if let Some(i) = opt {
                data.push(self.data[*i]);
                nulls.push(self.is_null_at(*i));
            } else {
                data.push(DateTime::epoch());
                nulls.push(true);
            }
        }
        let has_nulls = nulls.iter().any(|&v| v);
        Box::new(DateTimeSeries {
            name: self.name.clone(),
            data,
            null_mask: if has_nulls { Some(nulls) } else { None },
        })
    }

    fn compare_at(&self, a: usize, b: usize) -> std::cmp::Ordering {
        let a_null = self.is_null_at(a);
        let b_null = self.is_null_at(b);
        match (a_null, b_null) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => self.data[a].cmp(&self.data[b]),
        }
    }

    fn sort_indices(&self, indices: &mut [usize], ascending: bool) {
        indices.sort_unstable_by(|&a, &b| {
            let cmp = self.compare_at(a, b);
            if ascending { cmp } else { cmp.reverse() }
        });
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_datetime_from_ymd() {
        let dt = DateTime::from_ymd(2024, 3, 15).unwrap();
        assert_eq!(dt.year(), 2024);
        assert_eq!(dt.month(), 3);
        assert_eq!(dt.day(), 15);
        assert_eq!(dt.hour(), 0);
        assert_eq!(dt.minute(), 0);
        assert_eq!(dt.second(), 0);
    }

    #[test]
    fn test_datetime_from_ymd_hms() {
        let dt = DateTime::from_ymd_hms(2024, 12, 25, 14, 30, 45).unwrap();
        assert_eq!(dt.year(), 2024);
        assert_eq!(dt.month(), 12);
        assert_eq!(dt.day(), 25);
        assert_eq!(dt.hour(), 14);
        assert_eq!(dt.minute(), 30);
        assert_eq!(dt.second(), 45);
    }

    #[test]
    fn test_datetime_epoch() {
        let dt = DateTime::epoch();
        assert_eq!(dt.year(), 1970);
        assert_eq!(dt.month(), 1);
        assert_eq!(dt.day(), 1);
        assert_eq!(dt.timestamp(), 0);
    }

    #[test]
    fn test_datetime_from_secs() {
        // 2024-01-01 00:00:00 UTC = 1704067200 seconds
        let dt = DateTime::from_secs(1_704_067_200);
        assert_eq!(dt.year(), 2024);
        assert_eq!(dt.month(), 1);
        assert_eq!(dt.day(), 1);
    }

    #[test]
    fn test_datetime_weekday() {
        // 1970-01-01 was Thursday = 3
        assert_eq!(DateTime::epoch().weekday(), 3);
        // 2024-03-11 (Monday) = 0
        let dt = DateTime::from_ymd(2024, 3, 11).unwrap();
        assert_eq!(dt.weekday(), 0);
    }

    #[test]
    fn test_datetime_parse_date() {
        let dt = DateTime::parse("2024-06-15").unwrap();
        assert_eq!(dt.year(), 2024);
        assert_eq!(dt.month(), 6);
        assert_eq!(dt.day(), 15);
    }

    #[test]
    fn test_datetime_parse_datetime() {
        let dt = DateTime::parse("2024-06-15T10:30:00").unwrap();
        assert_eq!(dt.year(), 2024);
        assert_eq!(dt.month(), 6);
        assert_eq!(dt.day(), 15);
        assert_eq!(dt.hour(), 10);
        assert_eq!(dt.minute(), 30);
    }

    #[test]
    fn test_datetime_parse_space_separator() {
        let dt = DateTime::parse("2024-06-15 10:30:00").unwrap();
        assert_eq!(dt.hour(), 10);
        assert_eq!(dt.minute(), 30);
    }

    #[test]
    fn test_datetime_display() {
        let dt = DateTime::from_ymd(2024, 3, 15).unwrap();
        assert_eq!(dt.to_string(), "2024-03-15");

        let dt = DateTime::from_ymd_hms(2024, 3, 15, 9, 5, 30).unwrap();
        assert_eq!(dt.to_string(), "2024-03-15T09:05:30");
    }

    #[test]
    fn test_datetime_invalid() {
        assert!(DateTime::from_ymd(2024, 0, 1).is_err());
        assert!(DateTime::from_ymd(2024, 13, 1).is_err());
        assert!(DateTime::from_ymd(2024, 2, 30).is_err());
        assert!(DateTime::from_ymd_hms(2024, 1, 1, 24, 0, 0).is_err());
    }

    #[test]
    fn test_datetime_leap_year() {
        // Feb 29 valid on leap year
        assert!(DateTime::from_ymd(2024, 2, 29).is_ok());
        // Feb 29 invalid on non-leap year
        assert!(DateTime::from_ymd(2023, 2, 29).is_err());
    }

    #[test]
    fn test_duration_basics() {
        let d = Duration::days(2);
        assert_eq!(d.total_days(), 2);
        assert_eq!(d.total_hours(), 48);
        assert_eq!(d.total_seconds(), 172_800);

        let d = Duration::hours(3);
        assert_eq!(d.total_minutes(), 180);
    }

    #[test]
    fn test_duration_display() {
        assert_eq!(Duration::hours(2).to_string(), "02:00:00");
        assert_eq!(Duration::days(1).to_string(), "1d 00:00:00");
        assert_eq!(Duration::seconds(-90).to_string(), "-00:01:30");
    }

    #[test]
    fn test_datetime_arithmetic() {
        let dt = DateTime::from_ymd(2024, 1, 1).unwrap();
        let later = dt.add_duration(Duration::days(31));
        assert_eq!(later.year(), 2024);
        assert_eq!(later.month(), 2);
        assert_eq!(later.day(), 1);

        let diff = later.duration_since(&dt);
        assert_eq!(diff.total_days(), 31);
    }

    #[test]
    fn test_datetime_series_new() {
        let dates = vec![
            DateTime::from_ymd(2024, 1, 1).unwrap(),
            DateTime::from_ymd(2024, 6, 15).unwrap(),
            DateTime::from_ymd(2024, 12, 31).unwrap(),
        ];
        let s = DateTimeSeries::new("dates", dates);
        assert_eq!(s.len(), 3);
        assert_eq!(s.null_count(), 0);
        assert_eq!(s.get(0).unwrap().year(), 2024);
    }

    #[test]
    fn test_datetime_series_from_timestamps() {
        let s = DateTimeSeries::from_timestamps("ts", &[0, 86400, 172_800]);
        assert_eq!(s.len(), 3);
        assert_eq!(s.get(0).unwrap().year(), 1970);
        assert_eq!(s.get(1).unwrap().day(), 2);
    }

    #[test]
    fn test_datetime_series_parse() {
        let s = DateTimeSeries::parse("dates", &["2024-01-01", "invalid", "2024-12-31"]);
        assert_eq!(s.len(), 3);
        assert!(!s.is_null_at(0));
        assert!(s.is_null_at(1));
        assert!(!s.is_null_at(2));
        assert_eq!(s.null_count(), 1);
    }

    #[test]
    fn test_datetime_series_components() {
        let dates = vec![
            DateTime::from_ymd_hms(2024, 3, 15, 10, 30, 0).unwrap(),
            DateTime::from_ymd_hms(2024, 7, 4, 14, 0, 0).unwrap(),
        ];
        let s = DateTimeSeries::new("dt", dates);
        assert_eq!(s.year(), vec![Some(2024), Some(2024)]);
        assert_eq!(s.month(), vec![Some(3), Some(7)]);
        assert_eq!(s.day(), vec![Some(15), Some(4)]);
        assert_eq!(s.hour(), vec![Some(10), Some(14)]);
        assert_eq!(s.minute(), vec![Some(30), Some(0)]);
    }

    #[test]
    fn test_datetime_series_add_duration() {
        let dates = vec![
            DateTime::from_ymd(2024, 1, 1).unwrap(),
            DateTime::from_ymd(2024, 6, 15).unwrap(),
        ];
        let s = DateTimeSeries::new("dt", dates);
        let shifted = s.add_duration(Duration::days(1));
        assert_eq!(shifted.get(0).unwrap().day(), 2);
        assert_eq!(shifted.get(1).unwrap().day(), 16);
    }

    #[test]
    fn test_datetime_series_min_max() {
        let dates = vec![
            DateTime::from_ymd(2024, 6, 15).unwrap(),
            DateTime::from_ymd(2024, 1, 1).unwrap(),
            DateTime::from_ymd(2024, 12, 31).unwrap(),
        ];
        let s = DateTimeSeries::new("dt", dates);
        assert_eq!(s.min().unwrap().month(), 1);
        assert_eq!(s.max().unwrap().month(), 12);
    }

    #[test]
    fn test_datetime_series_date_range() {
        let start = DateTime::from_ymd(2024, 1, 1).unwrap();
        let end = DateTime::from_ymd(2024, 1, 5).unwrap();
        let s = DateTimeSeries::date_range("range", start, end, Duration::days(1)).unwrap();
        assert_eq!(s.len(), 5);
        assert_eq!(s.get(0).unwrap().day(), 1);
        assert_eq!(s.get(4).unwrap().day(), 5);
    }

    #[test]
    fn test_datetime_series_with_nulls() {
        let dates = vec![
            DateTime::from_ymd(2024, 1, 1).unwrap(),
            DateTime::epoch(),
            DateTime::from_ymd(2024, 12, 31).unwrap(),
        ];
        let s = DateTimeSeries::with_nulls("dt", dates, vec![false, true, false]).unwrap();
        assert_eq!(s.null_count(), 1);
        assert!(s.is_null_at(1));
        assert_eq!(s.year(), vec![Some(2024), None, Some(2024)]);
    }

    #[test]
    fn test_datetime_series_any_series() {
        let dates = vec![
            DateTime::from_ymd(2024, 1, 1).unwrap(),
            DateTime::from_ymd(2024, 6, 15).unwrap(),
        ];
        let s = DateTimeSeries::new("dt", dates);
        let boxed: Box<dyn AnySeries> = Box::new(s);
        assert_eq!(boxed.dtype(), DType::DateTime);
        assert_eq!(boxed.len(), 2);
        assert_eq!(boxed.display_value(0), "2024-01-01");

        let cloned = boxed.clone_box();
        assert_eq!(cloned.len(), 2);
    }

    #[test]
    fn test_datetime_series_filter() {
        let dates = vec![
            DateTime::from_ymd(2024, 1, 1).unwrap(),
            DateTime::from_ymd(2024, 6, 15).unwrap(),
            DateTime::from_ymd(2024, 12, 31).unwrap(),
        ];
        let s = DateTimeSeries::new("dt", dates);
        let boxed: Box<dyn AnySeries> = Box::new(s);
        let filtered = boxed.filter_mask(&[true, false, true]);
        assert_eq!(filtered.len(), 2);
        let dt_s = filtered.as_any().downcast_ref::<DateTimeSeries>().unwrap();
        assert_eq!(dt_s.get(0).unwrap().month(), 1);
        assert_eq!(dt_s.get(1).unwrap().month(), 12);
    }

    #[test]
    fn test_datetime_before_epoch() {
        let dt = DateTime::from_ymd(1960, 6, 15).unwrap();
        assert_eq!(dt.year(), 1960);
        assert_eq!(dt.month(), 6);
        assert_eq!(dt.day(), 15);
    }

    #[test]
    fn test_datetime_parse_invalid_strings() {
        assert!(DateTime::parse("not-a-date").is_err());
        assert!(DateTime::parse("").is_err());
        assert!(DateTime::parse("2024").is_err());
        assert!(DateTime::parse("20240101").is_err());
    }

    #[test]
    fn test_date_range_zero_step_error() {
        let start = DateTime::from_ymd(2024, 1, 1).unwrap();
        let end = DateTime::from_ymd(2024, 1, 5).unwrap();
        let result = DateTimeSeries::date_range("r", start, end, Duration::zero());
        assert!(result.is_err());
    }

    #[test]
    fn test_date_range_backward() {
        let start = DateTime::from_ymd(2024, 1, 5).unwrap();
        let end = DateTime::from_ymd(2024, 1, 1).unwrap();
        let s = DateTimeSeries::date_range("r", start, end, Duration::days(-1)).unwrap();
        assert_eq!(s.len(), 5);
        assert_eq!(s.get(0).unwrap().day(), 5);
        assert_eq!(s.get(4).unwrap().day(), 1);
    }

    #[test]
    fn test_datetime_day_of_year() {
        let dt = DateTime::from_ymd(2024, 1, 1).unwrap();
        assert_eq!(dt.day_of_year(), 1);
        let dt = DateTime::from_ymd(2024, 12, 31).unwrap();
        assert_eq!(dt.day_of_year(), 366); // 2024 is a leap year
    }

    #[test]
    fn test_datetime_from_millis() {
        let dt = DateTime::from_millis(1_704_067_200_000);
        assert_eq!(dt.year(), 2024);
        assert_eq!(dt.month(), 1);
        assert_eq!(dt.day(), 1);
    }

    #[test]
    fn test_datetime_sub_duration() {
        let dt = DateTime::from_ymd(2024, 2, 1).unwrap();
        let earlier = dt.sub_duration(Duration::days(1));
        assert_eq!(earlier.month(), 1);
        assert_eq!(earlier.day(), 31);
    }

    #[test]
    fn test_duration_abs() {
        let d = Duration::seconds(-90);
        let abs_d = d.abs();
        assert_eq!(abs_d.total_seconds(), 90);
    }

    #[test]
    fn test_datetime_series_empty() {
        let s = DateTimeSeries::new("empty", vec![]);
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
        assert_eq!(s.get(0), None);
        assert_eq!(s.min(), None);
        assert_eq!(s.max(), None);
    }

    #[test]
    fn test_datetime_series_from_millis() {
        let s = DateTimeSeries::from_millis("ts", &[0, 86_400_000]);
        assert_eq!(s.len(), 2);
        assert_eq!(s.get(0).unwrap().year(), 1970);
        assert_eq!(s.get(1).unwrap().day(), 2);
    }

    #[test]
    fn test_datetime_series_sub_duration() {
        let dates = vec![
            DateTime::from_ymd(2024, 1, 10).unwrap(),
            DateTime::from_ymd(2024, 6, 20).unwrap(),
        ];
        let s = DateTimeSeries::new("dt", dates);
        let shifted = s.sub_duration(Duration::days(1));
        assert_eq!(shifted.get(0).unwrap().day(), 9);
        assert_eq!(shifted.get(1).unwrap().day(), 19);
    }

    #[test]
    fn test_datetime_nanosecond() {
        let dt = DateTime::from_nanos(1_000_000_123);
        assert_eq!(dt.nanosecond(), 123);
    }

    #[test]
    fn test_datetime_timestamp_millis() {
        let dt = DateTime::from_millis(12345);
        assert_eq!(dt.timestamp_millis(), 12345);
    }
}
