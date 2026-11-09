//! Bit twiddling.

use super::IndexMode;
use crate::{Direction, Edge, Resolution, Vertex};
use core::{cmp, num::NonZeroU8};

/// Offset (in bits) of the mode in an H3 index.
const MODE_OFFSET: usize = 59;
const MODE_MASK: u64 = 0b1111 << MODE_OFFSET;

/// Offset (in bits) of the cell edge in an H3 index.
const EDGE_OFFSET: usize = 56;
const EDGE_MASK: u64 = 0b111 << EDGE_OFFSET;

/// Offset (in bits) of the cell vertex in an H3 index.
const VERTEX_OFFSET: usize = 56;
const VERTEX_MASK: u64 = 0b111 << VERTEX_OFFSET;

// Bitmask to select the directions bits in an H3 index.
pub const DIRECTIONS_MASK: u64 = 0x0000_1fff_ffff_ffff;

/// Returns the H3 index mode  bits.
#[must_use]
pub const fn get_mode(bits: u64) -> u8 {
    ((bits & MODE_MASK) >> MODE_OFFSET) as u8
}

/// Clears the H3 index mode bits.
#[must_use]
pub const fn clr_mode(bits: u64) -> u64 {
    bits & !MODE_MASK
}

/// Sets the H3 index mode bits.
#[must_use]
pub const fn set_mode(bits: u64, mode: IndexMode) -> u64 {
    clr_mode(bits) | ((mode as u64) << MODE_OFFSET)
}

/// Returns the H3 index cell edge bits.
#[must_use]
pub const fn get_edge(bits: u64) -> u8 {
    ((bits & EDGE_MASK) >> EDGE_OFFSET) as u8
}

/// Sets the H3 index mode bits.
#[must_use]
pub fn set_edge(bits: u64, edge: Edge) -> u64 {
    clr_edge(bits) | (u64::from(edge) << EDGE_OFFSET)
}

/// Clears the H3 index cell edge bits.
#[must_use]
pub const fn clr_edge(bits: u64) -> u64 {
    bits & !EDGE_MASK
}

/// Returns the H3 index cell vertex bits.
#[must_use]
pub const fn get_vertex(bits: u64) -> u8 {
    ((bits & VERTEX_MASK) >> VERTEX_OFFSET) as u8
}

/// Sets the H3 index mode bits.
#[must_use]
pub fn set_vertex(bits: u64, vertex: Vertex) -> u64 {
    clr_vertex(bits) | (u64::from(vertex) << VERTEX_OFFSET)
}

/// Clears the H3 index cell vertex bits.
#[must_use]
pub const fn clr_vertex(bits: u64) -> u64 {
    bits & !VERTEX_MASK
}

/// Returns the H3 index resolution.
#[must_use]
pub const fn get_resolution(bits: u64) -> Resolution {
    // SAFETY: the masking restricts the value on 4 bits (thus 0-15).
    Resolution::new_unchecked(h3o_bit::get_resolution(bits))
}

/// Sets the H3 index resolution bits.
#[must_use]
pub fn set_resolution(bits: u64, resolution: Resolution) -> u64 {
    h3o_bit::set_resolution(bits, resolution.into())
}

/// Returns the H3 index direction bits at the given resolution.
#[must_use]
pub fn get_direction(bits: u64, resolution: Resolution) -> u8 {
    debug_assert!(resolution != Resolution::Zero, "res 0 means no directions");
    h3o_bit::get_direction(bits, resolution.into())
}

/// Clears the H3 index direction at the given resolution.
#[must_use]
pub fn clr_direction(bits: u64, resolution: Resolution) -> u64 {
    debug_assert!(resolution != Resolution::Zero, "res 0 means no directions");
    h3o_bit::clr_direction(bits, resolution.into())
}

/// Set the H3 index direction bits at the given resolution.
pub fn set_direction(bits: u64, cell: u8, resolution: Resolution) -> u64 {
    debug_assert!(resolution != Resolution::Zero, "res 0 means no directions");
    h3o_bit::set_direction(bits, resolution.into(), cell)
}

/// Sets unused directions in an H3 index at the given resolution.
#[must_use]
pub fn set_unused(bits: u64, resolution: Resolution) -> u64 {
    let unused_end_offset = resolution.direction_offset();
    let unused_bits = (1 << unused_end_offset) - 1;

    bits | unused_bits
}

/// Returns the direction bits of the first non-center direction from an H3
/// index.
#[must_use]
pub fn first_axe(bits: u64) -> Option<NonZeroU8> {
    const PREFIX_LENGTH: u32 = DIRECTIONS_MASK.leading_zeros();

    let resolution = get_resolution(bits);
    // No direction at res 0 (only base cell), thus no non-zero direction.
    if resolution == Resolution::Zero {
        return None;
    }

    // Ignore the `0` from the hidden bits (hence `- PREFIX_LENGTH`).
    let bit_index = (bits & DIRECTIONS_MASK).leading_zeros() - PREFIX_LENGTH;

    // +1 because the first direction is at resolution 1, not 0.
    #[expect(clippy::cast_possible_truncation, reason = "values are in range")]
    let resolution = cmp::min(
        ((bit_index / h3o_bit::DIRECTION_BITSIZE as u32) + 1) as u8,
        resolution.into(),
    );

    NonZeroU8::new(get_direction(bits, Resolution::new_unchecked(resolution)))
}

/// Returns a cell rotated `count` time, by 60 degrees step.
pub fn rotate60<const CCW: bool>(bits: u64, count: usize) -> u64 {
    let digits = digits_mask(get_resolution(bits));
    let mut rotated = bits & digits;
    let mut count = count % 6;
    if count >= 3 {
        rotated = digits_rotate180(rotated);
        count -= 3;
    }
    for _ in 0..count {
        rotated = digits_rotate60::<CCW>(rotated);
    }

    (bits & !digits) | rotated
}

/// Bit 0 of every 3-bit direction digit.
const DIGIT_BIT0: u64 = 0x0000_0492_4924_9249;
/// Bit 1 of every 3-bit direction digit.
const DIGIT_BIT1: u64 = DIGIT_BIT0 << 1;
/// Bit 2 of every 3-bit direction digit.
const DIGIT_BIT2: u64 = DIGIT_BIT0 << 2;

/// Mask of the direction digits used at `resolution` (resolutions 1 to
/// `resolution`; the unused digits below them are excluded).
fn digits_mask(resolution: Resolution) -> u64 {
    DIRECTIONS_MASK & !((1 << resolution.direction_offset()) - 1)
}

/// Rotates every direction digit of `digits` by 60 degrees at once.
///
/// A digit is an IJK unit vector, one bit per axis (`i` = bit 2, `j` = bit
/// 1, `k` = bit 0): an axis direction has one bit set, a direction between
/// two axes has two. Rotating counterclockwise moves each axis to the next
/// one (`k` → `i` → `j` → `k`), so an axis direction gains the bit of the
/// axis it moves toward (`K` → `IK`) and a two-axis direction keeps only
/// the bit both rotations agree on (`IK` → `I`):
///
/// - one bit set: `d | rot(d)`;
/// - two bits set: `d & rot(d)`;
///
/// where `rot` rotates the three bits of each digit right (counterclockwise)
/// or left (clockwise). The center (`0`) stays `0` either way.
///
/// Every digit must be a valid direction (0 to 6).
const fn digits_rotate60<const CCW: bool>(digits: u64) -> u64 {
    let rot = if CCW {
        ((digits >> 1) & (DIGIT_BIT0 | DIGIT_BIT1))
            | ((digits << 2) & DIGIT_BIT2)
    } else {
        ((digits << 1) & (DIGIT_BIT1 | DIGIT_BIT2))
            | ((digits >> 2) & DIGIT_BIT0)
    };
    // Bit 0 of each digit with at least two bits set, spread to the digit.
    let b1 = digits >> 1;
    let b2 = digits >> 2;
    let two = ((digits & b1) | (digits & b2) | (b1 & b2)) & DIGIT_BIT0;
    let two = two * 0b111;

    (two & digits & rot) | (!two & (digits | rot))
}

/// Rotates every direction digit of `digits` by 180 degrees at once: each
/// non-center direction becomes its opposite (`7 - d`, its bitwise
/// complement); the center stays `0`.
///
/// Every digit must be a valid direction (0 to 6).
const fn digits_rotate180(digits: u64) -> u64 {
    let nonzero = (digits | (digits >> 1) | (digits >> 2)) & DIGIT_BIT0;

    digits ^ (nonzero * 0b111)
}

/// Returns a pentagonal cell rotated by 60 degrees.
pub fn pentagon_rotate60<const CCW: bool>(bits: u64) -> u64 {
    let res = get_resolution(bits);
    if res == Resolution::Zero {
        return bits;
    }

    // Direction that would be rotated to the K axe (and thus require an
    // adjustment).
    let direction = if CCW { Direction::JK } else { Direction::IK }.axe();
    let count = if first_axe(bits) == direction { 2 } else { 1 };

    rotate60::<CCW>(bits, count)
}

#[cfg(test)]
#[path = "./bits_tests.rs"]
mod tests;
