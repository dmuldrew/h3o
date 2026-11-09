use super::*;
use crate::{CellIndex, Resolution};

#[test]
fn first_axe_none() {
    let index =
        CellIndex::try_from(0x8f2800000000000).expect("valid cell index");
    let result = first_axe(index.into());
    assert_eq!(result, None, "zero til the end");

    let index =
        CellIndex::try_from(0x8029fffffffffff).expect("valid cell index");
    let result = first_axe(index.into());
    assert_eq!(result, None, "no cell");

    let index =
        CellIndex::try_from(0x832800fffffffff).expect("valid cell index");
    let result = first_axe(index.into());
    assert_eq!(result, None, "some zero then unused");
}

#[test]
fn first_axe_some() {
    let resolutions = Resolution::range(Resolution::One, Resolution::Fifteen);
    let indexes = [
        0x8f287478ab9c2ab, // 20-1-6-4-3-6-1-2-5-6-3-4-1-2-5-3
        0x8f283478ab9c2ab, // 20-0-6-4-3-6-1-2-5-6-3-4-1-2-5-3
        0x8f280478ab9c2ab, // 20-0-0-4-3-6-1-2-5-6-3-4-1-2-5-3
        0x8f280078ab9c2ab, // 20-0-0-0-3-6-1-2-5-6-3-4-1-2-5-3
        0x8f280018ab9c2ab, // 20-0-0-0-0-6-1-2-5-6-3-4-1-2-5-3
        0x8f280000ab9c2ab, // 20-0-0-0-0-0-1-2-5-6-3-4-1-2-5-3
        0x8f2800002b9c2ab, // 20-0-0-0-0-0-0-2-5-6-3-4-1-2-5-3
        0x8f2800000b9c2ab, // 20-0-0-0-0-0-0-0-5-6-3-4-1-2-5-3
        0x8f280000019c2ab, // 20-0-0-0-0-0-0-0-0-6-3-4-1-2-5-3
        0x8f280000001c2ab, // 20-0-0-0-0-0-0-0-0-0-3-4-1-2-5-3
        0x8f28000000042ab, // 20-0-0-0-0-0-0-0-0-0-0-4-1-2-5-3
        0x8f28000000002ab, // 20-0-0-0-0-0-0-0-0-0-0-0-1-2-5-3
        0x8f28000000000ab, // 20-0-0-0-0-0-0-0-0-0-0-0-0-2-5-3
        0x8f280000000002b, // 20-0-0-0-0-0-0-0-0-0-0-0-0-0-5-3
        0x8f2800000000003, // 20-0-0-0-0-0-0-0-0-0-0-0-0-0-0-3
    ];

    for (resolution, value) in resolutions.zip(indexes) {
        let index = CellIndex::try_from(value).expect("valid cells");
        let expected = index.direction_at(resolution).and_then(Direction::axe);

        let result = first_axe(index.into());

        assert_eq!(result, expected, "resolution {resolution:?}");
    }
}

/// Digit-by-digit rotation (the previous implementation), as a reference.
fn rotate60_reference<const CCW: bool>(mut bits: u64, count: usize) -> u64 {
    for resolution in Resolution::range(Resolution::One, get_resolution(bits)) {
        let direction =
            Direction::new_unchecked(get_direction(bits, resolution));
        bits = set_direction(
            bits,
            direction.rotate60::<CCW>(count).into(),
            resolution,
        );
    }
    bits
}

/// Digit-by-digit pentagon rotation (the previous implementation), as a
/// reference.
fn pentagon_rotate60_reference<const CCW: bool>(bits: u64) -> u64 {
    if get_resolution(bits) == Resolution::Zero {
        return bits;
    }
    let direction = if CCW { Direction::JK } else { Direction::IK }.axe();
    let count = if first_axe(bits) == direction { 2 } else { 1 };
    rotate60_reference::<CCW>(bits, count)
}

/// Cell-like bits at `resolution` with the given used digits (unused digits
/// set to 7, as in a real index), base cell 20.
fn bits_with_digits(
    resolution: Resolution,
    mut digits: impl FnMut() -> u8,
) -> u64 {
    let mut bits = set_resolution(0x0800_0000_0000_0000, resolution);
    bits = h3o_bit::set_base_cell(bits, 20);
    bits = set_unused(bits, resolution);
    for res in Resolution::range(Resolution::One, resolution) {
        bits = set_direction(bits, digits(), res);
    }
    bits
}

#[test]
fn digit_masks_cover_every_digit() {
    assert_eq!(DIGIT_BIT0 | DIGIT_BIT1 | DIGIT_BIT2, DIRECTIONS_MASK);
    assert_eq!(digits_mask(Resolution::Fifteen), DIRECTIONS_MASK);
    assert_eq!(digits_mask(Resolution::Zero), 0);
    assert_eq!(digits_mask(Resolution::One), 0b111 << 42);
}

#[test]
fn rotate60_matches_digit_by_digit_rotation() {
    // Every direction in every digit position, with every count in both
    // directions.
    for resolution in Resolution::range(Resolution::Zero, Resolution::Fifteen) {
        for position in 0..u8::from(resolution) {
            for direction in 0..=6 {
                let mut i = 0;
                let bits = bits_with_digits(resolution, || {
                    i += 1;
                    if i - 1 == position { direction } else { 0 }
                });
                for count in 0..13 {
                    assert_eq!(
                        rotate60::<true>(bits, count),
                        rotate60_reference::<true>(bits, count),
                        "CCW {resolution:?} position {position} direction {direction} count {count}"
                    );
                    assert_eq!(
                        rotate60::<false>(bits, count),
                        rotate60_reference::<false>(bits, count),
                        "CW {resolution:?} position {position} direction {direction} count {count}"
                    );
                }
            }
        }
    }

    // Random digit strings.
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 33) % 7) as u8
    };
    for _ in 0..20_000 {
        for resolution in
            Resolution::range(Resolution::One, Resolution::Fifteen)
        {
            let bits = bits_with_digits(resolution, &mut next);
            for count in 0..7 {
                assert_eq!(
                    rotate60::<true>(bits, count),
                    rotate60_reference::<true>(bits, count),
                    "CCW {bits:#x} count {count}"
                );
                assert_eq!(
                    rotate60::<false>(bits, count),
                    rotate60_reference::<false>(bits, count),
                    "CW {bits:#x} count {count}"
                );
            }
            assert_eq!(
                pentagon_rotate60::<true>(bits),
                pentagon_rotate60_reference::<true>(bits),
                "pentagon CCW {bits:#x}"
            );
            assert_eq!(
                pentagon_rotate60::<false>(bits),
                pentagon_rotate60_reference::<false>(bits),
                "pentagon CW {bits:#x}"
            );
        }
    }
}
