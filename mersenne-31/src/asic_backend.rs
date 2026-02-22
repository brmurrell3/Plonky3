extern crate std;

use alloc::boxed::Box;
use std::cell::RefCell;

use m31_accel_driver::batch::Batch;
use m31_accel_driver::connection::{AsicConnection, Transport};

std::thread_local! {
    static ASIC: RefCell<Option<AsicConnectionHolder>> = RefCell::new(None);
}

struct AsicConnectionHolder {
    conn: AsicConnection<Box<dyn Transport>>,
}

/// Initialize the ASIC backend with a serial port connection.
pub fn init_asic(port: &str, baud: u32) -> Result<(), m31_accel_driver::error::AsicError> {
    let conn = AsicConnection::open_transport(port, baud)?;
    ASIC.with(|cell| {
        *cell.borrow_mut() = Some(AsicConnectionHolder { conn });
    });
    Ok(())
}

/// Initialize the ASIC backend with a custom transport (e.g., mock).
pub fn init_asic_with_transport<T: Transport + 'static>(transport: T) {
    let boxed: Box<dyn Transport> = Box::new(transport);
    let conn = AsicConnection::new(boxed);
    ASIC.with(|cell| {
        *cell.borrow_mut() = Some(AsicConnectionHolder { conn });
    });
}

/// Check if the ASIC backend has been initialized.
pub fn is_initialized() -> bool {
    ASIC.with(|cell| cell.borrow().is_some())
}

/// Compute a dot product using the ASIC's MAC unit.
/// Returns `None` if the ASIC is not initialized or if the batch execution fails,
/// allowing the caller to fall back to software computation.
pub fn execute_dot_product(a: &[u32], b: &[u32]) -> Option<u32> {
    ASIC.with(|cell| {
        let mut borrow = cell.borrow_mut();
        let holder = borrow.as_mut()?;
        let mut batch = Batch::new();
        batch.push_dot_product(a, b);
        let results = holder.conn.execute_batch(&batch).ok()?;
        results.first().copied()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use m31_accel_driver::mock::MockAsic;
    use p3_field::PrimeCharacteristicRing;

    #[test]
    fn test_asic_dot_product_via_mock() {
        init_asic_with_transport(MockAsic::new());
        let result = execute_dot_product(&[2, 3], &[4, 5]).unwrap();
        assert_eq!(result, 23);
    }

    #[test]
    fn test_asic_dot_product_larger() {
        init_asic_with_transport(MockAsic::new());
        let result = execute_dot_product(&[1, 2, 3, 4], &[5, 6, 7, 8]).unwrap();
        assert_eq!(result, 70);
    }

    #[test]
    fn test_asic_dot_matches_software() {
        init_asic_with_transport(MockAsic::new());

        let lhs: [crate::Mersenne31; 16] = core::array::from_fn(|i| crate::Mersenne31::new(i as u32 * 7 + 3));
        let rhs: [crate::Mersenne31; 16] = core::array::from_fn(|i| crate::Mersenne31::new(i as u32 * 13 + 5));

        let asic_result = crate::Mersenne31::dot_product(&lhs, &rhs);
        let sw_result: crate::Mersenne31 = lhs.iter().zip(rhs.iter()).map(|(a, b)| *a * *b).sum();

        assert_eq!(asic_result, sw_result);
    }

    #[test]
    fn test_asic_dot_matches_software_100_random() {
        init_asic_with_transport(MockAsic::new());

        // Deterministic LCG PRNG
        let mut seed: u64 = 0xDEAD_BEEF_CAFE;
        let next = |s: &mut u64| -> u32 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((*s >> 33) as u32) % 0x7FFFFFFF
        };

        for _ in 0..100 {
            let lhs: [crate::Mersenne31; 16] = core::array::from_fn(|_| crate::Mersenne31::new(next(&mut seed)));
            let rhs: [crate::Mersenne31; 16] = core::array::from_fn(|_| crate::Mersenne31::new(next(&mut seed)));

            let asic_result = crate::Mersenne31::dot_product(&lhs, &rhs);
            let sw_result: crate::Mersenne31 = lhs.iter().zip(rhs.iter()).map(|(a, b)| *a * *b).sum();

            assert_eq!(asic_result, sw_result, "mismatch on random vector");
        }
    }
}
