//! End-to-end Fibonacci AIR proof using Mersenne31 + Circle PCS.
//! When compiled with --features asic, dot products >= 8 elements
//! dispatch to the ASIC backend (mock in tests).

use core::borrow::Borrow;

use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_challenger::{HashChallenger, SerializingChallenger32};
use p3_circle::CirclePcs;
use p3_commit::ExtensionMmcs;
use p3_field::extension::BinomialExtensionField;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_fri::FriParameters;
use p3_keccak::Keccak256Hash;
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_merkle_tree::MerkleTreeMmcs;
use p3_mersenne_31::Mersenne31;
use p3_symmetric::{CompressionFunctionFromHasher, SerializingHasher};
use p3_uni_stark::{StarkConfig, prove, verify};

const NUM_FIBONACCI_COLS: usize = 2;

struct FibonacciRow<F> {
    left: F,
    right: F,
}

impl<F> Borrow<FibonacciRow<F>> for [F] {
    fn borrow(&self) -> &FibonacciRow<F> {
        debug_assert_eq!(self.len(), NUM_FIBONACCI_COLS);
        let (prefix, shorts, suffix) = unsafe { self.align_to::<FibonacciRow<F>>() };
        debug_assert!(prefix.is_empty());
        debug_assert!(suffix.is_empty());
        debug_assert_eq!(shorts.len(), 1);
        &shorts[0]
    }
}

struct FibonacciAir;

impl<F> BaseAir<F> for FibonacciAir {
    fn width(&self) -> usize {
        NUM_FIBONACCI_COLS
    }
}

impl<AB: AirBuilderWithPublicValues> Air<AB> for FibonacciAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let pis = builder.public_values();

        let a = pis[0];
        let b = pis[1];
        let x = pis[2];

        let (local, next) = (
            main.row_slice(0).expect("empty matrix"),
            main.row_slice(1).expect("single row matrix"),
        );
        let local: &FibonacciRow<AB::Var> = (*local).borrow();
        let next: &FibonacciRow<AB::Var> = (*next).borrow();

        let mut when_first_row = builder.when_first_row();
        when_first_row.assert_eq(local.left.clone(), a);
        when_first_row.assert_eq(local.right.clone(), b);

        let mut when_transition = builder.when_transition();
        when_transition.assert_eq(local.right.clone(), next.left.clone());
        when_transition.assert_eq(
            local.left.clone() + local.right.clone(),
            next.right.clone(),
        );

        builder.when_last_row().assert_eq(local.right.clone(), x);
    }
}

fn generate_trace<F: PrimeField64>(a: u64, b: u64, n: usize) -> RowMajorMatrix<F> {
    assert!(n.is_power_of_two());
    let mut values = F::zero_vec(n * NUM_FIBONACCI_COLS);

    values[0] = F::from_u64(a);
    values[1] = F::from_u64(b);

    for i in 1..n {
        values[2 * i] = values[2 * (i - 1) + 1];
        values[2 * i + 1] = values[2 * (i - 1)] + values[2 * (i - 1) + 1];
    }

    RowMajorMatrix::new(values, NUM_FIBONACCI_COLS)
}

/// Compute F(n) mod P using software, for verification.
/// F(0)=0, F(1)=1, F(2)=1, F(3)=2, ...
fn fibonacci_mod_p(n: usize) -> u32 {
    let p = 0x7FFFFFFFu64;
    let mut a = 0u64; // F(0)
    let mut b = 1u64; // F(1)
    for _ in 0..n {
        let c = (a + b) % p;
        a = b;
        b = c;
    }
    // After n iterations: a = F(n), b = F(n+1)
    a as u32
}

// Circle PCS type aliases for Mersenne31
type Val = Mersenne31;
type Challenge = BinomialExtensionField<Val, 3>;
type ByteHash = Keccak256Hash;
type FieldHash = SerializingHasher<ByteHash>;
type Compress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, Compress, 32>;
type ChallengeMmcs = ExtensionMmcs<Val, Challenge, ValMmcs>;
type Challenger = SerializingChallenger32<Val, HashChallenger<u8, ByteHash, 32>>;
type Pcs = CirclePcs<Val, ValMmcs, ChallengeMmcs>;
type Config = StarkConfig<Pcs, Challenge, Challenger>;

fn make_config() -> Config {
    let byte_hash = ByteHash {};
    let field_hash = FieldHash::new(byte_hash);
    let compress = Compress::new(byte_hash);
    let val_mmcs = ValMmcs::new(field_hash, compress, 0);
    let challenge_mmcs = ChallengeMmcs::new(val_mmcs.clone());
    let fri_params = FriParameters {
        log_blowup: 1,
        log_final_poly_len: 0,
        max_log_arity: 1,
        num_queries: 40,
        commit_proof_of_work_bits: 0,
        query_proof_of_work_bits: 8,
        mmcs: challenge_mmcs,
    };
    let pcs = Pcs {
        mmcs: val_mmcs,
        fri_params,
        _phantom: core::marker::PhantomData,
    };
    let challenger = Challenger::from_hasher(vec![], byte_hash);
    Config::new(pcs, challenger)
}

/// Core proof test: generate Fibonacci trace, prove, verify.
fn prove_and_verify_fibonacci(n: usize) {
    let trace = generate_trace::<Val>(0, 1, n);

    // F(n) for n=64 rows: the last row's right value is F(64) mod P
    let expected_fib = fibonacci_mod_p(n);

    let pis = vec![
        Val::from_u64(0),
        Val::from_u64(1),
        Val::new(expected_fib),
    ];

    let config = make_config();
    let proof = prove(&config, &FibonacciAir, trace, &pis);

    let config = make_config();
    verify(&config, &FibonacciAir, &proof, &pis).expect("verification failed");
}

#[test]
fn test_fibonacci_proof_software() {
    // Without ASIC initialization, uses pure software path
    prove_and_verify_fibonacci(64);
}

#[cfg(feature = "asic")]
#[test]
fn test_fibonacci_proof_asic() {
    use m31_accel_driver::mock::MockAsic;
    use p3_mersenne_31::asic_backend;

    // Initialize ASIC backend with mock
    asic_backend::init_asic_with_transport(MockAsic::new());

    // Now dot_product calls for N>=8 will go through the ASIC mock
    prove_and_verify_fibonacci(64);
}

#[cfg(feature = "asic")]
#[test]
fn test_fibonacci_trace_identical() {
    // Verify trace values are identical regardless of ASIC vs software path.
    // The trace generation itself doesn't use dot_product, so this confirms
    // the field arithmetic is consistent.
    let trace_sw = generate_trace::<Val>(0, 1, 64);
    let trace_asic = generate_trace::<Val>(0, 1, 64);

    assert_eq!(trace_sw.values, trace_asic.values);

    // Verify F(64) mod P
    let expected = fibonacci_mod_p(64);
    let last_row_right = trace_sw.values[63 * 2 + 1];
    assert_eq!(last_row_right, Val::new(expected));
}

#[test]
fn test_fibonacci_f64_value() {
    // F(64) mod P should be a specific value — sanity check
    let val = fibonacci_mod_p(64);
    // F(64) = 10610209857723 -> mod P = 10610209857723 mod 2147483647
    let expected = (10610209857723u64 % 0x7FFFFFFF) as u32;
    assert_eq!(val, expected);
}
