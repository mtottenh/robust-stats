#[test]
fn test_auto_engine_is_parallel() {
    use robust_core::execution::{auto_engine, ExecutionEngine, ExecutionStrategy};
    
    let engine = auto_engine();
    
    println!("Auto engine configuration:");
    println!("  Strategy: {:?}", engine.strategy());
    println!("  Is parallel: {}", engine.is_parallel());
    println!("  Num threads: {}", engine.num_threads());
    
    #[cfg(feature = "parallel")]
    {
        // When parallel feature is enabled, auto_engine should use hierarchical/budgeted execution
        // which reports as parallel
        assert!(engine.is_parallel(), "auto_engine() should create a parallel engine when parallel feature is enabled");
        assert!(engine.num_threads() > 1, "auto_engine() should have multiple threads available");
        
        // Test that it actually executes in parallel
        let results = engine.execute_batch(4, |i| {
            println!("  Batch item {} in thread: {:?}", i, std::thread::current().id());
            std::thread::sleep(std::time::Duration::from_millis(10));
            i * 10
        });
        assert_eq!(results, vec![0, 10, 20, 30]);
    }
    
    #[cfg(not(feature = "parallel"))]
    {
        // Without parallel feature, should be sequential
        assert!(!engine.is_parallel(), "auto_engine() should create a sequential engine when parallel feature is disabled");
        assert_eq!(engine.num_threads(), 1);
    }
}