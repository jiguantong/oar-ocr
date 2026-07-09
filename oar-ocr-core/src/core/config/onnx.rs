//! ONNX Runtime configuration types and utilities.

use serde::{Deserialize, Serialize};

/// Graph optimization levels for ONNX Runtime.
///
/// This enum represents the different levels of graph optimization that can be applied
/// during ONNX Runtime session creation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum OrtGraphOptimizationLevel {
    /// Disable all optimizations.
    DisableAll,
    /// Enable basic optimizations.
    #[default]
    Level1,
    /// Enable extended optimizations.
    Level2,
    /// Enable all optimizations.
    Level3,
    /// Enable all optimizations (alias for Level3).
    All,
}

/// Execution providers for ONNX Runtime.
///
/// This enum represents the different execution providers that can be used
/// with ONNX Runtime for model inference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OrtExecutionProvider {
    /// CPU execution provider (always available)
    CPU {
        /// Whether to enable the CPU arena allocator.
        ///
        /// `Some(false)` is useful for desktop single-request workloads that prefer
        /// lower steady-state memory over peak throughput.
        arena_allocator: Option<bool>,
    },
    /// NVIDIA CUDA execution provider
    CUDA {
        /// CUDA device ID (default: 0)
        device_id: Option<i32>,
        /// Memory limit in bytes (optional)
        gpu_mem_limit: Option<usize>,
        /// Arena extend strategy: "NextPowerOfTwo" or "SameAsRequested"
        arena_extend_strategy: Option<String>,
        /// CUDNN convolution algorithm search: "Exhaustive", "Heuristic", or "Default"
        cudnn_conv_algo_search: Option<String>,
        /// CUDNN convolution use max workspace (default: true)
        cudnn_conv_use_max_workspace: Option<bool>,
    },
    /// DirectML execution provider (Windows only)
    DirectML {
        /// DirectML device ID (default: 0)
        device_id: Option<i32>,
    },
    /// OpenVINO execution provider
    OpenVINO {
        /// Device type (e.g., "CPU", "GPU", "MYRIAD")
        device_type: Option<String>,
        /// Number of threads (optional)
        num_threads: Option<usize>,
    },
    /// TensorRT execution provider
    TensorRT {
        /// TensorRT device ID (default: 0)
        device_id: Option<i32>,
        /// Maximum workspace size in bytes
        max_workspace_size: Option<usize>,
        /// Minimum subgraph size for TensorRT acceleration
        min_subgraph_size: Option<usize>,
        /// FP16 enable flag
        fp16_enable: Option<bool>,
        /// Enable use of timing cache to speed up builds
        timing_cache: Option<bool>,
        /// Set path for storing timing cache
        timing_cache_path: Option<String>,
        /// Force use of timing cache regardless of GPU match
        force_timing_cache: Option<bool>,
        /// Enable caching of TensorRT engines
        engine_cache: Option<bool>,
        /// Set path to store cached TensorRT engines
        engine_cache_path: Option<String>,
        /// Dump ep context model
        dump_ep_context_model: Option<bool>,
        /// The path of an embedded engine model
        ep_context_file_path: Option<String>,
    },
    /// CoreML execution provider (macOS/iOS only)
    CoreML {
        /// Use Apple Neural Engine only
        ane_only: Option<bool>,
        /// Enable subgraphs
        subgraphs: Option<bool>,
    },
    /// WebGPU execution provider
    WebGPU,
}

impl Default for OrtExecutionProvider {
    fn default() -> Self {
        Self::CPU {
            arena_allocator: None,
        }
    }
}

/// Configuration for ONNX Runtime sessions.
///
/// This struct contains various configuration options for ONNX Runtime sessions,
/// including threading, memory management, and optimization settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OrtSessionConfig {
    /// Number of threads used to parallelize execution within nodes
    pub intra_threads: Option<usize>,
    /// Number of threads used to parallelize execution across nodes
    pub inter_threads: Option<usize>,
    /// Enable parallel execution mode
    pub parallel_execution: Option<bool>,
    /// Graph optimization level
    pub optimization_level: Option<OrtGraphOptimizationLevel>,
    /// Execution providers in order of preference
    pub execution_providers: Option<Vec<OrtExecutionProvider>>,
    /// Enable memory pattern optimization
    pub enable_mem_pattern: Option<bool>,
    /// Log severity level (0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal)
    pub log_severity_level: Option<i32>,
    /// Log verbosity level
    pub log_verbosity_level: Option<i32>,
    /// Session configuration entries (key-value pairs)
    pub session_config_entries: Option<std::collections::HashMap<String, String>>,
}

impl OrtSessionConfig {
    /// Creates a new OrtSessionConfig with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the number of intra-op threads.
    pub fn with_intra_threads(mut self, threads: usize) -> Self {
        self.intra_threads = Some(threads);
        self
    }

    /// Sets the number of inter-op threads.
    pub fn with_inter_threads(mut self, threads: usize) -> Self {
        self.inter_threads = Some(threads);
        self
    }

    /// Enables or disables parallel execution.
    pub fn with_parallel_execution(mut self, enabled: bool) -> Self {
        self.parallel_execution = Some(enabled);
        self
    }

    /// Sets the graph optimization level.
    pub fn with_optimization_level(mut self, level: OrtGraphOptimizationLevel) -> Self {
        self.optimization_level = Some(level);
        self
    }

    /// Sets the execution providers, in order of preference.
    pub fn with_execution_providers(mut self, providers: Vec<OrtExecutionProvider>) -> Self {
        self.execution_providers = Some(providers);
        self
    }

    /// Appends a single execution provider.
    pub fn add_execution_provider(mut self, provider: OrtExecutionProvider) -> Self {
        if let Some(ref mut providers) = self.execution_providers {
            providers.push(provider);
        } else {
            self.execution_providers = Some(vec![provider]);
        }
        self
    }

    /// Adds a CPU execution provider with optional arena allocator control.
    pub fn add_cpu_execution_provider(self, arena_allocator: Option<bool>) -> Self {
        self.add_execution_provider(OrtExecutionProvider::CPU { arena_allocator })
    }

    /// Enables or disables memory pattern optimization.
    pub fn with_memory_pattern(mut self, enable: bool) -> Self {
        self.enable_mem_pattern = Some(enable);
        self
    }

    /// Sets the log severity level (0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal).
    pub fn with_log_severity_level(mut self, level: i32) -> Self {
        self.log_severity_level = Some(level);
        self
    }

    /// Sets the log verbosity level.
    pub fn with_log_verbosity_level(mut self, level: i32) -> Self {
        self.log_verbosity_level = Some(level);
        self
    }

    /// Adds a session configuration entry.
    pub fn add_config_entry<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        if let Some(ref mut entries) = self.session_config_entries {
            entries.insert(key.into(), value.into());
        } else {
            let mut entries = std::collections::HashMap::new();
            entries.insert(key.into(), value.into());
            self.session_config_entries = Some(entries);
        }
        self
    }

    /// Effective intra-op thread count, defaulting to available parallelism.
    pub fn get_intra_threads(&self) -> usize {
        self.intra_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
    }

    /// Effective inter-op thread count, defaulting to 1.
    pub fn get_inter_threads(&self) -> usize {
        self.inter_threads.unwrap_or(1)
    }

    /// Effective graph optimization level, defaulting to `OrtGraphOptimizationLevel::default()`.
    pub fn get_optimization_level(&self) -> OrtGraphOptimizationLevel {
        self.optimization_level.unwrap_or_default()
    }

    /// Configured execution providers, defaulting to CPU.
    pub fn get_execution_providers(&self) -> Vec<OrtExecutionProvider> {
        self.execution_providers
            .clone()
            .unwrap_or_else(|| vec![OrtExecutionProvider::default()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ort_session_config_builder() {
        let config = OrtSessionConfig::new()
            .with_intra_threads(4)
            .with_inter_threads(2)
            .with_optimization_level(OrtGraphOptimizationLevel::Level2)
            .with_memory_pattern(true)
            .add_cpu_execution_provider(Some(false));

        assert_eq!(config.intra_threads, Some(4));
        assert_eq!(config.inter_threads, Some(2));
        assert!(matches!(
            config.optimization_level,
            Some(OrtGraphOptimizationLevel::Level2)
        ));
        assert_eq!(config.enable_mem_pattern, Some(true));
        assert!(config.execution_providers.is_some());
        assert!(matches!(
            config.execution_providers.as_ref().and_then(|v| v.first()),
            Some(OrtExecutionProvider::CPU {
                arena_allocator: Some(false)
            })
        ));
    }

    #[test]
    fn test_ort_session_config_getters() {
        let config = OrtSessionConfig::new()
            .with_intra_threads(8)
            .with_inter_threads(4)
            .with_optimization_level(OrtGraphOptimizationLevel::All);

        assert_eq!(config.get_intra_threads(), 8);
        assert_eq!(config.get_inter_threads(), 4);
        assert!(matches!(
            config.get_optimization_level(),
            OrtGraphOptimizationLevel::All
        ));
    }
}
