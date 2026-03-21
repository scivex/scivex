use std::fmt;

/// Errors that can occur during graph operations.
///
/// # Examples
///
/// ```
/// # use scivex_graph::GraphError;
/// let err = GraphError::NodeNotFound { id: 42 };
/// assert_eq!(format!("{err}"), "node 42 not found");
/// ```
#[cfg_attr(
    feature = "serde-support",
    derive(serde::Serialize, serde::Deserialize)
)]
#[derive(Debug, Clone, PartialEq)]
pub enum GraphError {
    /// A referenced node does not exist or has been removed.
    NodeNotFound {
        /// The ID of the missing node.
        id: usize,
    },
    /// A referenced edge does not exist.
    EdgeNotFound {
        /// The source node of the missing edge.
        from: usize,
        /// The destination node of the missing edge.
        to: usize,
    },
    /// The graph contains no nodes.
    EmptyGraph,
    /// A negative edge weight was encountered where only non-negative weights are allowed.
    NegativeWeight,
    /// A negative-weight cycle was detected.
    NegativeCycle,
    /// A cycle was detected in a graph expected to be acyclic.
    CycleDetected,
    /// An invalid parameter was provided.
    InvalidParameter {
        /// The name of the invalid parameter.
        name: &'static str,
        /// A description of why the parameter is invalid.
        reason: &'static str,
    },
    /// An error propagated from `scivex-core`.
    CoreError(scivex_core::CoreError),
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeNotFound { id } => write!(f, "node {id} not found"),
            Self::EdgeNotFound { from, to } => {
                write!(f, "edge ({from}, {to}) not found")
            }
            Self::EmptyGraph => write!(f, "graph is empty"),
            Self::NegativeWeight => write!(f, "negative edge weight not allowed"),
            Self::NegativeCycle => write!(f, "negative cycle detected"),
            Self::CycleDetected => write!(f, "cycle detected"),
            Self::InvalidParameter { name, reason } => {
                write!(f, "invalid parameter `{name}`: {reason}")
            }
            Self::CoreError(e) => write!(f, "core error: {e}"),
        }
    }
}

impl std::error::Error for GraphError {}

impl From<scivex_core::CoreError> for GraphError {
    fn from(e: scivex_core::CoreError) -> Self {
        Self::CoreError(e)
    }
}

/// A specialized `Result` type for graph operations.
pub type Result<T> = std::result::Result<T, GraphError>;
