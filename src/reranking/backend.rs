#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RerankerRuntime {
    Off,
    LocalCpu,
    LocalGpu,
    RemoteCpu,
    RemoteGpu,
}

impl RerankerRuntime {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::LocalCpu => "local-cpu",
            Self::LocalGpu => "local-gpu",
            Self::RemoteCpu => "remote-cpu",
            Self::RemoteGpu => "remote-gpu",
        }
    }
}

impl std::fmt::Display for RerankerRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}
