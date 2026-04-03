use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::error::FemindError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendMode {
    RemotePrimary,
    LocalFallback,
    RemoteRecovering,
    Offline,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendFailureClass {
    Transient,
    Permanent,
}

#[derive(Debug)]
pub struct BackendPolicy {
    cooldown: Duration,
    state: Mutex<BackendPolicyState>,
}

#[derive(Debug)]
struct BackendPolicyState {
    mode: BackendMode,
    last_failure_at: Option<Instant>,
    last_failure_class: Option<BackendFailureClass>,
    last_failure_message: Option<String>,
    last_success_at: Option<Instant>,
}

impl BackendPolicy {
    pub fn new(cooldown: Duration) -> Self {
        Self {
            cooldown,
            state: Mutex::new(BackendPolicyState {
                mode: BackendMode::RemotePrimary,
                last_failure_at: None,
                last_failure_class: None,
                last_failure_message: None,
                last_success_at: None,
            }),
        }
    }

    pub fn mode(&self) -> BackendMode {
        self.lock_state().mode
    }

    pub fn last_failure_message(&self) -> Option<String> {
        self.lock_state().last_failure_message.clone()
    }

    pub fn should_attempt_primary(&self) -> bool {
        let state = self.lock_state();
        match state.mode {
            BackendMode::RemotePrimary | BackendMode::RemoteRecovering => true,
            BackendMode::LocalFallback => state
                .last_failure_at
                .is_none_or(|last| last.elapsed() >= self.cooldown),
            BackendMode::Offline => false,
        }
    }

    pub fn begin_recovery_attempt(&self) {
        let mut state = self.lock_state();
        if matches!(state.mode, BackendMode::LocalFallback)
            && state
                .last_failure_at
                .is_some_and(|last| last.elapsed() >= self.cooldown)
        {
            state.mode = BackendMode::RemoteRecovering;
        }
    }

    pub fn record_success(&self) {
        let mut state = self.lock_state();
        state.mode = BackendMode::RemotePrimary;
        state.last_success_at = Some(Instant::now());
        state.last_failure_at = None;
        state.last_failure_class = None;
        state.last_failure_message = None;
    }

    pub fn record_failure(&self, class: BackendFailureClass, message: impl Into<String>) {
        let mut state = self.lock_state();
        state.last_failure_at = Some(Instant::now());
        state.last_failure_class = Some(class);
        state.last_failure_message = Some(message.into());
        state.mode = match class {
            BackendFailureClass::Transient => BackendMode::LocalFallback,
            BackendFailureClass::Permanent => BackendMode::Offline,
        };
    }

    pub fn classify_error(error: &FemindError) -> BackendFailureClass {
        match error {
            FemindError::RemoteProfileMismatch(_)
            | FemindError::RemoteAuth(_)
            | FemindError::ModelNotAvailable(_) => BackendFailureClass::Permanent,
            FemindError::RemoteTimeout(_)
            | FemindError::RemoteTransport(_)
            | FemindError::RemoteUnavailable(_) => BackendFailureClass::Transient,
            _ => BackendFailureClass::Transient,
        }
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, BackendPolicyState> {
        match self.state.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}
