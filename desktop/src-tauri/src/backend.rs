//! Lifecycle of the packaged FastAPI backend: one child process per app run,
//! bound to 127.0.0.1 on a port chosen here, and killed when the app exits.
//! On Windows the child is placed in a kill-on-close Job Object, so it also
//! dies if the shell crashes. Only the process started here is ever stopped.

use std::ffi::OsString;
use std::fs::File;
use std::net::TcpListener;
use std::path::Path;
use std::process::{Child, Command, Stdio};

pub fn free_port() -> Result<u16, String> {
    let listener = TcpListener::bind("127.0.0.1:0").map_err(|e| format!("No free local port: {e}"))?;
    listener
        .local_addr()
        .map(|addr| addr.port())
        .map_err(|e| format!("No free local port: {e}"))
}

pub struct Backend {
    child: Child,
    #[cfg(windows)]
    _job: Option<job::Job>,
}

impl Backend {
    pub fn start(
        program: &Path,
        leading_args: &[OsString],
        port: u16,
        data_dir: &Path,
        logs_dir: &Path,
        api_key: Option<String>,
    ) -> Result<Self, String> {
        if !program.exists() {
            return Err(format!("The TactileGeo engine is missing: {}", program.display()));
        }
        let console = File::create(logs_dir.join("backend-console.log"))
            .map_err(|e| format!("Cannot write logs to {}: {e}", logs_dir.display()))?;
        let console_err = console.try_clone().map_err(|e| e.to_string())?;

        let mut command = Command::new(program);
        command
            .args(leading_args)
            .arg("--port")
            .arg(port.to_string())
            .arg("--data-dir")
            .arg(data_dir)
            .stdin(Stdio::null())
            .stdout(console)
            .stderr(console_err)
            .env_remove("OPENROUTER_API_KEY");
        if let Some(key) = api_key {
            command.env("OPENROUTER_API_KEY", key).env("MODEL_B_ENABLED", "true");
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x0800_0000;
            command.creation_flags(CREATE_NO_WINDOW);
        }

        let child = command
            .spawn()
            .map_err(|e| format!("Could not start the TactileGeo engine ({}): {e}", program.display()))?;
        #[cfg(windows)]
        let _job = match job::Job::kill_on_close_for(&child) {
            Ok(job) => Some(job),
            Err(error) => {
                eprintln!("Job Object unavailable, relying on exit handler: {error}");
                None
            }
        };
        Ok(Self {
            child,
            #[cfg(windows)]
            _job,
        })
    }

    /// `Some(description)` once the backend process has exited.
    pub fn exit_status(&mut self) -> Option<String> {
        match self.child.try_wait() {
            Ok(Some(status)) => Some(status.to_string()),
            Ok(None) => None,
            Err(error) => Some(error.to_string()),
        }
    }

    pub fn stop(&mut self) {
        if let Ok(None) = self.child.try_wait() {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(windows)]
mod job {
    use std::ffi::c_void;
    use std::os::windows::io::AsRawHandle;
    use std::process::Child;

    use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };

    pub struct Job(HANDLE);

    // The handle is only closed on drop; it is never used concurrently.
    unsafe impl Send for Job {}

    impl Job {
        pub fn kill_on_close_for(child: &Child) -> Result<Self, String> {
            unsafe {
                let handle = CreateJobObjectW(std::ptr::null(), std::ptr::null());
                if handle.is_null() {
                    return Err(format!("CreateJobObjectW failed ({})", GetLastError()));
                }
                let job = Job(handle);
                let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
                info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                if SetInformationJobObject(
                    job.0,
                    JobObjectExtendedLimitInformation,
                    &info as *const _ as *const c_void,
                    std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                ) == 0
                {
                    return Err(format!("SetInformationJobObject failed ({})", GetLastError()));
                }
                if AssignProcessToJobObject(job.0, child.as_raw_handle() as HANDLE) == 0 {
                    return Err(format!("AssignProcessToJobObject failed ({})", GetLastError()));
                }
                Ok(job)
            }
        }
    }

    impl Drop for Job {
        fn drop(&mut self) {
            unsafe {
                CloseHandle(self.0);
            }
        }
    }
}
