#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod backend;

use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::Mutex;

use serde_json::json;
use tauri::menu::{AboutMetadata, Menu, MenuItem, PredefinedMenuItem, Submenu};
use tauri::{AppHandle, Manager, RunEvent, State, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_dialog::DialogExt;

use backend::Backend;

const KEYRING_SERVICE: &str = "TactileGeo";
const KEYRING_USER: &str = "openrouter-api-key";

struct AppState {
    backend: Mutex<Option<Backend>>,
    logs_dir: PathBuf,
}

fn key_entry() -> Result<keyring::Entry, String> {
    keyring::Entry::new(KEYRING_SERVICE, KEYRING_USER).map_err(|e| e.to_string())
}

fn stored_api_key() -> Option<String> {
    key_entry()
        .ok()?
        .get_password()
        .ok()
        .map(|key| key.trim().to_string())
        .filter(|key| !key.is_empty())
}

#[tauri::command]
fn model_b_key_status() -> bool {
    stored_api_key().is_some()
}

#[tauri::command]
fn set_model_b_key(key: String) -> Result<(), String> {
    let key = key.trim();
    if key.is_empty() {
        return Err("Enter an OpenRouter API key.".into());
    }
    key_entry()?.set_password(key).map_err(|e| e.to_string())
}

#[tauri::command]
fn clear_model_b_key() -> Result<(), String> {
    match key_entry()?.delete_credential() {
        Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
        Err(error) => Err(error.to_string()),
    }
}

#[tauri::command]
fn backend_exit_status(state: State<'_, AppState>) -> Option<String> {
    state.backend.lock().ok()?.as_mut()?.exit_status()
}

#[tauri::command]
fn open_logs_folder(state: State<'_, AppState>) -> Result<(), String> {
    open_folder(&state.logs_dir)
}

#[tauri::command]
async fn save_svg(app: AppHandle, contents: String, default_name: String) -> Result<Option<String>, String> {
    let picked = app
        .dialog()
        .file()
        .set_title("Save tactile SVG")
        .set_file_name(&default_name)
        .add_filter("SVG image", &["svg"])
        .blocking_save_file();
    let Some(picked) = picked else { return Ok(None) };
    let path = picked.into_path().map_err(|e| e.to_string())?;
    std::fs::write(&path, contents).map_err(|e| format!("Could not save {}: {e}", path.display()))?;
    Ok(Some(path.display().to_string()))
}

fn open_folder(path: &PathBuf) -> Result<(), String> {
    #[cfg(windows)]
    let program = "explorer";
    #[cfg(target_os = "macos")]
    let program = "open";
    #[cfg(all(unix, not(target_os = "macos")))]
    let program = "xdg-open";
    std::process::Command::new(program)
        .arg(path)
        .spawn()
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// The packaged engine, or `TACTILEGEO_DEV_PYTHON` + backend/desktop_main.py in development.
fn backend_command(app: &AppHandle) -> Result<(PathBuf, Vec<OsString>), String> {
    if let Some(python) = std::env::var_os("TACTILEGEO_DEV_PYTHON") {
        let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../backend/desktop_main.py");
        return Ok((PathBuf::from(python), vec![script.into_os_string()]));
    }
    let resources = app.path().resource_dir().map_err(|e| e.to_string())?;
    let exe = if cfg!(windows) { "tactilegeo-backend.exe" } else { "tactilegeo-backend" };
    Ok((resources.join("backend").join(exe), Vec::new()))
}

fn start_backend(app: &AppHandle, logs_dir: &PathBuf) -> Result<(Backend, u16), String> {
    let data_dir = app.path().app_local_data_dir().map_err(|e| e.to_string())?;
    let (program, leading_args) = backend_command(app)?;
    let port = backend::free_port()?;
    let backend = Backend::start(&program, &leading_args, port, &data_dir, logs_dir, stored_api_key())?;
    Ok((backend, port))
}

fn build_menu(app: &AppHandle) -> tauri::Result<Menu<tauri::Wry>> {
    let about = AboutMetadata {
        name: Some("TactileGeo".into()),
        version: Some(app.package_info().version.to_string()),
        comments: Some("Turns maths worksheets into tactile diagrams. Model A runs on this computer; Model B (optional) needs internet and your own OpenRouter key.".into()),
        ..Default::default()
    };
    let help = Submenu::with_items(
        app,
        "Help",
        true,
        &[
            &MenuItem::with_id(app, "open_logs", "Open logs folder", true, None::<&str>)?,
            &PredefinedMenuItem::separator(app)?,
            &PredefinedMenuItem::about(app, Some("About TactileGeo"), Some(about))?,
        ],
    )?;
    let file = Submenu::with_items(app, "File", true, &[&PredefinedMenuItem::quit(app, Some("Exit"))?])?;
    Menu::with_items(app, &[&file, &help])
}

fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            model_b_key_status,
            set_model_b_key,
            clear_model_b_key,
            backend_exit_status,
            open_logs_folder,
            save_svg
        ])
        .setup(|app| {
            let handle = app.handle().clone();
            let logs_dir = handle.path().app_local_data_dir()?.join("logs");
            std::fs::create_dir_all(&logs_dir)?;

            let (backend, config) = match start_backend(&handle, &logs_dir) {
                Ok((backend, port)) => (Some(backend), json!({ "apiBase": format!("http://127.0.0.1:{port}"), "startupError": null })),
                Err(error) => (None, json!({ "apiBase": "", "startupError": error })),
            };
            let mut config = config;
            config["logsDir"] = json!(logs_dir.display().to_string());
            config["version"] = json!(handle.package_info().version.to_string());
            app.manage(AppState { backend: Mutex::new(backend), logs_dir });

            let menu = build_menu(&handle)?;
            WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
                .title("TactileGeo")
                .inner_size(1280.0, 860.0)
                .min_inner_size(900.0, 640.0)
                .menu(menu)
                .initialization_script(format!("window.__TACTILEGEO_DESKTOP__ = Object.freeze({config});"))
                .build()?;
            Ok(())
        })
        .on_menu_event(|app, event| {
            if event.id() == "open_logs" {
                let state = app.state::<AppState>();
                let _ = open_folder(&state.logs_dir);
            }
        })
        .build(tauri::generate_context!())
        .expect("failed to build TactileGeo")
        .run(|app, event| {
            if let RunEvent::Exit = event {
                if let Some(state) = app.try_state::<AppState>() {
                    if let Ok(mut backend) = state.backend.lock() {
                        if let Some(mut running) = backend.take() {
                            running.stop();
                        }
                    }
                }
            }
        });
}
