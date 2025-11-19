# app.py
# Streamlit UI — final with newline fix for terminal log, gap/time limit, assignments, pyarrow fallback
import io
import sys
import time
import zipfile
import tempfile
import subprocess
import threading
from pathlib import Path
from collections import deque
from queue import Queue, Empty

import pandas as pd
import streamlit as st
from ortools.sat.python import cp_model

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_HAS_API = True
try:
    from src.solver import build_solver_from_sources
except Exception:
    _HAS_API = False

st.set_page_config(page_title="Shift Solver — UI", layout="wide")
st.title("Shift Scheduling — UI")
st.caption("Modalità In‑process (parametri + assegnazioni) o CLI (log completo).")

# Sidebar
with st.sidebar:
    mode = st.radio("Modalità", ["In‑process", "CLI"], index=0 if _HAS_API else 1)
    st.header("Input")
    cfg_file = st.file_uploader("Config YAML", type=["yml","yaml"])
    data_zip = st.file_uploader("Dati (ZIP di CSV)", type=["zip"])
    if mode == "In‑process":
        st.header("Parametri solver")
        time_limit = st.number_input("Time limit (sec)", 0.0, 9999.0, 60.0, 5.0)
        gap = st.number_input("Gap relativo (0.0–1.0)", 0.0, 1.0, 0.0, 0.01)
        log_search = st.checkbox("Log ricerca OR-Tools", True)
    else:
        st.header("Opzioni CLI")
        extra_args = st.text_input("Argomenti extra CLI", "", help="Esempio: --seed 42")
    run_btn = st.button("Esegui solver", type="primary")

status_ph = st.empty()
log_ph = st.empty()
results_ph = st.container()

# ===== Utils =====
def _save_upload_to_tmp(uploaded, suffix):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded.read())
    tmp.flush(); tmp.close()
    return Path(tmp.name)

def _extract_zip_to_tmpdir(zip_path: Path) -> Path:
    out_dir = Path(tempfile.mkdtemp(prefix="data_"))
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    return out_dir

def _find_data_root(base: Path) -> Path:
    if (base / "employees.csv").exists():
        return base
    for p in base.rglob("employees.csv"):
        return p.parent
    for p in base.rglob("*.csv"):
        return p.parent
    return base

# ---- LOG RENDER (robust newline handling) ----
def _render_log(lines):
    # Join, then normalize CR/LF so every update appears on its own line
    text = "\n".join(str(ln) for ln in lines)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.endswith("\n"):
        text += "\n"
    # Use a code block (like before) for consistent monospace rendering
    log_ph.code(text, language="")

# ===== In‑process mode =====
if "log_buffer" not in st.session_state:
    st.session_state.log_buffer = deque(maxlen=20)

log_queue: "Queue[str]" = Queue()

class LiveSolutionCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, q): super().__init__(); self.q = q
    def OnSolutionCallback(self):
        obj = self.ObjectiveValue()
        bound = self.BestObjectiveBound()
        denom = max(1.0, abs(obj))
        gapv = 100.0 * abs(obj - bound) / denom
        self.q.put(f"Nuova soluzione: objective={obj:.6g}  bound={bound:.6g}  gap={gapv:.4g}%")

def _drain_log():
    moved = False
    while True:
        try: msg = log_queue.get_nowait()
        except Empty: break
        else:
            st.session_state.log_buffer.append(msg)
            moved = True
    if moved: _render_log(st.session_state.log_buffer)

def _build_assignments_df(solver, artifacts, bundle):
    x = artifacts.assign_vars
    emp_of = bundle.get("emp_of", {})
    slot_of = bundle.get("slot_of", {})
    slot_date = bundle.get("slot_date", {})
    slot_reparto = bundle.get("slot_reparto", {})
    slot_shiftcode = bundle.get("slot_shiftcode", {})
    rows = []
    for (e_idx, s_idx), var in x.items():
        if solver.Value(var) == 1:
            eid = emp_of.get(e_idx, e_idx)
            sid = slot_of.get(s_idx, s_idx)
            rows.append({
                "date": slot_date.get(sid),
                "reparto_id": slot_reparto.get(sid),
                "shift_code": slot_shiftcode.get(sid),
                "slot_id": sid,
                "employee_id": eid,
            })
    if not rows:
        return pd.DataFrame(columns=["date","reparto_id","shift_code","slot_id","employee_id"])
    df = pd.DataFrame(rows)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values(by=[c for c in ["date","reparto_id","shift_code","employee_id"] if c in df.columns]).reset_index(drop=True)

def _solve_inprocess(cfg_path, data_dir, time_limit_s, gap_rel, log_flag):
    from src.solver import build_solver_from_sources
    model, artifacts, context, bundle = build_solver_from_sources(str(cfg_path), str(data_dir))
    solver = cp_model.CpSolver()
    if time_limit_s > 0: solver.parameters.max_time_in_seconds = float(time_limit_s)
    if gap_rel > 0: solver.parameters.relative_gap_limit = float(gap_rel)
    solver.parameters.log_search_progress = log_flag
    solver.parameters.log_to_stdout = False
    cb = LiveSolutionCallback(log_queue)
    status = solver.Solve(model, cb)
    return solver, status, artifacts, bundle

# ===== CLI mode =====
def _reader_thread(proc, q):
    # Read raw stdout (line buffered on \n), but forward chunks as-is.
    for raw in iter(proc.stdout.readline, b""):
        chunk = raw.decode("utf-8", errors="replace")
        q.put(chunk)
    proc.stdout.close()

def _run_cli(cfg_path, data_dir, tokens):
    cmd = [sys.executable, "-u", "run_solver.py", "--config", str(cfg_path), "--data-dir", str(data_dir)]
    cmd.extend(tokens)
    return cmd

# ===== Run =====
if run_btn:
    if cfg_file is None or data_zip is None:
        st.error("Carica config e dati.")
        st.stop()

    cfg_path = _save_upload_to_tmp(cfg_file, ".yaml")
    data_dir = _find_data_root(_extract_zip_to_tmpdir(_save_upload_to_tmp(data_zip, ".zip")))
    st.caption(f"Dati rilevati in: `{data_dir}`")

    if mode == "In‑process":
        status_ph.info("Esecuzione solver in‑process…")
        st.session_state.log_buffer.clear()
        _render_log(["(log soluzione in tempo reale)"])
        result_box = {}
        def _worker():
            try:
                solver, status, artifacts, bundle = _solve_inprocess(cfg_path, data_dir, time_limit, gap, log_search)
                result_box.update(dict(solver=solver, status=status, artifacts=artifacts, bundle=bundle))
            except Exception as e:
                log_queue.put(f"Errore: {e}")
                result_box["error"] = str(e)
        t = threading.Thread(target=_worker, daemon=True); t.start()
        while t.is_alive(): _drain_log(); time.sleep(0.15)
        _drain_log()
        if "error" in result_box:
            status_ph.error(result_box["error"])
        else:
            solver = result_box["solver"]; artifacts = result_box["artifacts"]; bundle = result_box["bundle"]
            st.success("Solver completato.")
            try:
                st.metric("Objective", f"{solver.ObjectiveValue():.6g}")
                st.metric("Bound", f"{solver.BestObjectiveBound():.6g}")
            except Exception: pass
            st.subheader("Assegnazioni trovate")
            try:
                assign_df = _build_assignments_df(solver, artifacts, bundle)
                if assign_df.empty:
                    st.info("Nessuna assegnazione trovata.")
                else:
                    try:
                        st.dataframe(assign_df, use_container_width=True)
                    except ModuleNotFoundError:
                        st.warning("⚠️ PyArrow non installato: impossibile mostrare la tabella.\nInstalla con: pip install pyarrow")
                        st.text(assign_df.head().to_string())
                    csv_bytes = assign_df.to_csv(index=False).encode("utf-8")
                    st.download_button("Scarica assegnazioni (CSV)", data=csv_bytes, file_name="assignments.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Errore mostrando assegnazioni: {e}")

    else:  # CLI
        status_ph.info("Esecuzione solver (CLI)…")
        buffer = deque(maxlen=20)
        _render_log(["(log CLI in tempo reale)"])
        tokens = extra_args.strip().split() if extra_args.strip() else []
        cmd = _run_cli(cfg_path, data_dir, tokens)
        st.caption("Comando: `" + " ".join(cmd) + "`")
        try:
            proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
        except FileNotFoundError:
            status_ph.error("Non trovo run_solver.py")
            st.stop()
        q = Queue()
        t = threading.Thread(target=_reader_thread, args=(proc,q), daemon=True); t.start()
        last_solution = None
        # We receive 'chunk's (may contain multiple lines). Normalize CR/LF and split into lines.
        while True:
            if proc.poll() is not None and q.empty(): break
            try: chunk = q.get(timeout=0.1)
            except Empty: time.sleep(0.05); continue
            norm = chunk.replace("\r\n","\n").replace("\r","\n")
            for line in norm.split("\n"):
                if line == "":  # skip empty lines caused by split
                    continue
                buffer.append(line)
                if "Nuova soluzione:" in line: last_solution = line
            _render_log(buffer)
        exit_code = proc.wait()
        if exit_code == 0: status_ph.success("Solver terminato (exit 0)")
        else: status_ph.error(f"Exit {exit_code}")
        if last_solution: st.write(f"**Ultima soluzione:** {last_solution}")
        else: st.caption("Nessuna riga 'Nuova soluzione' trovata.")
