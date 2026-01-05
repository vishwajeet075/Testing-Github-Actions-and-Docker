import streamlit as st
import numpy as np
import pandas as pd
import threading, json, os
from http.server import HTTPServer, BaseHTTPRequestHandler

# File: /f:/small_python_app/main.py
import plotly.express as px

st.set_page_config(page_title="Mini Visualizer", layout="wide")
st.title("Mini Streamlit Visualizer")

dataset = st.sidebar.selectbox("Choose dataset", ["Sine wave", "Random walk", "Scatter clusters"])
n = st.sidebar.slider("Number of points", 50, 2000, 500)
noise = st.sidebar.slider("Noise (stddev)", 0.0, 2.0, 0.5, step=0.1)

if dataset == "Sine wave":
    x = np.linspace(0, 4 * np.pi, n)
    y = np.sin(x) + np.random.normal(scale=noise, size=n)
    df = pd.DataFrame({"x": x, "y": y})
    fig = px.line(df, x="x", y="y", title="Noisy Sine Wave")
    st.plotly_chart(fig, use_container_width=True)

    # lightweight health endpoint for CI/CD checks

    if not st.session_state.get("_health_server_started"):
        port = int(os.environ.get("HEALTH_PORT", "8000"))

        class _HealthHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def do_GET(self):
                if self.path.startswith("/health"):
                    payload = {
                        "status": "ok",
                        "dataset": dataset,
                        "rows": len(df),
                        "columns": len(df.columns),
                    }
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(payload).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

        def _run():
            try:
                HTTPServer(("0.0.0.0", port), _HealthHandler).serve_forever()
            except OSError:
                pass

        threading.Thread(target=_run, daemon=True).start()
        st.session_state["_health_server_started"] = True
        st.sidebar.markdown(f"Health: http://localhost:{port}/health")

elif dataset == "Random walk":
    steps = np.random.normal(scale=noise + 0.1, size=n)
    y = np.cumsum(steps)
    x = np.arange(n)
    df = pd.DataFrame({"step": x, "position": y})
    fig = px.line(df, x="step", y="position", title="Random Walk")
    st.plotly_chart(fig, use_container_width=True)

else:  # Scatter clusters
    k = st.sidebar.slider("Clusters", 1, 6, 3)
    points_per = max(10, n // k)
    xs, ys, labels = [], [], []
    rng = np.random.default_rng(42)
    centers = rng.uniform(-10, 10, size=(k, 2))
    for i, (cx, cy) in enumerate(centers):
        xs.append(rng.normal(cx, scale=1.0 + noise, size=points_per))
        ys.append(rng.normal(cy, scale=1.0 + noise, size=points_per))
        labels += [f"cluster_{i}"] * points_per
    xs = np.concatenate(xs)[:n]
    ys = np.concatenate(ys)[:n]
    labels = labels[:n]
    df = pd.DataFrame({"x": xs, "y": ys, "cluster": labels})
    fig = px.scatter(df, x="x", y="y", color="cluster", title="Scatter Clusters", width=900, height=500)
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Show data (first 100 rows)"):
    st.dataframe(df.head(100))

col1, col2 = st.columns(2)
with col1:
    st.metric("Rows", len(df))
with col2:
    st.metric("Columns", len(df.columns))

st.markdown("---")
st.subheader("Distribution")
if "y" in df.columns:
    hist_col = "y"
elif "position" in df.columns:
    hist_col = "position"
else:
    hist_col = "x"
fig_hist = px.histogram(df, x=hist_col, nbins=30, title=f"Histogram of {hist_col}")
st.plotly_chart(fig_hist, use_container_width=True)