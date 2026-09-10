"""Reusable Plotly + ipywidgets analytics panels for a trained classifier.

Every build_*_panel function takes its inputs as explicit arguments (labels,
predictions, features, callables for fetching audio/spectrograms, etc.) and
returns an ipywidgets.VBox. Nothing here reads notebook globals, so any
notebook can use these panels as long as it supplies the same inputs — not
specific to this project's CNN or dataset classes.

None of these panels use go.FigureWidget. Newer Plotly ships FigureWidget on
top of the anywidget protocol, which some notebook front ends (VS Code's
Jupyter extension included) don't render reliably — the widget's controls show
up but the plot itself stays blank. Every figure here is instead shown inside
a plain ipywidgets.Output via fig.show(), and "live" panels (confusion matrix,
sample browser, saliency map) just clear and re-show that Output on every
interaction. Slightly more redraw work per update, but it only depends on the
same static rendering path a bare fig.show() in a cell already uses.
"""

import base64
import io
import itertools
import wave as _wave

import numpy as np
import torch
import ipywidgets as widgets
import plotly.graph_objects as go
import plotly.express as px
from IPython.display import display, HTML
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA

_PANEL_BORDER = "1px solid rgba(128, 128, 128, 0.4)"
_player_ids = itertools.count()


def _figure_output(fig):
    """An Output widget showing one Plotly figure, sized to fit its container.

    Output widgets carry a default max-height/overflow-y:auto style (meant for
    long text/stdout), which clips a figure and adds a scrollbar even when the
    figure itself fits fine — overflow="visible" turns that off. config=
    {"responsive": True} makes the figure resize to whatever width the Output
    actually has instead of Plotly's fixed default width, which is what forces
    a horizontal scrollbar when a panel sits in a narrow grid column.
    """
    out = widgets.Output(layout=widgets.Layout(overflow="visible"))
    with out:
        fig.show(config={"responsive": True})
    return out


def _redraw(output, fig):
    """Replaces an Output's figure in place, for panels that redraw on interaction."""
    output.clear_output(wait=True)
    with output:
        fig.show(config={"responsive": True})


def _panel(*children):
    return widgets.VBox(list(children))


def close_widget_tree(widget):
    """Recursively closes a widget and all its descendant widgets' comms, so the
    frontend actually tears down the old view instead of leaving it displayed
    alongside a freshly built replacement. Some notebook front ends (VS Code's
    Jupyter extension included) don't do this on their own just because a cell's
    output was cleared/replaced - see the audio player rebuild in
    build_sample_browser_panel below for the same issue in a smaller, local form.
    Call this on a previous build's root widget (e.g. a GridspecLayout) right
    before displaying a fresh one on cell re-run. Safe to call with None."""
    if widget is None:
        return
    for child in getattr(widget, "children", None) or []:
        close_widget_tree(child)
    widget.close()


def build_training_curves_panel(history):
    """history: dict with keys train_loss, val_loss, val_macro_f1 (one list per metric, one value per epoch)."""
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], name="train_loss"))
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], name="val_loss"))
    # macro F1 lives on its own right-hand axis (yaxis2) since it's a 0-1 score,
    # not a loss value, so sharing the left axis with loss would squash the scale
    fig.add_trace(go.Scatter(x=epochs, y=history["val_macro_f1"], name="val_macro_f1", yaxis="y2"))
    fig.update_layout(
        title="Training curves", xaxis_title="epoch",
        yaxis=dict(title="loss"),
        yaxis2=dict(title="macro F1", overlaying="y", side="right", range=[0, 1]),
        legend=dict(orientation="h"), height=340, margin=dict(t=40, b=30),
    )
    return _panel(_figure_output(fig))


def build_confusion_matrix_panel(y_true, y_pred, class_names):
    normalize_cb = widgets.Checkbox(value=False, description="Row-normalize")
    fig_output = widgets.Output(layout=widgets.Layout(overflow="visible"))

    def render(change=None):
        # change=None lets this be called directly (initial draw) as well as
        # from an ipywidgets .observe() callback, which always passes a change dict
        cm = confusion_matrix(y_true, y_pred)
        if normalize_cb.value:
            # row-normalize: each row (true class) sums to 1, showing the
            # percentage of that class's samples landing in each predicted bucket
            cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig = go.Figure(go.Heatmap(
            z=cm, x=class_names, y=class_names, colorscale="Blues",
            text=cm, texttemplate="%{text:.2f}" if normalize_cb.value else "%{text}",
            colorbar=dict(title="Share of row" if normalize_cb.value else "Clip count"),
        ))
        fig.update_layout(title="Confusion matrix", xaxis_title="Predicted class",
                           yaxis_title="Actual class", height=340, margin=dict(t=40, b=30))
        _redraw(fig_output, fig)

    # re-run render() every time the checkbox is toggled, so the heatmap
    # updates in place instead of the user needing to re-run the cell
    normalize_cb.observe(render, names="value")
    render()  # initial draw
    return _panel(normalize_cb, fig_output)


def build_per_class_panel(y_true, y_pred, class_names):
    # output_dict=True gives back {"Angry": {"precision": ..., "recall": ..., "f1-score": ...}, ...}
    # instead of the printable string version, so we can pull numbers out of it directly
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)

    fig = go.Figure()
    for metric in ["precision", "recall", "f1-score"]:
        fig.add_trace(go.Bar(name=metric, x=class_names, y=[report[label][metric] for label in class_names]))
    fig.update_layout(barmode="group", title="Per-class precision / recall / F1",
                       xaxis_title="Class", yaxis=dict(title="Score", range=[0, 1]),
                       height=340, margin=dict(t=40, b=30))
    return _panel(_figure_output(fig))


def _wav_data_uri(waveform, sample_rate):
    """Encodes a mono float waveform as a 16-bit PCM WAV data: URI, so the player
    below can embed it directly in a plain <audio> tag instead of going through
    IPython.display.Audio's own Output-based widget — that widget is what was
    accumulating duplicate players in some notebook front ends, since each
    render adds a brand-new one rather than actually replacing the last."""
    pcm16 = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with _wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm16.tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _waveform_player_html(waveform, sample_rate):
    """An <audio> element paired with a canvas drawing of the waveform and a
    playhead that tracks audio.currentTime while it plays, so you can see which
    part of the clip is making the sound you're hearing."""
    uri = _wav_data_uri(waveform, sample_rate)
    player_id = f"wf-player-{next(_player_ids)}"

    # downsample to a fixed bar count so canvas draw cost doesn't scale with clip length
    n_bars = 400
    chunks = np.array_split(waveform, n_bars)
    bars = [float(np.abs(c).max()) if len(c) else 0.0 for c in chunks]
    peak = max(bars) or 1.0
    bars = [round(b / peak, 4) for b in bars]

    return HTML(f"""
<div style="width:100%;">
  <canvas id="{player_id}-canvas" style="width:100%; height:70px; display:block;"></canvas>
  <audio id="{player_id}-audio" controls style="width:100%;" src="{uri}"></audio>
</div>
<script>
(function() {{
  const bars = {bars};
  const canvas = document.getElementById("{player_id}-canvas");
  const audio = document.getElementById("{player_id}-audio");
  if (!canvas || !audio) return;
  const ctx = canvas.getContext("2d");

  function draw() {{
    const w = canvas.clientWidth || 300;
    const h = canvas.clientHeight || 70;
    canvas.width = w;
    canvas.height = h;
    ctx.clearRect(0, 0, w, h);
    const barWidth = w / bars.length;
    const mid = h / 2;
    ctx.fillStyle = "#4c78a8";
    bars.forEach((v, i) => {{
      const barH = Math.max(1, v * (h - 4));
      ctx.fillRect(i * barWidth, mid - barH / 2, Math.max(1, barWidth - 1), barH);
    }});
    if (audio.duration) {{
      const x = (audio.currentTime / audio.duration) * w;
      ctx.strokeStyle = "#e45756";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }}
  }}

  let raf = null;
  function loop() {{
    draw();
    raf = (!audio.paused && !audio.ended) ? requestAnimationFrame(loop) : null;
  }}
  audio.addEventListener("play", () => {{ if (!raf) loop(); }});
  audio.addEventListener("pause", draw);
  audio.addEventListener("seeked", draw);
  audio.addEventListener("loadedmetadata", draw);
  new ResizeObserver(draw).observe(canvas);
  draw();
}})();
</script>
""")


def build_sample_browser_panel(class_names, y_true, y_pred, y_prob, get_waveform, get_mel, sample_rate, num_samples):
    """
    get_waveform(i) -> 1D np.ndarray audio for sample i
    get_mel(i)      -> 2D np.ndarray log-mel spectrogram for sample i
    """
    # --- controls ---
    filter_dd = widgets.Dropdown(options=["All"] + list(class_names) + ["Misclassified only"], value="All", description="Filter:")
    index_slider = widgets.IntSlider(min=0, max=0, description="Sample:")  # max gets set once we know the filtered count

    # --- output areas, one per thing we display ---
    info_label = widgets.HTML()
    mel_output = widgets.Output(layout=widgets.Layout(overflow="visible"))
    prob_output = widgets.Output(layout=widgets.Layout(overflow="visible"))
    # the audio player is rebuilt (not just cleared) on every render — see render() below
    audio_slot = widgets.VBox([widgets.Output()])

    label_to_idx = {name: i for i, name in enumerate(class_names)}
    y_true_arr, y_pred_arr = np.array(y_true), np.array(y_pred)

    def get_filtered_indices():
        """Returns the list of test-set row indices matching the current dropdown choice."""
        if filter_dd.value == "All":
            return list(range(num_samples))
        if filter_dd.value == "Misclassified only":
            return list(np.where(y_true_arr != y_pred_arr)[0])
        # otherwise the dropdown value is a class name — show only true examples of that class
        idx = label_to_idx[filter_dd.value]
        return list(np.where(y_true_arr == idx)[0])

    def render(change=None):
        indices = get_filtered_indices()
        if not indices:
            info_label.value = "No samples match this filter."
            return

        # clamp the slider to the new filtered list's size, then pick the sample it points at
        index_slider.max = len(indices) - 1
        i = indices[min(index_slider.value, len(indices) - 1)]

        waveform = get_waveform(i)
        mel = get_mel(i)

        info_label.value = f"<b>True:</b> {class_names[y_true[i]]} &nbsp; <b>Predicted:</b> {class_names[y_pred[i]]}"

        mel_fig = go.Figure(go.Heatmap(z=mel, showscale=False))
        mel_fig.update_layout(title="Log-mel spectrogram", xaxis_title="Time frame", yaxis_title="Mel bin",
                               height=160, margin=dict(t=30, b=10))
        _redraw(mel_output, mel_fig)

        prob_fig = go.Figure(go.Bar(x=list(class_names), y=y_prob[i]))
        prob_fig.update_layout(title="Predicted probabilities", xaxis_title="Class",
                                yaxis=dict(title="Probability", range=[0, 1]), height=160, margin=dict(t=30, b=10))
        _redraw(prob_output, prob_fig)

        # Clearing an Output's contents in place (even with a hard .outputs reset) wasn't
        # enough to stop the audio player from duplicating in some notebook front ends —
        # rebuild the Output widget itself each render and swap it into audio_slot, then
        # .close() the old one so its comm/view is actually torn down, not just its content.
        old_output = audio_slot.children[0]
        new_output = widgets.Output()
        with new_output:
            display(_waveform_player_html(waveform, sample_rate))
        audio_slot.children = (new_output,)
        old_output.close()

    # both the filter dropdown and the sample slider should trigger a re-render
    filter_dd.observe(render, names="value")
    index_slider.observe(render, names="value")
    render()  # initial draw
    return _panel(filter_dd, index_slider, info_label, mel_output, prob_output, audio_slot)


def build_duration_bias_panel(y_true, y_pred, durations, class_names):
    """Checks whether clip duration predicts correctness — a known confound in this
    kind of audio dataset (some classes are naturally shorter/longer on average)."""
    correct = np.array(y_true) == np.array(y_pred)
    fig = px.box(
        x=[class_names[i] for i in y_true], y=durations,
        color=np.where(correct, "correct", "incorrect"),
        labels={"x": "True label", "y": "duration (s)", "color": "prediction"},
        title="Clip duration vs. classification outcome",
    )
    fig.update_layout(height=340, margin=dict(t=40, b=30))
    return _panel(_figure_output(fig))


def _embedding_scatter_panel(embedding_2d, true_idx, pred_idx, class_names, title, axis_prefix):
    """Shared plotting code for the UMAP and PCA panels below — both just
    reduce features to 2D differently, then plot the same way."""
    fig = px.scatter(
        x=embedding_2d[:, 0], y=embedding_2d[:, 1],
        color=[class_names[i] for i in true_idx],
        hover_name=[f"pred: {class_names[p]}" for p in pred_idx],
        title=title,
        labels={"x": f"{axis_prefix} dimension 1", "y": f"{axis_prefix} dimension 2", "color": "True class"},
    )
    fig.update_layout(height=340, margin=dict(t=40, b=30))
    return _panel(_figure_output(fig))


def build_umap_panel(features, true_idx, pred_idx, class_names, random_state=42):
    """UMAP: nonlinear reduction, good at revealing local cluster structure."""
    import umap  # imported here, not at module top, since it's an optional/heavier dependency
    embedding_2d = umap.UMAP(n_components=2, random_state=random_state).fit_transform(features)
    return _embedding_scatter_panel(embedding_2d, true_idx, pred_idx, class_names, "UMAP of learned features", "UMAP")


def build_pca_panel(features, true_idx, pred_idx, class_names, random_state=42):
    """PCA: linear reduction, shows raw variance structure — a useful contrast to UMAP."""
    embedding_2d = PCA(n_components=2, random_state=random_state).fit_transform(features)
    return _embedding_scatter_panel(embedding_2d, true_idx, pred_idx, class_names, "PCA of learned features", "PCA")


class GradCAM:
    """Grad-CAM: highlights which part of a conv layer's input the model 'looked
    at' most to produce a given class score. Works for any PyTorch model — you
    just point it at the layer to explain."""

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None  # filled in by _save_activation on every forward pass
        self.gradients = None    # filled in by _save_gradient on every backward pass
        # forward/backward hooks fire every time target_layer processes a forward or
        # backward pass; handles are kept so remove_hooks() can unregister them later -
        # without that, re-creating a GradCAM on the same layer (e.g. a notebook cell
        # re-run) stacks another pair of hooks on top of the old ones forever
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        """Unregisters this instance's hooks from the model. Call on a previous
        GradCAM before creating a new one on the same layer (e.g. right before
        re-running the cell that builds grad_cam), so hooks don't silently
        accumulate across repeated runs."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __call__(self, x, class_idx):
        self.model.zero_grad()
        logits = self.model(x)
        # backprop from a single class's score (not the full loss) — this tells us
        # which input regions push *that specific class's* score up or down
        logits[0, class_idx].backward()

        # average each channel's gradient over its spatial dims -> one "importance
        # weight" per channel, then use those weights to combine the activation maps
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam / (cam.max() + 1e-8)  # normalize to [0, 1] for display
        return cam[0, 0].cpu().numpy()


def build_saliency_panel(grad_cam, class_names, y_true, y_pred, get_normalized_mel, to_input_tensor, num_samples):
    """
    grad_cam              -> a GradCAM instance already wired to your model/layer
    get_normalized_mel(i) -> 2D np.ndarray, the already-normalized mel spectrogram for sample i
    to_input_tensor(mel)  -> converts a 2D mel array into the model's expected input tensor
    """
    sample_dd = widgets.Dropdown(
        options=[(f"{i}: true={class_names[y_true[i]]} pred={class_names[y_pred[i]]}", i) for i in range(num_samples)],
        description="Sample:",
    )
    # lets you compare "what should it have looked at" (true class) vs
    # "what did it actually look at" (predicted class) for the same sample
    class_toggle = widgets.ToggleButtons(options=["Predicted class", "True class"], description="Explain:")
    fig_output = widgets.Output(layout=widgets.Layout(overflow="visible"))

    def render(change=None):
        i = sample_dd.value
        mel = get_normalized_mel(i)
        x = to_input_tensor(mel)
        target = int(y_pred[i]) if class_toggle.value == "Predicted class" else int(y_true[i])

        cam = grad_cam(x, target)
        # Grad-CAM's output is much smaller than the original spectrogram (it comes
        # from a downsampled conv layer), so stretch it back up to overlay correctly
        cam_resized = torch.nn.functional.interpolate(
            torch.from_numpy(cam)[None, None], size=mel.shape, mode="bilinear", align_corners=False
        )[0, 0].numpy()

        fig = go.Figure()
        fig.add_trace(go.Heatmap(z=mel, colorscale="Greys", showscale=False))       # base spectrogram
        fig.add_trace(go.Heatmap(
            z=cam_resized, colorscale="Jet", opacity=0.5, showscale=True,
            colorbar=dict(title="Model attention"),
        ))  # saliency overlay
        fig.update_layout(title=f"Saliency map — explaining '{class_names[target]}'",
                           xaxis_title="Time frame", yaxis_title="Mel bin",
                           height=340, margin=dict(t=40, b=30))
        _redraw(fig_output, fig)

    sample_dd.observe(render, names="value")
    class_toggle.observe(render, names="value")
    render()  # initial draw
    return _panel(sample_dd, class_toggle, fig_output)


def build_analytics_grid(panels):
    """panels: list of (row, col, widget) tuples — lets the caller lay out
    however many panels they want in whatever grid shape they want."""
    n_rows = max(r for r, c, _ in panels) + 1
    n_cols = max(c for r, c, _ in panels) + 1
    grid = widgets.GridspecLayout(n_rows, n_cols, height="1500px", grid_gap="20px")
    for r, c, panel in panels:
        # a simple line border + padding around each panel so the grid_gap reads as
        # separation between distinct cards instead of one undifferentiated block
        panel.layout.border = _PANEL_BORDER
        panel.layout.padding = "8px"
        panel.layout.overflow = "visible"
        grid[r, c] = panel
    return grid
