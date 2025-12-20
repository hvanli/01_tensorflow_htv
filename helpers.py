from IPython.display import HTML

def display_tensorboard_logs(url: str = "http://localhost:6006"):
    """
    Display a clickable TensorBoard link inside Jupyter.
    """
    return HTML(f'<a href="{url}" target="_blank">🚀 Open TensorBoard</a>')
