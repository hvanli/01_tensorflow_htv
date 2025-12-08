import os
import re
import threading

def install_tf_log_filter():
    # 1) Hide TF INFO logs, keep WARNING+
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    patterns = [
        r'numa_node',
        r'Unable to register cuFFT factory',
        r'Unable to register cuDNN factory',
        r'Unable to register cuBLAS factory',
        r'gpu_timer\.cc:114',
        r"\+ptx85' is not a recognized feature",
        r"Your kernel may have been built without NUMA support\.",
    ]
    compiled = [re.compile(p) for p in patterns]

    r_fd, w_fd = os.pipe()
    orig_stderr_fd = os.dup(2)
    os.dup2(w_fd, 2)

    def stderr_filter_thread():
        with os.fdopen(r_fd, 'r') as stream, os.fdopen(orig_stderr_fd, 'w') as real_stderr:
            for line in stream:
                if any(p.search(line) for p in compiled):
                    continue
                real_stderr.write(line)
                real_stderr.flush()

    threading.Thread(target=stderr_filter_thread, daemon=True).start()
    print("TF log filter installed.")