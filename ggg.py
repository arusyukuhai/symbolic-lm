import numpy as np
import matplotlib.pyplot as plt

def slog_a(x, a=2.0, eps=1e-12):
    """
    Plotting-oriented continuous-ish super-logarithm approximation for x>0.
    - For x in [1, a]: slog = log_a(x)
    - For x > a: slog(x) = 1 + slog(log_a(x))
    - For x < 1: extend by inverse rule slog(x) = slog(a^x) - 1 style is nontrivial;
      here we support (0,1) by symmetric recursion using exponentiation:
        slog(x) = -1 + slog(a**x)  until it lands in [1,a].
      (This is a pragmatic extension; if you need strict Kneser slog, say so.)
    """
    x = np.asarray(x, dtype=np.float64)

    if np.any(x <= 0):
        raise ValueError("slog_a is defined here only for x>0.")

    la = np.log(a)

    def _slog_scalar(v):
        # handle (0,1) by pushing upward with a**v (inverse of log_a)
        s = 0.0
        vv = float(v)

        # push up if vv < 1
        while vv < 1.0 - eps:
            vv = a ** vv
            s -= 1.0
            # safety: vv approaches 1 quickly for typical a in (1, e^(1/e)] range
            if s < -1000:
                break

        # pull down if vv > a
        while vv > a + eps:
            vv = np.log(vv) / la  # log_a(vv)
            s += 1.0
            if s > 1000:
                break

        # now vv ~ in [1, a]
        return s + (np.log(vv) / la)

    vec = np.vectorize(_slog_scalar, otypes=[np.float64])
    return vec(x)

def sloglog_plot(x, y, a=2.0, *, ax=None, **plot_kwargs):
    """
    Make a 'both super-logarithmic' plot:
      X-axis: slog_a(x)
      Y-axis: slog_a(y)
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    sx = slog_a(x, a=a)
    sy = slog_a(y, a=a)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(sx, sy, **plot_kwargs)
    ax.set_xlabel(f"slog_{a}(x)")
    ax.set_ylabel(f"slog_{a}(y)")
    ax.grid(True)
    return ax

# ---- example ----
if __name__ == "__main__":
    a = 1.5
    x = np.linspace(1, 2048, 600)     # 1 .. 1e6
    y = 1.25**x                       # example relationship
    ax = sloglog_plot(x, y, a=a, linewidth=2)
    plt.show()
