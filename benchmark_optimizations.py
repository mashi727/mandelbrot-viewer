"""マンデルブロ集合計算の最適化比較ベンチマーク"""
import numpy as np
import numba
import time
import os

os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

# 現在の実装
@numba.njit(parallel=True)
def mandelbrot_current(c_real, c_imag, n_max):
    """現在の実装"""
    height = len(c_imag)
    width = len(c_real)
    z = np.zeros((height, width))

    for i in numba.prange(width):
        for j in range(height):
            c = complex(c_real[i], c_imag[j])
            z0 = 0j
            n = 0
            while abs(z0) <= 2.0 and n < n_max:
                z0 = z0**2 + c
                n += 1
            z[j, i] = n if n < n_max else 0
    return z[::-1]


# 最適化版1: べき乗→乗算、abs→実数演算
@numba.njit(parallel=True)
def mandelbrot_opt1(c_real, c_imag, n_max):
    """最適化版1: べき乗と絶対値の最適化"""
    height = len(c_imag)
    width = len(c_real)
    z = np.zeros((height, width))

    for i in numba.prange(width):
        for j in range(height):
            cr = c_real[i]
            ci = c_imag[j]
            zr = 0.0
            zi = 0.0
            n = 0

            while zr*zr + zi*zi <= 4.0 and n < n_max:
                # z = z*z + c を実数・虚数部で展開
                zr_new = zr*zr - zi*zi + cr
                zi = 2.0*zr*zi + ci
                zr = zr_new
                n += 1

            z[j, i] = n if n < n_max else 0
    return z[::-1]


# 最適化版2: 早期脱出条件の追加
@numba.njit(parallel=True)
def mandelbrot_opt2(c_real, c_imag, n_max):
    """最適化版2: 早期脱出条件の追加"""
    height = len(c_imag)
    width = len(c_real)
    z = np.zeros((height, width))

    for i in numba.prange(width):
        for j in range(height):
            cr = c_real[i]
            ci = c_imag[j]

            # 主カーディオイド判定（最大の領域）
            q = (cr - 0.25)**2 + ci**2
            if q * (q + (cr - 0.25)) <= 0.25 * ci**2:
                z[j, i] = 0
                continue

            # 周期2バルブ判定
            if (cr + 1.0)**2 + ci**2 <= 0.0625:
                z[j, i] = 0
                continue

            zr = 0.0
            zi = 0.0
            n = 0

            while zr*zr + zi*zi <= 4.0 and n < n_max:
                zr_new = zr*zr - zi*zi + cr
                zi = 2.0*zr*zi + ci
                zr = zr_new
                n += 1

            z[j, i] = n if n < n_max else 0
    return z[::-1]


# 最適化版3: キャッシュ効率の改善（ループ順序変更）
@numba.njit(parallel=True)
def mandelbrot_opt3(c_real, c_imag, n_max):
    """最適化版3: キャッシュ効率の改善"""
    height = len(c_imag)
    width = len(c_real)
    z = np.zeros((height, width))

    for j in numba.prange(height):
        ci = c_imag[j]
        for i in range(width):
            cr = c_real[i]

            # 早期脱出判定
            q = (cr - 0.25)**2 + ci**2
            if q * (q + (cr - 0.25)) <= 0.25 * ci**2:
                z[j, i] = 0
                continue

            if (cr + 1.0)**2 + ci**2 <= 0.0625:
                z[j, i] = 0
                continue

            zr = 0.0
            zi = 0.0
            n = 0

            while zr*zr + zi*zi <= 4.0 and n < n_max:
                zr_new = zr*zr - zi*zi + cr
                zi = 2.0*zr*zi + ci
                zr = zr_new
                n += 1

            z[j, i] = n if n < n_max else 0
    return z[::-1]


def benchmark(func, name, c_real, c_imag, n_max, iterations=5):
    """ベンチマーク実行"""
    # ウォームアップ
    _ = func(c_real[:100], c_imag[:100], 50)

    # 計測
    times = []
    for _ in range(iterations):
        start = time.time()
        result = func(c_real, c_imag, n_max)
        elapsed = time.time() - start
        times.append(elapsed)

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"{name:20s}: {avg_time:.4f}s ± {std_time:.4f}s  "
          f"({result.size / avg_time / 1e6:.1f} M pixels/sec)")

    return avg_time, result


def main():
    print("=" * 70)
    print("マンデルブロ集合計算の最適化ベンチマーク")
    print("=" * 70)

    # テストパラメータ
    resolutions = [
        (1000, "1000x1000 (100万ピクセル)"),
        (2000, "2000x2000 (400万ピクセル)"),
    ]

    n_max = 170

    for res, desc in resolutions:
        print(f"\n【{desc}】")
        print("-" * 70)

        c_real = np.linspace(-2.0, 1.0, res)
        c_imag = np.linspace(-1.5, 1.5, res)

        time_current, result_current = benchmark(
            mandelbrot_current, "現在の実装", c_real, c_imag, n_max
        )

        time_opt1, result_opt1 = benchmark(
            mandelbrot_opt1, "最適化版1 (演算)", c_real, c_imag, n_max
        )

        time_opt2, result_opt2 = benchmark(
            mandelbrot_opt2, "最適化版2 (早期脱出)", c_real, c_imag, n_max
        )

        time_opt3, result_opt3 = benchmark(
            mandelbrot_opt3, "最適化版3 (キャッシュ)", c_real, c_imag, n_max
        )

        print(f"\n高速化率:")
        print(f"  最適化版1: {time_current / time_opt1:.2f}x")
        print(f"  最適化版2: {time_current / time_opt2:.2f}x")
        print(f"  最適化版3: {time_current / time_opt3:.2f}x")

        # 結果の一致確認
        print(f"\n結果の一致確認:")
        print(f"  最適化版1: {np.allclose(result_current, result_opt1)}")
        print(f"  最適化版2: {np.allclose(result_current, result_opt2)}")
        print(f"  最適化版3: {np.allclose(result_current, result_opt3)}")


if __name__ == "__main__":
    main()
