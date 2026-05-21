"""Interactive Mandelbrot set viewer / マンデルブロ集合の対話的可視化ビューア

3 つの連動するプロットで、マンデルブロ集合の詳細を段階的に拡大表示する。
Three linked plots progressively zoom into the Mandelbrot set:

- GW1 (左上 / top-left)   : 全体表示 + ROI1 / overview with ROI1
- GW2 (左下 / bottom-left): ROI1 の領域を拡大 + ROI2 / ROI1 region, with ROI2
- GW3 (右 / right)        : ROI2 の領域を拡大 (マウスでズーム/パン可)
                            ROI2 region (mouse zoom / pan enabled)

ROI (Region of Interest) を移動・リサイズすると下流のプロットが即座に再計算される。
集合の計算は Numba で JIT/並列化し、座標は ``decimal.Decimal`` で高精度に保持する。
"""

import os
import sys
from dataclasses import dataclass
from decimal import Decimal, getcontext

import numpy as np
import pyqtgraph as pg
from PySide6 import QtGui
from PySide6.QtWidgets import QApplication, QMainWindow

from imgPlotUinoDock2 import Ui_MainWindow

# 座標を高精度で扱うため Decimal の精度を 50 桁に設定する。
getcontext().prec = 50

# Numba の並列バックエンドに workqueue を強制する（OpenMP/TBB 非依存で移植性が高い）。
# import numba より前に、かつ既存の環境変数を上書きする形で設定する必要がある。
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"
import numba  # noqa: E402  (環境変数を設定してから import する)


def setprop(plt):
    """プロットの軸・グリッド・凡例など共通の見た目を設定する。"""
    font_css = {"font-family": "Arial, Meiryo", "font-size": "24pt"}
    plt.getAxis("left").setWidth(80)
    plt.showGrid(x=True, y=True)
    plt.getAxis("top").setLabel(**font_css)
    plt.getAxis("left").setStyle(tickFont=QtGui.QFont("Arial", 18))
    plt.getAxis("bottom").setStyle(tickFont=QtGui.QFont("Arial", 18))
    plt.getAxis("right").setStyle(tickFont=QtGui.QFont("Arial", 18))
    plt.getAxis("top").setStyle(tickFont=QtGui.QFont("Arial", 18))
    plt.setAutoVisible(y=True)
    plt.showAxis("left")
    plt.showAxis("bottom")
    plt.showAxis("top")
    plt.showAxis("right")
    plt.addLegend(offset=(0, 0))


@dataclass
class Data:
    """アプリ全体の状態を保持する（座標は Decimal 型で高精度管理）。"""

    initFlag: str = "initialPlot"
    default_reso: str = "500"

    # GW1（メインウィンドウ：全体表示）
    h11: Decimal = Decimal("-2.0")
    h12: Decimal = Decimal("1.0")
    v11: Decimal = Decimal("-1.5")
    v12: Decimal = Decimal("1.5")
    res1: str = "2000"
    n_max1: str = "170"
    cmap1: str = "CET_D1"
    roi1h: Decimal = Decimal("-2.0")
    roi1v: Decimal = Decimal("-1.5")
    roi1dh: Decimal = Decimal("3.0")
    roi1dv: Decimal = Decimal("3.0")

    # GW2（ROI1 の拡大）
    res2: str = "500"
    n_max2: str = "120"
    cmap2: str = "CET_D1"
    h21: Decimal = Decimal("-2.0")
    h22: Decimal = Decimal("1.0")
    v21: Decimal = Decimal("-1.5")
    v22: Decimal = Decimal("1.5")
    roi2h: Decimal = Decimal("-0.8")
    roi2v: Decimal = Decimal("-0.3")
    roi2dh: Decimal = Decimal("0.6")
    roi2dv: Decimal = Decimal("0.6")

    # GW3（ROI2 の拡大）
    default_reso3: str = "2000"
    n_max3: str = "120"
    cmap3: str = "CET_D1"
    res3: str = "2000"
    h31: Decimal = Decimal("-0.8")
    h32: Decimal = Decimal("-0.2")
    v31: Decimal = Decimal("-0.3")
    v32: Decimal = Decimal("0.3")


def scale_factor(h1, h2, v1, v2, reso):
    """ピクセル→複素平面のスケール係数 (dx, dy) を返す。"""
    return (h2 - h1) / reso, (v2 - v1) / reso


def translate_factor(h1, h2, v1, v2, reso):
    """画像の平行移動係数 (tx, ty) を返す。"""
    return h1 * reso / np.abs(h2 - h1), v2 * reso / np.abs(v2 - v1)


@numba.njit(parallel=True)
def mandelbrot(c_real, c_imag, n_max):
    """マンデルブロ集合の脱出反復回数を計算する（Numba 並列・最適化版）。

    最適化のポイント:
    1. 複素数演算を実数演算に分解
    2. 主カーディオイド／周期 2 バルブの早期脱出判定
    3. キャッシュ効率を意識したループ順序（行優先）
    """
    height = len(c_imag)
    width = len(c_real)
    z = np.zeros((height, width))

    for j in numba.prange(height):
        ci = c_imag[j]
        for i in range(width):
            cr = c_real[i]

            # 早期脱出 1: 主カーディオイド（集合の約 75%）
            q = (cr - 0.25) ** 2 + ci**2
            if q * (q + (cr - 0.25)) <= 0.25 * ci**2:
                z[j, i] = 0
                continue

            # 早期脱出 2: 周期 2 バルブ（集合の約 20%）
            if (cr + 1.0) ** 2 + ci**2 <= 0.0625:
                z[j, i] = 0
                continue

            # 反復計算（実数演算）
            zr = 0.0
            zi = 0.0
            n = 0
            while zr * zr + zi * zi <= 4.0 and n < n_max:
                zr_new = zr * zr - zi * zi + cr
                zi = 2.0 * zr * zi + ci
                zr = zr_new
                n += 1

            z[j, i] = n if n < n_max else 0
    return z[::-1]


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.show()
        self.init_ui()

        # ROI とプロットウィジェットを保持するインスタンス変数
        self.gw1 = None
        self.gw2 = None
        self.gw3 = None
        self.roi1 = None
        self.roi2 = None
        self.img1 = None
        self.img2 = None
        self.img3 = None

        # 初期値設定
        self.doubleSpinBox_h11.setValue(float(Data.h11))
        self.doubleSpinBox_h12.setValue(float(Data.h12))
        self.doubleSpinBox_v11.setValue(float(Data.v11))
        self.doubleSpinBox_v12.setValue(float(Data.v12))

        self.lineEdit_res2.setText(Data.default_reso)
        self.lineEdit_res3.setText(Data.default_reso3)
        self.label_Res1.setText(Data.res1)

        self.lineEdit_n_max1.setText(Data.n_max1)
        self.lineEdit_n_max2.setText(Data.n_max2)
        self.lineEdit_n_max3.setText(Data.n_max3)

        self.cmap1.setItemText(0, Data.cmap1)
        self.cmap2.setItemText(0, Data.cmap2)
        self.cmap3.setItemText(0, Data.cmap3)

        # UI 全体のフォントサイズ
        font = QtGui.QFont()
        font.setPointSize(20)
        self.setFont(font)

        # ステータスバーのフォント
        status_font = QtGui.QFont()
        status_font.setPointSize(20)
        status_font.setBold(True)
        self.statusBar().setFont(status_font)
        self.statusBar().setStyleSheet(
            "QStatusBar { font-size: 20pt; font-weight: bold; padding: 5px; }"
        )

        # 初回プロット
        self.setup_plots()

        # シグナル接続（パラメータ変更で該当プロットを再描画）
        self.lineEdit_res2.returnPressed.connect(self.update_gw2)
        self.lineEdit_res3.returnPressed.connect(self.update_gw3)
        self.lineEdit_n_max1.returnPressed.connect(self.update_gw1)
        self.lineEdit_n_max2.returnPressed.connect(self.update_gw2)
        self.lineEdit_n_max3.returnPressed.connect(self.update_gw3)
        self.cmap1.currentTextChanged.connect(self.update_gw1)
        self.cmap2.currentTextChanged.connect(self.update_gw2)
        self.cmap3.currentTextChanged.connect(self.update_gw3)

    def mandelresult(self, h1, h2, v1, v2, resolution, n_max):
        """指定範囲・解像度でマンデルブロ集合を計算する（Decimal→float に変換）。"""
        c_real = np.linspace(float(h1), float(h2), resolution)
        c_imag = np.linspace(float(v1), float(v2), resolution)
        return mandelbrot(c_real, c_imag, n_max)

    def init_ui(self):
        self.setGeometry(10, 10, 1620, 1100)

    def setup_plots(self):
        """初回のみ実行：全プロットウィンドウと ROI をセットアップする。"""
        # GW1
        self.graphicsLayoutWidget_gw1.clear()
        self.gw1 = self.graphicsLayoutWidget_gw1.addPlot(row=0, col=0)

        self.img1 = pg.ImageItem()
        self.gw1.addItem(self.img1)

        self.roi1 = pg.RectROI(
            [float(Data.roi1h), float(Data.roi1v)],
            [float(Data.roi1dh), float(Data.roi1dv)],
            pen=pg.mkPen("g", width=2),
            hoverPen=pg.mkPen("g", width=2),
        )
        self.roi1.addScaleHandle([1, 0], [0, 1], lockAspect=True)
        self.gw1.addItem(self.roi1)
        self.gw1.setAspectLocked()
        setprop(self.gw1)

        # GW2
        self.graphicsLayoutWidget_gw2.clear()
        self.gw2 = self.graphicsLayoutWidget_gw2.addPlot(row=0, col=0)

        self.img2 = pg.ImageItem()
        self.gw2.addItem(self.img2)

        self.roi2 = pg.RectROI(
            [float(Data.roi2h), float(Data.roi2v)],
            [float(Data.roi2dh), float(Data.roi2dv)],
            pen=pg.mkPen("g", width=2),
            hoverPen=pg.mkPen("g", width=2),
        )
        self.roi2.addScaleHandle([1, 0], [0, 1], lockAspect=True)
        self.gw2.addItem(self.roi2)
        self.gw2.setAspectLocked()
        setprop(self.gw2)

        # GW3
        self.graphicsLayoutWidget_gw3.clear()
        self.gw3 = self.graphicsLayoutWidget_gw3.addPlot(row=0, col=0)

        self.img3 = pg.ImageItem()
        self.gw3.addItem(self.img3)
        self.gw3.setAspectLocked()

        # マウスによるズーム・パンを有効化
        self.gw3.setMouseEnabled(x=True, y=True)
        self.gw3.enableAutoRange(enable=False)

        setprop(self.gw3)

        # シグナル接続
        self.roi1.sigRegionChanged.connect(self.on_roi1_changed)
        self.roi2.sigRegionChanged.connect(self.on_roi2_changed)
        self.gw3.getViewBox().sigRangeChanged.connect(self.on_gw3_view_changed)

        # 初回更新
        self.update_gw1()
        self.update_gw2()
        self.update_gw3()

    def on_roi1_changed(self):
        """ROI1 が変更されたとき：GW2 の表示範囲と ROI2 の位置を更新する。"""
        Data.roi1h = Decimal(str(self.roi1.pos()[0]))
        Data.roi1v = Decimal(str(self.roi1.pos()[1]))
        Data.roi1dh = Decimal(str(self.roi1.size()[0]))
        Data.roi1dv = Decimal(str(self.roi1.size()[1]))

        # GW2 の表示範囲を ROI1 に合わせる
        Data.h21 = Data.roi1h
        Data.h22 = Data.roi1h + Data.roi1dh
        Data.v21 = Data.roi1v
        Data.v22 = Data.roi1v + Data.roi1dv

        # ROI2 を ROI1 内に維持
        Data.roi2h = Data.roi1h + 2 * Data.roi1dh / 5
        Data.roi2v = Data.roi1v + 2 * Data.roi1dv / 5
        Data.roi2dh = Data.roi1dh / 5
        Data.roi2dv = Data.roi1dv / 5
        self.roi2.setPos([float(Data.roi2h), float(Data.roi2v)])
        self.roi2.setSize([float(Data.roi2dh), float(Data.roi2dv)])

        self.label_h2x.setText(
            f"{Data.roi1h:.4f}, {Data.roi1v:.4f}, {Data.roi1dh:.4f}, {Data.roi1dv:.4f}"
        )

        self.update_gw2()

    def on_roi2_changed(self):
        """ROI2 が変更されたとき：GW3 の表示範囲を更新する。"""
        Data.roi2h = Decimal(str(self.roi2.pos()[0]))
        Data.roi2v = Decimal(str(self.roi2.pos()[1]))
        Data.roi2dh = Decimal(str(self.roi2.size()[0]))
        Data.roi2dv = Decimal(str(self.roi2.size()[1]))

        # GW3 の表示範囲を ROI2 に合わせる
        Data.h31 = Data.roi2h
        Data.h32 = Data.roi2h + Data.roi2dh
        Data.v31 = Data.roi2v
        Data.v32 = Data.roi2v + Data.roi2dv

        # GW3 の ViewBox を ROI2 の範囲にリセット（再帰防止のためシグナルをブロック）
        self.gw3.getViewBox().blockSignals(True)
        self.gw3.setXRange(float(Data.h31), float(Data.h32), padding=0)
        self.gw3.setYRange(float(Data.v31), float(Data.v32), padding=0)
        self.gw3.getViewBox().blockSignals(False)

        self.label_h3x.setText(
            f"{Data.roi2h:.4f}, {Data.roi2v:.4f}, {Data.roi2dh:.4f}, {Data.roi2dv:.4f}"
        )

        self.update_gw3()

    def on_gw3_view_changed(self):
        """GW3 でマウスズーム／パンが行われたとき：正方形を保ちつつ再計算する。"""
        # float64 の精度限界を考慮した最小ズームサイズ
        MIN_ZOOM_SIZE = Decimal("1e-12")

        self.statusBar().showMessage("再計算中...", 500)

        view_box = self.gw3.getViewBox()
        [[x_min, x_max], [y_min, y_max]] = view_box.viewRange()

        h31_new = Decimal(str(x_min))
        h32_new = Decimal(str(x_max))
        v31_new = Decimal(str(y_min))
        v32_new = Decimal(str(y_max))

        # 正方形を強制（大きい方の辺に合わせる）
        current_h_size = h32_new - h31_new
        current_v_size = v32_new - v31_new
        max_size = max(current_h_size, current_v_size)

        # 精度限界チェック：限界に達したら直前の範囲に戻す
        if max_size < MIN_ZOOM_SIZE:
            self.gw3.getViewBox().blockSignals(True)
            self.gw3.setXRange(float(Data.h31), float(Data.h32), padding=0)
            self.gw3.setYRange(float(Data.v31), float(Data.v32), padding=0)
            self.gw3.getViewBox().blockSignals(False)
            self.statusBar().showMessage(
                f"⚠ 精度限界: 最小ズームサイズ {MIN_ZOOM_SIZE:.2e} に到達", 5000
            )
            return

        # 中心を保って正方形に調整
        h_center = (h31_new + h32_new) / 2
        v_center = (v31_new + v32_new) / 2
        Data.h31 = h_center - max_size / 2
        Data.h32 = h_center + max_size / 2
        Data.v31 = v_center - max_size / 2
        Data.v32 = v_center + max_size / 2

        # ROI2 の範囲外に出ないよう制限
        roi2_h_min = Data.roi2h
        roi2_h_max = Data.roi2h + Data.roi2dh
        roi2_v_min = Data.roi2v
        roi2_v_max = Data.roi2v + Data.roi2dv

        if Data.h31 < roi2_h_min:
            Data.h31 = roi2_h_min
            Data.h32 = Data.h31 + max_size
        if Data.h32 > roi2_h_max:
            Data.h32 = roi2_h_max
            Data.h31 = Data.h32 - max_size
        if Data.v31 < roi2_v_min:
            Data.v31 = roi2_v_min
            Data.v32 = Data.v31 + max_size
        if Data.v32 > roi2_v_max:
            Data.v32 = roi2_v_max
            Data.v31 = Data.v32 - max_size

        # ViewBox を更新（無限ループ防止のためシグナルをブロック）
        self.gw3.getViewBox().blockSignals(True)
        self.gw3.setXRange(float(Data.h31), float(Data.h32), padding=0)
        self.gw3.setYRange(float(Data.v31), float(Data.v32), padding=0)
        self.gw3.getViewBox().blockSignals(False)

        self.update_gw3()

        self.statusBar().showMessage(f"🔍 ズーム: サイズ={float(max_size):.2e}", 3000)

    def update_gw1(self):
        """GW1 の画像を再計算して更新する。"""
        Data.n_max1 = self.lineEdit_n_max1.text()
        Data.cmap1 = self.cmap1.currentText()

        mandelbrot1 = self.mandelresult(
            Data.h11, Data.h12, Data.v11, Data.v12, int(Data.res1), int(Data.n_max1)
        )

        cmap = pg.colormap.get(Data.cmap1, source="colorcet")
        self.img1.setImage(mandelbrot1)
        self.img1.setLookupTable(cmap.getLookupTable())

        self._apply_transform(
            self.img1, Data.h11, Data.h12, Data.v11, Data.v12, int(Data.res1)
        )

    def update_gw2(self):
        """GW2 の画像を再計算して更新する。"""
        Data.res2 = self.lineEdit_res2.text()
        Data.n_max2 = self.lineEdit_n_max2.text()
        Data.cmap2 = self.cmap2.currentText()

        mandelbrot2 = self.mandelresult(
            Data.h21, Data.h22, Data.v21, Data.v22, int(Data.res2), int(Data.n_max2)
        )

        cmap = pg.colormap.get(Data.cmap2, source="colorcet")
        self.img2.setImage(mandelbrot2)
        self.img2.setLookupTable(cmap.getLookupTable())

        self._apply_transform(
            self.img2, Data.h21, Data.h22, Data.v21, Data.v22, int(Data.res2)
        )

    def update_gw3(self):
        """GW3 の画像を再計算して更新する。"""
        Data.res3 = self.lineEdit_res3.text()
        Data.n_max3 = self.lineEdit_n_max3.text()
        Data.cmap3 = self.cmap3.currentText()

        mandelbrot3 = self.mandelresult(
            Data.h31, Data.h32, Data.v31, Data.v32, int(Data.res3), int(Data.n_max3)
        )

        cmap = pg.colormap.get(Data.cmap3, source="colorcet")
        self.img3.setImage(mandelbrot3)
        self.img3.setLookupTable(cmap.getLookupTable())

        self._apply_transform(
            self.img3, Data.h31, Data.h32, Data.v31, Data.v32, int(Data.res3)
        )

    @staticmethod
    def _apply_transform(img, h1, h2, v1, v2, reso):
        """画像アイテムを複素平面の座標系に合わせて変換する。"""
        sx, sy = scale_factor(h1, h2, v1, v2, reso)
        tx, ty = translate_factor(h1, h2, v1, v2, reso)
        tr = QtGui.QTransform()
        tr.scale(sx, sy)
        tr.translate(tx, ty)
        tr.rotate(-90)
        img.setTransform(tr)


def main():
    app = QApplication(sys.argv)

    # qdarktheme は任意依存。存在すればダークテーマを適用し、無ければ標準テーマで動作する。
    try:
        import qdarktheme

        app.setStyleSheet(qdarktheme.load_stylesheet())
    except Exception:
        pass

    window = MainWindow()  # GC されないよう参照を保持する
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
