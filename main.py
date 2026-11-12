import os
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pydicom
from PIL import Image, ImageTk
from scipy import ndimage


class DicomMPRViewer:
  def __init__(self, root):
    self.root = root
    self.root.title("DICOM 3断面ビューワー & 3D肺レンダリング")
    self.root.geometry("1150x680")

    # 3Dボリュームデータとメタデータ
    self.volume = None  # shape: (Z, Y, X)
    self.spacing = (1.0, 1.0, 1.0)  # (z_spacing, y_spacing, x_spacing)
    self.z_scale = 1.0

    # 表示キャンバスサイズ
    self.disp_size = 340

    # Tkinter変数（スライダー連動用）
    self.z_var = tk.IntVar(value=0)
    self.y_var = tk.IntVar(value=0)
    self.x_var = tk.IntVar(value=0)
    self.wl_var = tk.IntVar(value=-500)
    self.ww_var = tk.IntVar(value=1500)

    # 画像参照保持用
    self.photo_axial = None
    self.photo_coronal = None
    self.photo_sagittal = None

    self._setup_menu()
    self._setup_ui()

  def _setup_menu(self):
    menubar = tk.Menu(self.root)
    file_menu = tk.Menu(menubar, tearoff=0)
    file_menu.add_command(label="DICOMフォルダを開く...",
                          command=self.load_dicom_folder)
    file_menu.add_separator()
    file_menu.add_command(label="肺領域の3D表示 (ボリュームレンダリング)",
                          command=self.show_3d_lung)
    file_menu.add_separator()
    file_menu.add_command(label="終了", command=self.root.quit)
    menubar.add_cascade(label="ファイル(F)", menu=file_menu)
    self.root.config(menu=menubar)

  def _setup_ui(self):
    # 上部：3断面表示エリア
    view_frame = tk.Frame(self.root, bg="#222222")
    view_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Axial
    f_ax = tk.Frame(view_frame, bg="#222222")
    f_ax.pack(side=tk.LEFT, expand=True)
    tk.Label(f_ax, text="Axial (水平断)", fg="white",
             bg="#222222", font=("Arial", 11, "bold")).pack()
    self.canvas_axial = tk.Canvas(
        f_ax, width=self.disp_size, height=self.disp_size, bg="black", highlightthickness=1)
    self.canvas_axial.pack(pady=5)

    # Coronal
    f_cor = tk.Frame(view_frame, bg="#222222")
    f_cor.pack(side=tk.LEFT, expand=True)
    tk.Label(f_cor, text="Coronal (冠状断)", fg="white",
             bg="#222222", font=("Arial", 11, "bold")).pack()
    self.canvas_coronal = tk.Canvas(
        f_cor, width=self.disp_size, height=self.disp_size, bg="black", highlightthickness=1)
    self.canvas_coronal.pack(pady=5)

    # Sagittal
    f_sag = tk.Frame(view_frame, bg="#222222")
    f_sag.pack(side=tk.LEFT, expand=True)
    tk.Label(f_sag, text="Sagittal (矢状断)", fg="white",
             bg="#222222", font=("Arial", 11, "bold")).pack()
    self.canvas_sagittal = tk.Canvas(
        f_sag, width=self.disp_size, height=self.disp_size, bg="black", highlightthickness=1)
    self.canvas_sagittal.pack(pady=5)

    # 下部：コントロールエリア
    ctrl_frame = tk.Frame(self.root)
    ctrl_frame.pack(fill=tk.X, padx=20, pady=5)

    # スライス位置スライダー
    pos_frame = tk.LabelFrame(ctrl_frame, text="スライス位置", padx=10, pady=5)
    pos_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

    self.scale_z = self._create_slider(
        pos_frame, "Axial (Z):", self.z_var, 0, 100)
    self.scale_y = self._create_slider(
        pos_frame, "Coronal (Y):", self.y_var, 0, 100)
    self.scale_x = self._create_slider(
        pos_frame, "Sagittal (X):", self.x_var, 0, 100)

    # WL / WW スライダー & 3Dボタン
    right_frame = tk.Frame(ctrl_frame)
    right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

    win_frame = tk.LabelFrame(
        right_frame, text="階調変換 (Windowing)", padx=10, pady=5)
    win_frame.pack(fill=tk.X)

    self.scale_wl = self._create_slider(
        win_frame, "WL (レベル):", self.wl_var, -1500, 1500)
    self.scale_ww = self._create_slider(
        win_frame, "WW (幅):", self.ww_var, 1, 4000)

    # 3Dボリュームレンダリング起動ボタン（観点7）
    btn_3d = tk.Button(
        right_frame,
        text="肺領域を3D表示 (ボリュームレンダリング)",
        command=self.show_3d_lung,
        bg="#0066CC",
        fg="white",
        font=("Arial", 10, "bold"),
        pady=6,
    )
    btn_3d.pack(fill=tk.X, pady=8)

  def _create_slider(self, parent, label_text, variable, from_, to_):
    row = tk.Frame(parent)
    row.pack(fill=tk.X, pady=2)
    tk.Label(row, text=label_text, width=12, anchor="w").pack(side=tk.LEFT)
    scale = tk.Scale(
        row,
        variable=variable,
        from_=from_,
        to=to_,
        orient=tk.HORIZONTAL,
        command=lambda _: self.update_views(),
    )
    scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
    return scale

  def load_dicom_folder(self):
    folder_path = filedialog.askdirectory(title="DICOMファイルが含まれるフォルダを選択")
    if not folder_path:
      return

    slices = []
    for root_dir, _, files in os.walk(folder_path):
      for file in files:
        filepath = os.path.join(root_dir, file)
        try:
          ds = pydicom.dcmread(filepath)
          if hasattr(ds, "pixel_array"):
            slices.append(ds)
        except Exception:
          continue

    if not slices:
      messagebox.showerror("エラー", "有効なDICOM画像が見つかりませんでした。")
      return

    if hasattr(slices[0], "ImagePositionPatient"):
      slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))
    elif hasattr(slices[0], "InstanceNumber"):
      slices.sort(key=lambda s: int(s.InstanceNumber))

    vol_list = []
    for s in slices:
      arr = s.pixel_array.astype(np.float32)
      slope = float(getattr(s, "RescaleSlope", 1.0))
      intercept = float(getattr(s, "RescaleIntercept", 0.0))
      arr = arr * slope + intercept
      vol_list.append(arr)

    self.volume = np.stack(vol_list, axis=0)[::-1, :, :]

    try:
      px_y = float(slices[0].PixelSpacing[0])
      px_x = float(slices[0].PixelSpacing[1])
      if len(slices) > 1 and hasattr(slices[0], "ImagePositionPatient"):
        z_diff = abs(float(slices[1].ImagePositionPatient[2]) -
                     float(slices[0].ImagePositionPatient[2]))
      else:
        z_diff = float(getattr(slices[0], "SliceThickness", px_y))
      self.spacing = (z_diff, px_y, px_x)
      self.z_scale = z_diff / px_y if px_y > 0 else 1.0
    except Exception:
      self.spacing = (1.0, 1.0, 1.0)
      self.z_scale = 1.0

    z_dim, y_dim, x_dim = self.volume.shape
    self.scale_z.config(to=z_dim - 1)
    self.scale_y.config(to=y_dim - 1)
    self.scale_x.config(to=x_dim - 1)

    self.z_var.set(z_dim // 2)
    self.y_var.set(y_dim // 2)
    self.x_var.set(x_dim // 2)

    self.update_views()

  def _apply_windowing(self, slice_2d):
    wl = self.wl_var.get()
    ww = max(1, self.ww_var.get())
    min_val = wl - ww / 2.0
    max_val = wl + ww / 2.0
    clipped = np.clip(slice_2d, min_val, max_val)
    norm = ((clipped - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)
    return norm

  def _render_to_canvas(self, canvas, img_2d, is_z_vertical=False):
    h, w = img_2d.shape
    phys_h = h * (self.z_scale if is_z_vertical else 1.0)
    phys_w = float(w)

    scale = min(self.disp_size / phys_w, self.disp_size / phys_h)
    new_w = max(1, int(phys_w * scale))
    new_h = max(1, int(phys_h * scale))

    pil_img = Image.fromarray(img_2d).resize(
        (new_w, new_h), Image.Resampling.BILINEAR)
    photo = ImageTk.PhotoImage(pil_img)

    canvas.delete("all")
    x0 = (self.disp_size - new_w) // 2
    y0 = (self.disp_size - new_h) // 2
    canvas.create_image(x0, y0, anchor=tk.NW, image=photo)
    return photo, x0, y0, new_w, new_h

  def update_views(self):
    if self.volume is None:
      return

    z_dim, y_dim, x_dim = self.volume.shape
    z = self.z_var.get()
    y = self.y_var.get()
    x = self.x_var.get()

    # 1. Axial
    ax_img = self._apply_windowing(self.volume[z, :, :])
    self.photo_axial, ax_x0, ax_y0, ax_w, ax_h = self._render_to_canvas(
        self.canvas_axial, ax_img, False)
    cx_ax = ax_x0 + (x / max(1, x_dim - 1)) * ax_w
    cy_ax = ax_y0 + (y / max(1, y_dim - 1)) * ax_h
    self.canvas_axial.create_line(
        cx_ax, ax_y0, cx_ax, ax_y0 + ax_h, fill="#00FF00", width=1)
    self.canvas_axial.create_line(
        ax_x0, cy_ax, ax_x0 + ax_w, cy_ax, fill="#00FF00", width=1)

    # 2. Coronal
    cor_img = self._apply_windowing(self.volume[:, y, :])
    self.photo_coronal, cor_x0, cor_y0, cor_w, cor_h = self._render_to_canvas(
        self.canvas_coronal, cor_img, True)
    cx_cor = cor_x0 + (x / max(1, x_dim - 1)) * cor_w
    cy_cor = cor_y0 + (z / max(1, z_dim - 1)) * cor_h
    self.canvas_coronal.create_line(
        cx_cor, cor_y0, cx_cor, cor_y0 + cor_h, fill="#00FF00", width=1)
    self.canvas_coronal.create_line(
        cor_x0, cy_cor, cor_x0 + cor_w, cy_cor, fill="#00FF00", width=1)

    # 3. Sagittal
    sag_img = self._apply_windowing(self.volume[:, :, x])
    self.photo_sagittal, sag_x0, sag_y0, sag_w, sag_h = self._render_to_canvas(
        self.canvas_sagittal, sag_img, True)
    cx_sag = sag_x0 + (y / max(1, y_dim - 1)) * sag_w
    cy_sag = sag_y0 + (z / max(1, z_dim - 1)) * sag_h
    self.canvas_sagittal.create_line(
        cx_sag, sag_y0, cx_sag, sag_y0 + sag_h, fill="#00FF00", width=1)
    self.canvas_sagittal.create_line(
        sag_x0, cy_sag, sag_x0 + sag_w, cy_sag, fill="#00FF00", width=1)

  def extract_lung_volume(self):
    """CTボリュームから肺領域を抽出し、ボリュームレンダリング用のCT値配列を返す"""
    # 1. 空気・肺野のしきい値処理 (-1000 HU 〜 -350 HU)
    air_mask = (self.volume > -1000) & (self.volume < -350)

    # 2. 体外の空気を除去（各スライスの四隅とつながっている領域を背景として除外）
    body_air = np.zeros_like(air_mask, dtype=bool)
    for z in range(air_mask.shape[0]):
      sl = air_mask[z]
      labeled, num_features = ndimage.label(sl)
      if num_features == 0:
        continue
      border_labels = set(
          np.concatenate([labeled[0, :], labeled[-1, :],
                         labeled[:, 0], labeled[:, -1]])
      )
      border_labels.discard(0)
      internal = np.isin(labeled, list(border_labels),
                         invert=True) & (labeled > 0)
      body_air[z] = ndimage.binary_fill_holes(internal)

    # 3. 3Dラベリングで小さなノイズを除去し、大きな連結成分（肺）を抽出
    labeled_3d, num_3d = ndimage.label(body_air)
    if num_3d > 0:
      sizes = ndimage.sum(body_air, labeled_3d, range(1, num_3d + 1))
      max_size = np.max(sizes)
      valid_labels = [i + 1 for i,
                      sz in enumerate(sizes) if sz >= max_size * 0.15]
      lung_mask = np.isin(labeled_3d, valid_labels)
    else:
      lung_mask = body_air

    # ボリュームレンダリング用に、肺領域内の元のCT値（を1〜255にスケーリングした値）を格納
    lung_vol = np.zeros_like(self.volume, dtype=np.float32)
    vals = np.clip((self.volume[lung_mask] + 950.0) /
                   650.0 * 254.0 + 1.0, 1.0, 255.0)
    lung_vol[lung_mask] = vals
    return lung_vol

  def show_3d_lung(self):
    """観点7: 抽出した肺領域をボリュームレンダリングで3D表示"""
    if self.volume is None:
      messagebox.showwarning("警告", "先にDICOMフォルダを読み込んでください。")
      return

    try:
      import pyvista as pv
    except Exception as e:
      messagebox.showerror("エラー", f"PyVistaの読み込みに失敗しました:\n{e}")
      return

    try:
      lung_vol = self.extract_lung_volume()
      if np.max(lung_vol) == 0:
        messagebox.showerror("エラー", "肺領域を抽出できませんでした。")
        return

      vol_for_pv = lung_vol[::-1, :, :]
      grid = pv.ImageData()
      grid.dimensions = np.array(vol_for_pv.shape[::-1])
      grid.spacing = (self.spacing[2], self.spacing[1], self.spacing[0])
      grid.point_data["values"] = vol_for_pv.flatten(order="C")

      plotter = pv.Plotter(title="肺領域 ボリュームレンダリング (3D)")
      plotter.set_background("black")
      opacity = [0.0, 0.05, 0.15, 0.4, 0.8]
      plotter.add_volume(
          grid,
          scalars="values",
          cmap="bone",
          opacity=opacity,
          shade=True,
      )
      plotter.add_axes()
      plotter.show()
    except Exception as e:
      messagebox.showerror("エラー", f"3D表示中にエラーが発生しました:\n{e}")


if __name__ == "__main__":
  root = tk.Tk()
  app = DicomMPRViewer(root)
  root.mainloop()
