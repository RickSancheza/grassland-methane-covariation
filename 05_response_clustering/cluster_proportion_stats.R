library(terra)
library(dplyr)
library(ggplot2)
library(readr)
library(tidyr)

# ============================================================
# 1. 路径设置
# ============================================================

cluster_file <- "F:/DOCTOR/graduating/test2/40/3_thirdpart/2_聚类/聚类.tif"
out_dir <- "F:/DOCTOR/graduating/test2/40/3_thirdpart/2_聚类/其他相关结果/2_不同簇占比"

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# 临时英文路径（避免中文路径保存 pdf 出问题）
tmp_dir <- "F:/DOCTOR/graduating/test2/40/3_thirdpart/tmp_export_cluster_ratio"
dir.create(tmp_dir, recursive = TRUE, showWarnings = FALSE)

# ============================================================
# 2. 字体设置
# ============================================================

if (.Platform$OS.type == "windows") {
  windowsFonts(SimHei = windowsFont("SimHei"))
  cn_font <- "SimHei"
} else {
  cn_font <- "sans"
}

# 如果安装了 showtext，就优先启用，中文更稳
if (requireNamespace("showtext", quietly = TRUE)) {
  showtext::showtext_auto()
}

# ============================================================
# 3. 簇名称与颜色
# ============================================================

cluster_labels <- c(
  "1" = "牲畜主导正向响应型",
  "2" = "农业主导正向响应型",
  "3" = "牲畜主导反向响应型",
  "4" = "废弃物主导反向响应型",
  "5" = "废弃物能源复合响应型",
  "6" = "湿地主导正向响应型"
)

cluster_order <- c(
  "牲畜主导正向响应型",
  "农业主导正向响应型",
  "牲畜主导反向响应型",
  "废弃物主导反向响应型",
  "废弃物能源复合响应型",
  "湿地主导正向响应型"
)

# 颜色按你给的图例来
cluster_colors <- c(
  "牲畜主导正向响应型" = "#8DC63F",  # 绿色
  "农业主导正向响应型" = "#F4B000",  # 橙色
  "牲畜主导反向响应型" = "#F4766D",  # 粉红/鲑红
  "废弃物主导反向响应型" = "#A9A9A9",  # 灰色
  "废弃物能源复合响应型" = "#9B7B3D",  # 棕褐色
  "湿地主导正向响应型" = "#1F6FE5"   # 蓝色
)

# ============================================================
# 4. 读取聚类图
# ============================================================

cluster_r <- rast(cluster_file)
cluster_r <- round(cluster_r)

# 0 作为无效值
cluster_r[cluster_r == 0] <- NA

# ============================================================
# 5. 转为数据框并统计不同簇占比
# ============================================================

df <- data.frame(
  cluster = values(cluster_r, mat = FALSE)
)

df <- df %>%
  filter(is.finite(cluster)) %>%
  mutate(cluster = as.integer(round(cluster))) %>%
  filter(cluster %in% 1:6)

# 统计像元数
stat_df <- df %>%
  group_by(cluster) %>%
  summarise(
    n_pixels = n(),
    .groups = "drop"
  )

# 补齐 1:6，避免某个簇没有像元时报错
all_clusters <- data.frame(cluster = 1:6)

stat_df <- all_clusters %>%
  left_join(stat_df, by = "cluster") %>%
  mutate(
    n_pixels = ifelse(is.na(n_pixels), 0, n_pixels),
    total_pixels = sum(n_pixels, na.rm = TRUE),
    proportion_percent = ifelse(
      total_pixels > 0,
      n_pixels / total_pixels * 100,
      0
    ),
    cluster_name = cluster_labels[as.character(cluster)]
  )

stat_df$cluster_name <- factor(
  stat_df$cluster_name,
  levels = cluster_order
)

# ============================================================
# 6. 输出 csv
# ============================================================

# 长表
write_excel_csv(
  stat_df,
  file.path(out_dir, "不同簇占比统计.csv")
)

# 宽表（可选）
stat_wide <- stat_df %>%
  select(cluster_name, proportion_percent) %>%
  tidyr::pivot_wider(
    names_from = cluster_name,
    values_from = proportion_percent
  )

write_excel_csv(
  stat_wide,
  file.path(out_dir, "不同簇占比统计_宽表.csv")
)

# ============================================================
# 7. 绘图
# ============================================================

p <- ggplot(
  stat_df,
  aes(x = cluster_name, y = proportion_percent, fill = cluster_name)
) +
  geom_col(
    width = 0.72,
    color = "black",
    linewidth = 0.3
  ) +
  geom_text(
    aes(label = sprintf("%.2f%%", proportion_percent)),
    vjust = -0.35,
    size = 4.2,
    family = cn_font
  ) +
  scale_fill_manual(
    values = cluster_colors,
    breaks = cluster_order
  ) +
  scale_y_continuous(
    expand = expansion(mult = c(0, 0.08))
  ) +
  labs(
    x = "响应簇类型",
    y = "像元占比 (%)",
    title = "不同簇的像元占比"
  ) +
  theme_bw(base_size = 13, base_family = cn_font) +
  theme(
    text = element_text(family = cn_font, color = "black"),
    plot.title = element_text(
      hjust = 0.5,
      size = 16,
      face = "bold"
    ),
    axis.title = element_text(size = 13),
    axis.text.x = element_text(
      angle = 20,
      hjust = 1,
      size = 11
    ),
    axis.text.y = element_text(size = 11),
    legend.position = "none",
    panel.grid.minor = element_blank()
  )

print(stat_df)
print(p)

# ============================================================
# 8. 保存 PNG 和 PDF
# 为避免中文路径保存失败，先保存到英文临时路径，再复制过去
# ============================================================

png_tmp   <- file.path(tmp_dir, "cluster_ratio.png")
pdf_tmp   <- file.path(tmp_dir, "cluster_ratio.pdf")

png_final <- file.path(out_dir, "不同簇占比.png")
pdf_final <- file.path(out_dir, "不同簇占比.pdf")

# 保存 PNG
ggsave(
  filename = png_tmp,
  plot = p,
  width = 10,
  height = 6,
  dpi = 300
)

# 保存 PDF
if (capabilities("cairo")) {
  ggsave(
    filename = pdf_tmp,
    plot = p,
    width = 10,
    height = 6,
    device = cairo_pdf
  )
} else {
  ggsave(
    filename = pdf_tmp,
    plot = p,
    width = 10,
    height = 6,
    device = "pdf"
  )
}

# 复制到目标中文路径
file.copy(png_tmp, png_final, overwrite = TRUE)
file.copy(pdf_tmp, pdf_final, overwrite = TRUE)

# ============================================================
# 9. 完成提示
# ============================================================

message("完成！结果已保存到：")
message(out_dir)

message("输出文件包括：")
message("1. 不同簇占比统计.csv")
message("2. 不同簇占比统计_宽表.csv")
message("3. 不同簇占比.png")
message("4. 不同簇占比.pdf")