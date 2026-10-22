# ============================================================
# 按草地类型计算不同部门 CH4 排放变化与 ΔGR 的协变关系
# 版本：直接使用已经算好的 ΔGR 和 Δ排放
# 不再重新计算 2004 基准差值
#
# 输出指标：
# 1. covariance：带符号协方差
# 2. correlation：相关系数
# 3. abs_contribution_percent：绝对相对协变强度
# 4. signed_contribution_percent：带符号相对协变强度
# ============================================================

library(terra)
library(dplyr)
library(stringr)
library(ggplot2)
library(readr)
library(tidyr)

# ============================================================
# 1. 中文输出和字体设置
# ============================================================

try(Sys.setlocale("LC_CTYPE", "Chinese"), silent = TRUE)

save_csv <- function(x, path) {
  readr::write_excel_csv(x, path)
}

if (.Platform$OS.type == "windows") {
  windowsFonts(SimHei = windowsFont("SimHei"))
  cn_font <- "SimHei"
} else {
  cn_font <- "sans"
}

# ============================================================
# 2. 路径设置：你现在的数据位置
# ============================================================

# 已经算好的 ΔGR
gr_dir <- "F:/DOCTOR/graduating/test2/40/0_data/1_Delta_GR"

# 已经算好的全部部门 Δ排放
sector_delta_root <- "F:/DOCTOR/graduating/test2/40/0_data/3_change"

# 草地类型
grassland_type_file <- "F:/DOCTOR/graduating/test2/40/0_data/grasslandtype.tif"

# 输出文件夹
out_dir <- "F:/DOCTOR/graduating/test2/40/2_secondpart/2_contribuation"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# 分析年份
start_year <- 2005
end_year   <- 2020

# ------------------------------------------------------------
# 排放栅格如果是“每个像元的年排放量变化”，用 FALSE
# 如果是 kg m-2 yr-1 这种通量，才改 TRUE
# ------------------------------------------------------------
emission_is_flux <- FALSE

# ------------------------------------------------------------
# 单位换算到 Tg CH4 yr-1
# Δ排放单位是 ton CH4 cell-1 yr-1：1e6
# Δ排放单位是 kg CH4 cell-1 yr-1：1e9
# 已经是 Tg CH4 yr-1：1
# ------------------------------------------------------------
emission_to_Tg_divisor <- 1e6

# 至少多少年有效才计算协方差和相关系数
min_valid_years <- 5

# 草地类型中 0 作为无效值排除
exclude_grass_values <- c(0)

# ============================================================
# 3. 草地类型和部门名称映射
# ============================================================

grassland_labels <- c(
  "1" = "热带—亚热带草地类",
  "2" = "温带草地类",
  "3" = "洪泛湿生草地类",
  "4" = "山地草地类",
  "5" = "冻原类",
  "6" = "干旱草地—荒漠草灌类"
)

grassland_order <- c(
  "热带—亚热带草地类",
  "温带草地类",
  "洪泛湿生草地类",
  "山地草地类",
  "冻原类",
  "干旱草地—荒漠草灌类"
)

# 兼容 3_change 里可能出现的 Enteric 或 Livestock 文件夹名
sector_labels <- c(
  "Enteric"     = "牲畜",
  "Livestock"   = "牲畜",
  "Agriculture" = "农业",
  "Energy"      = "能源",
  "Industrial"  = "工业",
  "Waste"       = "废弃物",
  "Wetland"     = "湿地"
)

sector_order <- c(
  "牲畜",
  "农业",
  "能源",
  "工业",
  "废弃物",
  "湿地"
)

# ============================================================
# 4. 工具函数
# ============================================================

get_year_files <- function(folder) {
  files <- list.files(folder, pattern = "\\.tif$", full.names = TRUE)
  years <- str_extract(basename(files), "\\d{4}") |> as.integer()
  
  data.frame(
    year = years,
    file = files,
    stringsAsFactors = FALSE
  ) |>
    filter(!is.na(year)) |>
    filter(year >= start_year, year <= end_year) |>
    arrange(year)
}

read_align <- function(file, template, method = "bilinear") {
  r <- rast(file)
  
  if (!same.crs(r, template)) {
    r <- project(r, template, method = method)
  }
  
  if (!compareGeom(r, template, stopOnError = FALSE)) {
    r <- resample(r, template, method = method)
  }
  
  return(r)
}

# ΔGR 是浓度变化，用面积加权平均
calc_gr_mean_by_type <- function(file, template, grass_type, type_value) {
  r <- read_align(file, template, method = "bilinear")
  r <- mask(r, template)
  
  m <- ifel(grass_type == type_value, 1, NA)
  r_m <- r * m
  
  area_km2 <- cellSize(template, unit = "km")
  
  num <- global(r_m * area_km2, "sum", na.rm = TRUE)[1, 1]
  den <- global(mask(area_km2 * m, r_m), "sum", na.rm = TRUE)[1, 1]
  
  if (is.na(den) || den == 0) return(NA_real_)
  return(num / den)
}

# Δ排放是质量变化，用区域总量
calc_emission_sum_by_type <- function(file, template, grass_type, type_value) {
  r <- read_align(file, template, method = "bilinear")
  r <- mask(r, template)
  
  m <- ifel(grass_type == type_value, 1, NA)
  r_m <- r * m
  
  if (emission_is_flux) {
    area_m2 <- cellSize(template, unit = "m")
    total_value <- global(r_m * area_m2, "sum", na.rm = TRUE)[1, 1]
  } else {
    total_value <- global(r_m, "sum", na.rm = TRUE)[1, 1]
  }
  
  total_Tg <- total_value / emission_to_Tg_divisor
  return(total_Tg)
}

# 跳过 backup 文件夹
list_sector_dirs <- function(root) {
  dirs <- list.dirs(root, recursive = FALSE, full.names = TRUE)
  dirs <- dirs[!str_detect(tolower(basename(dirs)), "^_backup")]
  dirs
}

safe_cov <- function(x, y, min_n = 5) {
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < min_n) return(NA_real_)
  cov(x[ok], y[ok])
}

safe_cor <- function(x, y, min_n = 5) {
  ok <- is.finite(x) & is.finite(y)
  if (sum(ok) < min_n) return(NA_real_)
  if (sd(x[ok]) == 0 || sd(y[ok]) == 0) return(NA_real_)
  cor(x[ok], y[ok])
}

# ============================================================
# 5. 读取模板和草地类型
# ============================================================

gr_files <- get_year_files(gr_dir)

if (nrow(gr_files) == 0) {
  stop("ΔGR 文件夹中没有识别到 2005—2020 年 tif，请检查路径。")
}

missing_gr_years <- setdiff(start_year:end_year, gr_files$year)
if (length(missing_gr_years) > 0) {
  stop(paste0("ΔGR 缺少年份：", paste(missing_gr_years, collapse = ", ")))
}

template <- rast(gr_files$file[gr_files$year == start_year][1])

grass_type <- rast(grassland_type_file)

if (!same.crs(grass_type, template)) {
  grass_type <- project(grass_type, template, method = "near")
}

if (!compareGeom(grass_type, template, stopOnError = FALSE)) {
  grass_type <- resample(grass_type, template, method = "near")
}

grass_type <- mask(grass_type, template)

grass_values <- unique(values(grass_type, mat = FALSE))
grass_values <- sort(na.omit(grass_values))
grass_values <- grass_values[!(grass_values %in% exclude_grass_values)]

cat("识别到的草地类型编号：\n")
print(grass_values)

cat("对应草地类型名称：\n")
print(grassland_labels[as.character(grass_values)])

check_type <- data.frame(
  grassland_type = grass_values,
  grassland_name = grassland_labels[as.character(grass_values)]
)

save_csv(
  check_type,
  file.path(out_dir, "00_check_grassland_type_mapping.csv")
)

# ============================================================
# 6. 直接计算每个草地类型的 ΔGR 年序列
# ============================================================

gr_series_list <- list()

for (g in grass_values) {
  
  message("正在计算草地类型 ", g, " 的 ΔGR 年序列...")
  
  tmp <- gr_files |>
    rowwise() |>
    mutate(
      grassland_type = g,
      grassland_name = grassland_labels[as.character(g)],
      deltaGR = calc_gr_mean_by_type(file, template, grass_type, g)
    ) |>
    ungroup() |>
    select(grassland_type, grassland_name, year, deltaGR)
  
  gr_series_list[[as.character(g)]] <- tmp
}

gr_series <- bind_rows(gr_series_list) |>
  mutate(
    grassland_name = factor(grassland_name, levels = grassland_order)
  )

save_csv(
  gr_series,
  file.path(out_dir, "01_deltaGR_by_grassland_type.csv")
)

# ============================================================
# 7. 读取所有部门 Δ排放文件夹
# ============================================================

sector_dirs_raw <- list_sector_dirs(sector_delta_root)

if (length(sector_dirs_raw) == 0) {
  stop("3_change 文件夹中没有识别到部门子文件夹。")
}

sector_dirs <- setNames(sector_dirs_raw, basename(sector_dirs_raw))

cat("识别到的排放部门文件夹：\n")
print(sector_dirs)

# ============================================================
# 8. 直接计算每个草地类型、每个部门的 Δ排放年序列
# ============================================================

emission_list <- list()

for (sector in names(sector_dirs)) {
  
  message("正在处理部门：", sector)
  
  sector_folder <- sector_dirs[[sector]]
  sector_files <- get_year_files(sector_folder)
  
  missing_years <- setdiff(start_year:end_year, sector_files$year)
  if (length(missing_years) > 0) {
    stop(paste0(
      "部门 ", sector, " 缺少年份：",
      paste(missing_years, collapse = ", "),
      "；路径：", sector_folder
    ))
  }
  
  sector_cn <- sector_labels[sector]
  if (is.na(sector_cn)) sector_cn <- sector
  
  for (g in grass_values) {
    
    message("  草地类型：", g)
    
    tmp <- sector_files |>
      rowwise() |>
      mutate(
        grassland_type = g,
        grassland_name = grassland_labels[as.character(g)],
        sector = sector,
        sector_cn = unname(sector_cn),
        deltaE_Tg = calc_emission_sum_by_type(file, template, grass_type, g)
      ) |>
      ungroup() |>
      select(grassland_type, grassland_name, sector, sector_cn, year, deltaE_Tg)
    
    emission_list[[paste(sector, g, sep = "_")]] <- tmp
  }
}

emission_series <- bind_rows(emission_list) |>
  mutate(
    grassland_name = factor(grassland_name, levels = grassland_order),
    sector_cn = factor(sector_cn, levels = sector_order)
  )

save_csv(
  emission_series,
  file.path(out_dir, "02_sector_deltaE_by_grassland_type.csv")
)

# ============================================================
# 9. 合并 ΔGR 和 Δ排放
# ============================================================

analysis_df <- emission_series |>
  left_join(
    gr_series |> select(grassland_type, year, deltaGR),
    by = c("grassland_type", "year")
  ) |>
  filter(
    year >= start_year,
    year <= end_year,
    is.finite(deltaE_Tg),
    is.finite(deltaGR)
  )

save_csv(
  analysis_df,
  file.path(out_dir, "03_covariance_input_by_grassland_type.csv")
)

# ============================================================
# 10. 计算协方差、相关系数、绝对/带符号相对协变强度
# ============================================================

cov_result <- analysis_df |>
  group_by(grassland_type, grassland_name, sector, sector_cn) |>
  summarise(
    n_years = sum(is.finite(deltaE_Tg) & is.finite(deltaGR)),
    covariance = safe_cov(deltaE_Tg, deltaGR, min_valid_years),
    correlation = safe_cor(deltaE_Tg, deltaGR, min_valid_years),
    .groups = "drop"
  ) |>
  mutate(
    abs_covariance = abs(covariance),
    direction = case_when(
      covariance > 0 ~ "同向协变",
      covariance < 0 ~ "反向协变",
      TRUE ~ "无明显协变"
    )
  ) |>
  group_by(grassland_type, grassland_name) |>
  mutate(
    total_abs_covariance = sum(abs_covariance, na.rm = TRUE),
    abs_contribution_percent = if_else(
      is.finite(total_abs_covariance) & total_abs_covariance > 0,
      abs_covariance / total_abs_covariance * 100,
      NA_real_
    ),
    signed_contribution_percent = if_else(
      is.finite(total_abs_covariance) & total_abs_covariance > 0,
      covariance / total_abs_covariance * 100,
      NA_real_
    )
  ) |>
  ungroup() |>
  mutate(
    grassland_name = factor(grassland_name, levels = grassland_order),
    sector_cn = factor(sector_cn, levels = sector_order)
  ) |>
  arrange(grassland_type, desc(abs_contribution_percent))

save_csv(
  cov_result,
  file.path(out_dir, "04_covariance_response_by_grassland_type.csv")
)

# 兼容旧文件名
save_csv(
  cov_result,
  file.path(out_dir, "04_covariance_contribution_by_grassland_type.csv")
)

print(cov_result)

# ============================================================
# 11. 输出宽表
# ============================================================

wide_abs <- cov_result |>
  select(grassland_type, grassland_name, sector, abs_contribution_percent) |>
  pivot_wider(
    names_from = sector,
    values_from = abs_contribution_percent,
    names_prefix = "C_"
  )

wide_signed <- cov_result |>
  select(grassland_type, grassland_name, sector, signed_contribution_percent) |>
  pivot_wider(
    names_from = sector,
    values_from = signed_contribution_percent,
    names_prefix = "SC_"
  )

wide_corr <- cov_result |>
  select(grassland_type, grassland_name, sector, correlation) |>
  pivot_wider(
    names_from = sector,
    values_from = correlation,
    names_prefix = "r_"
  )

wide_summary <- wide_abs |>
  left_join(wide_signed, by = c("grassland_type", "grassland_name")) |>
  left_join(wide_corr, by = c("grassland_type", "grassland_name"))

save_csv(
  wide_summary,
  file.path(out_dir, "05_response_index_wide_by_grassland_type.csv")
)

# ============================================================
# 12. 主导部门判别表
# ============================================================

dominant_result <- cov_result |>
  group_by(grassland_type, grassland_name) |>
  slice_max(abs_contribution_percent, n = 1, with_ties = FALSE) |>
  ungroup() |>
  transmute(
    grassland_type,
    grassland_name,
    dominant_sector = sector,
    dominant_sector_cn = sector_cn,
    dominant_abs_contribution = abs_contribution_percent,
    dominant_signed_contribution = signed_contribution_percent,
    dominant_correlation = correlation,
    dominant_direction = direction
  )

save_csv(
  dominant_result,
  file.path(out_dir, "06_dominant_sector_by_grassland_type.csv")
)

print(dominant_result)

# ============================================================
# 13. 单独输出牲畜部门结果
# ============================================================

livestock_result <- cov_result |>
  filter(sector %in% c("Enteric", "Livestock")) |>
  arrange(desc(abs_contribution_percent))

save_csv(
  livestock_result,
  file.path(out_dir, "08_livestock_contribution_by_grassland_type.csv")
)

print(livestock_result)

# ============================================================
# 14. 作图：带符号协方差
# ============================================================

p_cov <- ggplot(
  cov_result,
  aes(x = covariance, y = sector_cn)
) +
  geom_col(width = 0.7, fill = "grey35") +
  geom_vline(xintercept = 0, linewidth = 0.4) +
  facet_wrap(~ grassland_name, scales = "free_x") +
  labs(
    x = expression(paste("与 ", Delta, "GR 的协方差")),
    y = "排放部门",
    title = expression(paste("不同草地类型中各部门排放变化与 ", Delta, "GR 的带符号协方差"))
  ) +
  theme_bw(base_size = 12, base_family = cn_font) +
  theme(
    strip.text = element_text(size = 11),
    axis.text.y = element_text(size = 9),
    plot.title = element_text(hjust = 0.5)
  )

ggsave(
  filename = file.path(out_dir, "05_signed_covariance_by_grassland_type_free_x.png"),
  plot = p_cov,
  width = 12,
  height = 8,
  dpi = 300
)

# ============================================================
# 15. 作图：绝对相对协变强度
# ============================================================

p_abs <- ggplot(
  cov_result,
  aes(x = sector_cn, y = abs_contribution_percent)
) +
  geom_col(width = 0.7, fill = "grey35") +
  coord_flip() +
  facet_wrap(~ grassland_name, scales = "free_y") +
  labs(
    x = "排放部门",
    y = "绝对相对协变强度 (%)",
    title = expression(paste("不同草地类型中各部门与 ", Delta, "GR 的绝对相对协变强度"))
  ) +
  theme_bw(base_size = 12, base_family = cn_font) +
  theme(
    strip.text = element_text(size = 11),
    axis.text.y = element_text(size = 9),
    plot.title = element_text(hjust = 0.5)
  )

ggsave(
  filename = file.path(out_dir, "06_abs_relative_covariance_strength_by_grassland_type.png"),
  plot = p_abs,
  width = 12,
  height = 8,
  dpi = 300
)

# 兼容旧文件名
ggsave(
  filename = file.path(out_dir, "06_contribution_percent_by_grassland_type.png"),
  plot = p_abs,
  width = 12,
  height = 8,
  dpi = 300
)

# ============================================================
# 16. 作图：带符号相对协变强度
# ============================================================

p_signed <- ggplot(
  cov_result,
  aes(x = sector_cn, y = signed_contribution_percent)
) +
  geom_col(width = 0.7, fill = "grey35") +
  geom_hline(yintercept = 0, linewidth = 0.4) +
  coord_flip() +
  facet_wrap(~ grassland_name, scales = "fixed") +
  labs(
    x = "排放部门",
    y = "带符号相对协变强度 (%)",
    title = expression(paste("不同草地类型中各部门与 ", Delta, "GR 的带符号相对协变强度"))
  ) +
  theme_bw(base_size = 12, base_family = cn_font) +
  theme(
    strip.text = element_text(size = 11),
    axis.text.y = element_text(size = 9),
    plot.title = element_text(hjust = 0.5)
  )

ggsave(
  filename = file.path(out_dir, "07_signed_relative_covariance_strength_by_grassland_type.png"),
  plot = p_signed,
  width = 12,
  height = 8,
  dpi = 300
)

# ============================================================
# 17. 作图：100% 堆叠绝对相对协变强度
# ============================================================

p_stack <- ggplot(
  cov_result,
  aes(x = grassland_name, y = abs_contribution_percent, fill = sector_cn)
) +
  geom_col(width = 0.7) +
  labs(
    x = "草地类型",
    y = "绝对相对协变强度 (%)",
    fill = "排放部门",
    title = expression(paste("不同草地类型中各部门与 ", Delta, "GR 的绝对相对协变强度构成"))
  ) +
  theme_bw(base_size = 12, base_family = cn_font) +
  theme(
    axis.text.x = element_text(angle = 35, hjust = 1),
    legend.position = "right",
    plot.title = element_text(hjust = 0.5)
  )

ggsave(
  filename = file.path(out_dir, "07_stacked_contribution_by_grassland_type.png"),
  plot = p_stack,
  width = 11,
  height = 6,
  dpi = 300
)

ggsave(
  filename = file.path(out_dir, "09_stacked_abs_relative_covariance_strength_by_grassland_type.png"),
  plot = p_stack,
  width = 11,
  height = 6,
  dpi = 300
)

# ============================================================
# 18. 作图：相关系数热图
# ============================================================

p_corr <- ggplot(
  cov_result,
  aes(x = sector_cn, y = grassland_name, fill = correlation)
) +
  geom_tile(color = "white", linewidth = 0.4) +
  scale_fill_gradient2(
    low = "#2166AC",
    mid = "white",
    high = "#B2182B",
    midpoint = 0,
    limits = c(-1, 1),
    na.value = "grey90"
  ) +
  labs(
    x = "排放部门",
    y = "草地类型",
    fill = "相关系数",
    title = expression(paste("不同草地类型中各部门排放变化与 ", Delta, "GR 的相关方向"))
  ) +
  theme_bw(base_size = 12, base_family = cn_font) +
  theme(
    axis.text.x = element_text(angle = 35, hjust = 1),
    plot.title = element_text(hjust = 0.5)
  )

ggsave(
  filename = file.path(out_dir, "10_correlation_heatmap_by_grassland_type.png"),
  plot = p_corr,
  width = 9,
  height = 5.5,
  dpi = 300
)

# ============================================================
# 19. 输出完成
# ============================================================

message("计算完成！结果保存到：")
message(out_dir)

message("主要输出：")
message("00_check_grassland_type_mapping.csv")
message("01_deltaGR_by_grassland_type.csv")
message("02_sector_deltaE_by_grassland_type.csv")
message("03_covariance_input_by_grassland_type.csv")
message("04_covariance_response_by_grassland_type.csv")
message("05_response_index_wide_by_grassland_type.csv")
message("06_dominant_sector_by_grassland_type.csv")
message("07_signed_relative_covariance_strength_by_grassland_type.png")
message("09_stacked_abs_relative_covariance_strength_by_grassland_type.png")
message("10_correlation_heatmap_by_grassland_type.png")