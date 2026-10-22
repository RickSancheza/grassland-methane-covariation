# ============================================================
# ΔGR 与牲畜排放变化关系
# 每个草地类型每年一个点
# ΔGR：面积加权平均值
# 牲畜排放变化：区域总量，ton CH4 / cell / yr -> Tg CH4 / yr
# 输出：数据表、回归结果表、PDF 图
# ============================================================

library(terra)
library(dplyr)
library(stringr)
library(ggplot2)
library(readr)

# ============================================================
# 1. 中文字体设置
# ============================================================

try(Sys.setlocale("LC_CTYPE", "Chinese"), silent = TRUE)

if (.Platform$OS.type == "windows") {
  windowsFonts(SimHei = windowsFont("SimHei"))
  cn_font <- "SimHei"
} else {
  cn_font <- "sans"
}

save_csv <- function(x, path) {
  readr::write_excel_csv(x, path)
}

# ============================================================
# 2. 路径设置：按你现在的数据
# ============================================================

# ΔGR 文件夹
gr_dir <- enc2native("F:/DOCTOR/graduating/test2/40/0_data/1_Delta_GR")

# 牲畜排放变化文件夹
livestock_dir <- enc2native("F:/DOCTOR/graduating/test2/40/0_data/3_分部门变化/Enteric")

# 草地类型图
grassland_type_file <- enc2native("F:/DOCTOR/graduating/test2/40/0_data/grasslandtype.tif")

# 输出文件夹
out_dir <- enc2native("F:/DOCTOR/graduating/test2/40/2_secondpart/1_线性关系")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# 年份
start_year <- 2005
end_year   <- 2020

# ============================================================
# 3. 牲畜排放单位设置
# ============================================================

# 如果 ΔEnteric 单位是 ton CH4 / cell / yr，用 1e6
# 如果是 kg CH4 / cell / yr，改成 1e9
# 如果已经是 Tg CH4 / yr，改成 1
emission_to_Tg_divisor <- 1e6

# 如果排放栅格是每个像元的年排放量变化，FALSE
# 如果是 kg m-2 yr-1 这类通量，TRUE
emission_is_flux <- FALSE

# ============================================================
# 4. 草地类型名称
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

# ============================================================
# 5. 工具函数
# ============================================================

get_year_files <- function(folder, start_year, end_year) {
  
  folder <- enc2native(folder)
  
  cat("\n========== 扫描文件夹 ==========\n")
  cat("folder = ", folder, "\n")
  cat("dir.exists = ", dir.exists(folder), "\n")
  
  if (!dir.exists(folder)) {
    stop(paste0("文件夹不存在：", folder))
  }
  
  files <- list.files(
    folder,
    pattern = "\\.tif$",
    full.names = TRUE,
    recursive = FALSE,
    ignore.case = TRUE
  )
  
  files <- files[!grepl("\\.ovr$", files, ignore.case = TRUE)]
  files <- files[!grepl("_backup", files, ignore.case = TRUE)]
  
  cat("找到 tif 数量：", length(files), "\n")
  
  if (length(files) == 0) {
    stop(paste0("没有找到 tif 文件：", folder))
  }
  
  years <- str_extract(basename(files), "(19\\d{2}|20\\d{2})") |> as.integer()
  
  df <- data.frame(
    year = years,
    file = normalizePath(files, winslash = "/", mustWork = TRUE),
    stringsAsFactors = FALSE
  ) |>
    filter(!is.na(year)) |>
    filter(year >= start_year, year <= end_year) |>
    arrange(year)
  
  cat("\n识别到的年份文件：\n")
  print(df)
  
  if (nrow(df) == 0) {
    stop(paste0("找到 tif，但没有识别到 ", start_year, "—", end_year, " 年文件：", folder))
  }
  
  missing_years <- setdiff(start_year:end_year, df$year)
  
  if (length(missing_years) > 0) {
    stop(paste0(
      "缺少年份：",
      paste(missing_years, collapse = ", "),
      "\n路径：",
      folder
    ))
  }
  
  return(df)
}

read_align <- function(file, template, method = "bilinear") {
  
  if (length(file) == 0 || is.na(file) || file == "") {
    stop("传入 read_align() 的 file 是空值。请检查年份文件表。")
  }
  
  file <- normalizePath(file, winslash = "/", mustWork = TRUE)
  
  if (!file.exists(file)) {
    stop(paste0("文件不存在：", file))
  }
  
  r <- rast(file)
  
  if (!same.crs(r, template)) {
    r <- project(r, template, method = method)
  }
  
  if (!compareGeom(r, template, stopOnError = FALSE)) {
    r <- resample(r, template, method = method)
  }
  
  return(r)
}

calc_gr_mean_by_type <- function(file, template, grass_type, type_value, area_km2) {
  
  r <- read_align(file, template, method = "bilinear")
  r <- mask(r, template)
  
  m <- ifel(grass_type == type_value, 1, NA)
  r_m <- r * m
  
  num <- global(r_m * area_km2, "sum", na.rm = TRUE)[1, 1]
  den <- global(mask(area_km2 * m, r_m), "sum", na.rm = TRUE)[1, 1]
  
  if (is.na(den) || den == 0) {
    return(NA_real_)
  }
  
  return(num / den)
}

calc_livestock_sum_by_type <- function(file, template, grass_type, type_value) {
  
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

format_p <- function(p) {
  ifelse(
    is.na(p), "NA",
    ifelse(p < 0.001, "<0.001", sprintf("%.3f", p))
  )
}

format_num <- function(x, digits = 3) {
  sprintf(paste0("%.", digits, "f"), x)
}

safe_lm <- function(df) {
  
  df <- df |>
    filter(
      !is.na(deltaGR_mean),
      !is.na(livestock_Tg)
    )
  
  if (nrow(df) < 3) {
    return(data.frame(
      n = nrow(df),
      slope = NA_real_,
      intercept = NA_real_,
      r2 = NA_real_,
      p_value = NA_real_
    ))
  }
  
  if (length(unique(df$livestock_Tg)) < 2) {
    return(data.frame(
      n = nrow(df),
      slope = NA_real_,
      intercept = NA_real_,
      r2 = NA_real_,
      p_value = NA_real_
    ))
  }
  
  fit <- lm(deltaGR_mean ~ livestock_Tg, data = df)
  sm <- summary(fit)
  
  data.frame(
    n = nrow(df),
    slope = coef(fit)[2],
    intercept = coef(fit)[1],
    r2 = sm$r.squared,
    p_value = sm$coefficients[2, 4]
  )
}

# ============================================================
# 6. 读取文件表、模板、草地类型
# ============================================================

cat("\n========== 读取 ΔGR 文件 ==========\n")
gr_files <- get_year_files(gr_dir, start_year, end_year)

cat("\n========== 读取 Enteric 文件 ==========\n")
livestock_files <- get_year_files(livestock_dir, start_year, end_year)

cat("\n========== gr_files ==========\n")
print(gr_files)

cat("\n========== livestock_files ==========\n")
print(livestock_files)

template <- rast(gr_files$file[gr_files$year == start_year][1])

cat("\n========== 模板信息 ==========\n")
print(template)

if (!file.exists(grassland_type_file)) {
  stop(paste0("草地类型文件不存在：", grassland_type_file))
}

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
grass_values <- grass_values[grass_values != 0]

cat("\n========== 识别到的草地类型 ==========\n")
grass_check <- data.frame(
  grassland_type = grass_values,
  grassland_name = grassland_labels[as.character(grass_values)]
)
print(grass_check)

save_csv(
  grass_check,
  file.path(out_dir, "00_check_grassland_type_mapping.csv")
)

# 面积只算一次，提高速度
area_km2 <- cellSize(template, unit = "km")

# ============================================================
# 7. 计算每个草地类型、每年的 ΔGR 面积加权平均值
# ============================================================

gr_records <- list()
idx <- 1

for (g in grass_values) {
  
  message("正在计算 ΔGR：草地类型 ", g)
  
  for (i in seq_len(nrow(gr_files))) {
    
    yr <- gr_files$year[i]
    f  <- gr_files$file[i]
    
    val <- calc_gr_mean_by_type(
      file = f,
      template = template,
      grass_type = grass_type,
      type_value = g,
      area_km2 = area_km2
    )
    
    gr_records[[idx]] <- data.frame(
      grassland_type = g,
      grassland_name = grassland_labels[as.character(g)],
      year = yr,
      deltaGR_mean = val
    )
    
    idx <- idx + 1
  }
}

gr_df <- bind_rows(gr_records) |>
  mutate(
    grassland_name = factor(grassland_name, levels = grassland_order)
  )

save_csv(
  gr_df,
  file.path(out_dir, "01_deltaGR_by_grassland_type.csv")
)

cat("\n========== ΔGR 年序列 ==========\n")
print(gr_df)

# ============================================================
# 8. 计算每个草地类型、每年的牲畜排放变化总量
# ============================================================

livestock_records <- list()
idx <- 1

for (g in grass_values) {
  
  message("正在计算牲畜排放：草地类型 ", g)
  
  for (i in seq_len(nrow(livestock_files))) {
    
    yr <- livestock_files$year[i]
    f  <- livestock_files$file[i]
    
    val <- calc_livestock_sum_by_type(
      file = f,
      template = template,
      grass_type = grass_type,
      type_value = g
    )
    
    livestock_records[[idx]] <- data.frame(
      grassland_type = g,
      grassland_name = grassland_labels[as.character(g)],
      year = yr,
      livestock_Tg = val
    )
    
    idx <- idx + 1
  }
}

livestock_df <- bind_rows(livestock_records) |>
  mutate(
    grassland_name = factor(grassland_name, levels = grassland_order)
  )

save_csv(
  livestock_df,
  file.path(out_dir, "02_livestock_delta_by_grassland_type.csv")
)

cat("\n========== 牲畜排放年序列 ==========\n")
print(livestock_df)

# ============================================================
# 9. 合并数据
# ============================================================

plot_df <- gr_df |>
  left_join(
    livestock_df,
    by = c("grassland_type", "grassland_name", "year")
  ) |>
  filter(
    !is.na(deltaGR_mean),
    !is.na(livestock_Tg)
  ) |>
  mutate(
    grassland_name = factor(grassland_name, levels = grassland_order)
  )

save_csv(
  plot_df,
  file.path(out_dir, "03_linear_relationship_input.csv")
)

cat("\n========== 线性关系输入表 ==========\n")
print(plot_df)

# ============================================================
# 10. 每个草地类型计算线性回归
# ============================================================

lm_result <- plot_df |>
  group_by(grassland_type, grassland_name) |>
  group_modify(~ safe_lm(.x)) |>
  ungroup() |>
  mutate(
    equation = case_when(
      is.na(slope) ~ "Insufficient data",
      intercept >= 0 ~ paste0(
        "y = ", format_num(slope, 3), "x + ", format_num(intercept, 3),
        "\nR² = ", format_num(r2, 3),
        "\nP = ", format_p(p_value)
      ),
      intercept < 0 ~ paste0(
        "y = ", format_num(slope, 3), "x - ", format_num(abs(intercept), 3),
        "\nR² = ", format_num(r2, 3),
        "\nP = ", format_p(p_value)
      )
    )
  )

save_csv(
  lm_result,
  file.path(out_dir, "04_lm_result_by_grassland_type.csv")
)

cat("\n========== 回归结果 ==========\n")
print(lm_result)

# ============================================================
# 11. 绘图
# ============================================================

p <- ggplot(plot_df, aes(x = livestock_Tg, y = deltaGR_mean)) +
  geom_point(size = 2.4, alpha = 0.8) +
  geom_smooth(
    method = "lm",
    se = TRUE,
    linewidth = 0.8
  ) +
  geom_label(
    data = lm_result,
    aes(
      x = -Inf,
      y = Inf,
      label = equation
    ),
    inherit.aes = FALSE,
    hjust = -0.05,
    vjust = 1.08,
    size = 3.7,
    label.size = 0.25,
    fill = "white",
    family = cn_font
  ) +
  facet_wrap(~ grassland_name, scales = "free") +
  labs(
    x = expression(paste(Delta, "Livestock CH"[4], " emissions (Tg CH"[4], " yr"^-1, ")")),
    y = expression(paste(Delta, "CH"[4], " GR (ppb yr"^-1, ")")),
    title = expression(paste("Relationship between livestock CH"[4], " emission changes and ", Delta, "CH"[4], " GR"))
  ) +
  theme_bw(base_size = 12, base_family = cn_font) +
  theme(
    strip.text = element_text(size = 11),
    plot.title = element_text(hjust = 0.5)
  )

print(p)

# ============================================================
# 12. 保存 PDF 和 PNG
# ============================================================

out_pdf <- file.path(out_dir, "Livestock_deltaGR_relationship.pdf")
out_png <- file.path(out_dir, "Livestock_deltaGR_relationship.png")

ggsave(
  filename = out_pdf,
  plot = p,
  width = 10,
  height = 7,
  units = "in",
  device = cairo_pdf,
  limitsize = FALSE
)

ggsave(
  filename = out_png,
  plot = p,
  width = 10,
  height = 7,
  units = "in",
  dpi = 300,
  limitsize = FALSE
)

cat("\nPDF 已保存到：", out_pdf, "\n")
cat("PNG 已保存到：", out_png, "\n")
cat("全部完成！结果保存到：", out_dir, "\n")