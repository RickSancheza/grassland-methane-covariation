# Grassland-type sector covariance vs Delta GR (R)
# Source lineage: recovered covariance_by_grassland_type.R; English paths via
# environment variables GRASSLAND_CH4_DATA_ROOT / GRASSLAND_CH4_OUTPUT_ROOT.
# Formulas unchanged: cov, cor, abs% = |cov|/sum|cov|*100, signed% = cov/sum|cov|*100.

suppressPackageStartupMessages({
  library(terra)
  library(dplyr)
  library(stringr)
  library(readr)
})

# ---- paths from environment (no drive letters) ----
data_root <- Sys.getenv("GRASSLAND_CH4_DATA_ROOT", unset = file.path(getwd(), "data"))
out_root <- Sys.getenv("GRASSLAND_CH4_OUTPUT_ROOT", unset = file.path(getwd(), "results"))
gr_dir <- file.path(data_root, "delta_gr")
sector_root <- file.path(data_root, "emission_changes")
grass_file <- file.path(data_root, "grasslandtype.tif")
out_dir <- file.path(out_root, "04_covariance", "by_grassland_type")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

start_year <- 2005
end_year <- 2020
emission_to_Tg_divisor <- 1e6
min_valid_years <- 5

# Grassland eco-type labels (codes 1-6 match grasslandtype.tif)
grassland_labels <- c(
  "1" = "tropical_subtropical_grassland",
  "2" = "temperate_grassland",
  "3" = "flooded_wet_grassland",
  "4" = "montane_grassland",
  "5" = "tundra",
  "6" = "arid_desert_grassland"
)

# Sector folder aliases (Enteric used in some archives)
sector_label <- function(folder_name) {
  if (tolower(folder_name) %in% c("enteric", "livestock")) return("Livestock")
  folder_name
}

get_year_files <- function(folder) {
  files <- list.files(folder, pattern = "\\.tif$", full.names = TRUE)
  years <- as.integer(str_extract(basename(files), "\\d{4}"))
  data.frame(year = years, file = files, stringsAsFactors = FALSE) %>%
    filter(!is.na(year), year >= start_year, year <= end_year) %>%
    arrange(year)
}

read_align <- function(file, template, method = "bilinear") {
  r <- rast(file)
  if (!same.crs(r, template)) r <- project(r, template, method = method)
  if (!compareGeom(r, template, stopOnError = FALSE)) r <- resample(r, template, method = method)
  r
}

calc_gr_mean_by_type <- function(file, template, grass_type, type_value) {
  r <- mask(read_align(file, template), template)
  m <- ifel(grass_type == type_value, 1, NA)
  r_m <- r * m
  area_km2 <- cellSize(template, unit = "km")
  num <- global(r_m * area_km2, "sum", na.rm = TRUE)[1, 1]
  den <- global(mask(area_km2 * m, r_m), "sum", na.rm = TRUE)[1, 1]
  if (is.na(den) || den == 0) return(NA_real_)
  num / den
}

calc_emission_sum_by_type <- function(file, template, grass_type, type_value) {
  r <- mask(read_align(file, template), template)
  m <- ifel(grass_type == type_value, 1, NA)
  total <- global(r * m, "sum", na.rm = TRUE)[1, 1]
  total / emission_to_Tg_divisor
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

gr_files <- get_year_files(gr_dir)
stopifnot(nrow(gr_files) > 0)
template <- rast(gr_files$file[1])
grass_type <- rast(grass_file)
if (!same.crs(grass_type, template)) grass_type <- project(grass_type, template, method = "near")
if (!compareGeom(grass_type, template, stopOnError = FALSE)) {
  grass_type <- resample(grass_type, template, method = "near")
}
grass_type <- mask(grass_type, template)

grass_values <- sort(na.omit(unique(values(grass_type, mat = FALSE))))
grass_values <- grass_values[grass_values != 0]

gr_series <- bind_rows(lapply(grass_values, function(g) {
  data.frame(
    grassland_type = g,
    grassland_name = unname(grassland_labels[as.character(g)]),
    year = gr_files$year,
    deltaGR = vapply(gr_files$file, calc_gr_mean_by_type, numeric(1),
                     template = template, grass_type = grass_type, type_value = g)
  )
}))

sector_dirs <- list.dirs(sector_root, recursive = FALSE, full.names = TRUE)
emission_series <- bind_rows(lapply(sector_dirs, function(sdir) {
  files <- get_year_files(sdir)
  sec <- sector_label(basename(sdir))
  bind_rows(lapply(grass_values, function(g) {
    data.frame(
      grassland_type = g,
      grassland_name = unname(grassland_labels[as.character(g)]),
      sector = sec,
      year = files$year,
      deltaE_Tg = vapply(files$file, calc_emission_sum_by_type, numeric(1),
                         template = template, grass_type = grass_type, type_value = g)
    )
  }))
}))

analysis_df <- emission_series %>%
  left_join(gr_series %>% select(grassland_type, year, deltaGR),
            by = c("grassland_type", "year")) %>%
  filter(is.finite(deltaE_Tg), is.finite(deltaGR))

cov_result <- analysis_df %>%
  group_by(grassland_type, grassland_name, sector) %>%
  summarise(
    n_years = n(),
    covariance = safe_cov(deltaE_Tg, deltaGR, min_valid_years),
    correlation = safe_cor(deltaE_Tg, deltaGR, min_valid_years),
    .groups = "drop"
  ) %>%
  mutate(abs_covariance = abs(covariance)) %>%
  group_by(grassland_type, grassland_name) %>%
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
  ) %>%
  ungroup()

write_excel_csv(cov_result, file.path(out_dir, "covariance_by_grassland_type.csv"))
write_excel_csv(
  analysis_df,
  file.path(out_dir, "covariance_input_by_grassland_type.csv")
)
message("Wrote ", out_dir)
