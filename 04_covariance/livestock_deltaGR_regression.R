# Area-weighted Delta GR vs regional livestock emission change (R)
# Source lineage: recovered linear_relationship_deltaGR_livestock.R.
# Paths via GRASSLAND_CH4_DATA_ROOT / GRASSLAND_CH4_OUTPUT_ROOT.
# Formulas unchanged: area-weighted mean DeltaGR; sum of Delta Enteric in Tg; lm.

suppressPackageStartupMessages({
  library(terra)
  library(dplyr)
  library(stringr)
  library(ggplot2)
  library(readr)
})

data_root <- Sys.getenv("GRASSLAND_CH4_DATA_ROOT", unset = file.path(getwd(), "data"))
out_root <- Sys.getenv("GRASSLAND_CH4_OUTPUT_ROOT", unset = file.path(getwd(), "results"))
gr_dir <- file.path(data_root, "delta_gr")
livestock_dir <- file.path(data_root, "emission_changes", "Livestock")
if (!dir.exists(livestock_dir)) {
  livestock_dir <- file.path(data_root, "emission_changes", "Enteric")
}
grass_file <- file.path(data_root, "grasslandtype.tif")
out_dir <- file.path(out_root, "04_covariance", "livestock_deltaGR")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

start_year <- 2005
end_year <- 2020
emission_to_Tg_divisor <- 1e6

grassland_labels <- c(
  "1" = "tropical_subtropical_grassland",
  "2" = "temperate_grassland",
  "3" = "flooded_wet_grassland",
  "4" = "montane_grassland",
  "5" = "tundra",
  "6" = "arid_desert_grassland"
)

get_year_files <- function(folder) {
  files <- list.files(folder, pattern = "\\.tif$", full.names = TRUE)
  years <- as.integer(str_extract(basename(files), "\\d{4}"))
  df <- data.frame(year = years, file = files, stringsAsFactors = FALSE) %>%
    filter(!is.na(year), year >= start_year, year <= end_year) %>%
    arrange(year)
  missing <- setdiff(start_year:end_year, df$year)
  if (length(missing) > 0) stop("Missing years: ", paste(missing, collapse = ", "))
  df
}

read_align <- function(file, template, method = "bilinear") {
  r <- rast(file)
  if (!same.crs(r, template)) r <- project(r, template, method = method)
  if (!compareGeom(r, template, stopOnError = FALSE)) r <- resample(r, template, method = method)
  r
}

safe_lm <- function(df) {
  df <- df %>% filter(!is.na(deltaGR_mean), !is.na(livestock_Tg))
  if (nrow(df) < 3 || length(unique(df$livestock_Tg)) < 2) {
    return(data.frame(n = nrow(df), slope = NA_real_, intercept = NA_real_,
                      r2 = NA_real_, p_value = NA_real_))
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

gr_files <- get_year_files(gr_dir)
liv_files <- get_year_files(livestock_dir)
template <- rast(gr_files$file[1])
grass_type <- rast(grass_file)
if (!same.crs(grass_type, template)) grass_type <- project(grass_type, template, method = "near")
if (!compareGeom(grass_type, template, stopOnError = FALSE)) {
  grass_type <- resample(grass_type, template, method = "near")
}
grass_type <- mask(grass_type, template)
area_km2 <- cellSize(template, unit = "km")
grass_values <- sort(na.omit(unique(values(grass_type, mat = FALSE))))
grass_values <- grass_values[grass_values != 0]

gr_records <- list()
idx <- 1
for (g in grass_values) {
  for (i in seq_len(nrow(gr_files))) {
    r <- mask(read_align(gr_files$file[i], template), template)
    m <- ifel(grass_type == g, 1, NA)
    r_m <- r * m
    num <- global(r_m * area_km2, "sum", na.rm = TRUE)[1, 1]
    den <- global(mask(area_km2 * m, r_m), "sum", na.rm = TRUE)[1, 1]
    val <- if (is.na(den) || den == 0) NA_real_ else num / den
    gr_records[[idx]] <- data.frame(
      grassland_type = g,
      grassland_name = unname(grassland_labels[as.character(g)]),
      year = gr_files$year[i],
      deltaGR_mean = val
    )
    idx <- idx + 1
  }
}
gr_df <- bind_rows(gr_records)

liv_records <- list()
idx <- 1
for (g in grass_values) {
  for (i in seq_len(nrow(liv_files))) {
    r <- mask(read_align(liv_files$file[i], template), template)
    m <- ifel(grass_type == g, 1, NA)
    total_Tg <- global(r * m, "sum", na.rm = TRUE)[1, 1] / emission_to_Tg_divisor
    liv_records[[idx]] <- data.frame(
      grassland_type = g,
      grassland_name = unname(grassland_labels[as.character(g)]),
      year = liv_files$year[i],
      livestock_Tg = total_Tg
    )
    idx <- idx + 1
  }
}
liv_df <- bind_rows(liv_records)

plot_df <- gr_df %>%
  left_join(liv_df, by = c("grassland_type", "grassland_name", "year")) %>%
  filter(!is.na(deltaGR_mean), !is.na(livestock_Tg))

lm_result <- plot_df %>%
  group_by(grassland_type, grassland_name) %>%
  group_modify(~ safe_lm(.x)) %>%
  ungroup()

write_excel_csv(plot_df, file.path(out_dir, "linear_relationship_input.csv"))
write_excel_csv(lm_result, file.path(out_dir, "lm_result_by_grassland_type.csv"))

p <- ggplot(plot_df, aes(x = livestock_Tg, y = deltaGR_mean)) +
  geom_point(size = 2.2, alpha = 0.8) +
  geom_smooth(method = "lm", se = TRUE, linewidth = 0.8) +
  facet_wrap(~ grassland_name, scales = "free") +
  labs(
    x = "Delta livestock CH4 (Tg CH4 / yr)",
    y = "Delta CH4 GR (ppb / yr)"
  ) +
  theme_bw(base_size = 11)

ggsave(file.path(out_dir, "livestock_deltaGR_relationship.pdf"), p, width = 10, height = 7)
ggsave(file.path(out_dir, "livestock_deltaGR_relationship.png"), p, width = 10, height = 7, dpi = 300)
message("Wrote ", out_dir)
